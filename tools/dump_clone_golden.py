#!/usr/bin/env python3
"""Dump a matched end-to-end conditioned fixture from the PyTorch runtime.

The reference WAV is optional so the same dumper can cover text-only,
caption-only, clone, and clone+caption paths with identical capture semantics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

import irodori_tts.inference_runtime as inference_runtime  # noqa: E402
import irodori_tts.rf as rf  # noqa: E402
from irodori_tts.inference_runtime import (  # noqa: E402
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    get_cached_runtime,
)

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_TEXT = "こんにちは、色とりどりの世界へようこそ。"


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().contiguous().numpy()


def _write_f32(path: Path, tensor: torch.Tensor) -> list[int]:
    value = _array(tensor)
    value.tofile(path)
    return list(value.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.steps <= 0 or args.threads <= 0:
        parser.error("--steps and --threads must be positive")
    reference_path = args.ref.resolve() if args.ref is not None else None
    if reference_path is not None and not reference_path.is_file():
        parser.error(f"reference WAV does not exist: {reference_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    runtime, _ = get_cached_runtime(
        RuntimeKey(
            checkpoint=checkpoint,
            model_device="cpu",
            codec_repo=CODEC_REPO,
            model_precision="fp32",
            codec_device="cpu",
            codec_precision="fp32",
        )
    )

    captured: dict[str, torch.Tensor] = {}
    original_load_ref = runtime._load_reference_latent
    original_encode = runtime.model.encode_conditions
    original_duration = runtime.model.predict_duration_log_frames
    original_forward = runtime.model.forward_with_encoded_conditions
    original_sample = inference_runtime.sample_euler_rf_cfg
    original_decode = runtime.codec.decode_latent
    velocities: list[torch.Tensor] = []

    def load_ref_wrapper(*wrapper_args, **wrapper_kwargs):
        latent, mask = original_load_ref(*wrapper_args, **wrapper_kwargs)
        if latent is not None:
            captured["reference_latent"] = latent
        if mask is not None:
            captured["reference_mask"] = mask
        return latent, mask

    def encode_wrapper(*wrapper_args, **wrapper_kwargs):
        result = original_encode(*wrapper_args, **wrapper_kwargs)
        captured["text_state"] = result[0]
        captured["text_mask"] = result[1]
        if result[2] is not None:
            captured["speaker_state"] = result[2]
        if result[3] is not None:
            captured["speaker_mask"] = result[3]
        if result[4] is not None:
            captured["caption_state"] = result[4]
        if result[5] is not None:
            captured["caption_mask"] = result[5]
        return result

    def duration_wrapper(*wrapper_args, **wrapper_kwargs):
        result = original_duration(*wrapper_args, **wrapper_kwargs)
        captured["duration_log_frames"] = result
        return result

    def sample_wrapper(*wrapper_args, **wrapper_kwargs):
        model = wrapper_kwargs["model"]
        batch = int(wrapper_kwargs["text_input_ids"].shape[0])
        sequence = int(wrapper_kwargs["sequence_length"])
        generator, generator_device = rf._make_rng(
            seed=int(wrapper_kwargs["seed"]), device=model.device
        )
        noise = torch.randn(
            (batch, sequence, model.cfg.patched_latent_dim),
            device=generator_device,
            dtype=model.dtype,
            generator=generator,
        )
        if generator_device != model.device:
            noise = noise.to(device=model.device)
        captured["noise"] = noise
        result = original_sample(*wrapper_args, **wrapper_kwargs)
        captured["latent_patched"] = result
        return result

    def forward_wrapper(*wrapper_args, **wrapper_kwargs):
        result = original_forward(*wrapper_args, **wrapper_kwargs)
        velocities.append(result.detach().cpu())
        return result

    def decode_wrapper(latent: torch.Tensor):
        captured["latent_final"] = latent
        return original_decode(latent)

    runtime._load_reference_latent = load_ref_wrapper
    runtime.model.encode_conditions = encode_wrapper
    runtime.model.predict_duration_log_frames = duration_wrapper
    runtime.model.forward_with_encoded_conditions = forward_wrapper
    inference_runtime.sample_euler_rf_cfg = sample_wrapper
    runtime.codec.decode_latent = decode_wrapper
    try:
        result = runtime.synthesize(
            SamplingRequest(
                text=args.text,
                caption=args.caption,
                ref_wav=None if reference_path is None else str(reference_path),
                no_ref=reference_path is None,
                num_steps=args.steps,
                seed=args.seed,
                cfg_guidance_mode="independent",
                cfg_scale_text=3.0,
                cfg_scale_caption=3.0 if args.caption else 0.0,
                cfg_scale_speaker=5.0 if reference_path is not None else 0.0,
            ),
            log_fn=print,
        )
    finally:
        runtime._load_reference_latent = original_load_ref
        runtime.model.encode_conditions = original_encode
        runtime.model.predict_duration_log_frames = original_duration
        runtime.model.forward_with_encoded_conditions = original_forward
        inference_runtime.sample_euler_rf_cfg = original_sample
        runtime.codec.decode_latent = original_decode

    required = ["duration_log_frames", "noise", "latent_final"]
    if reference_path is not None:
        required.extend(["reference_latent", "speaker_state"])
    if args.caption:
        required.append("caption_state")
    missing = [name for name in required if name not in captured]
    if missing:
        raise RuntimeError(f"failed to capture: {', '.join(missing)}")

    shapes: dict[str, list[int]] = {}
    for name in required:
        shapes[name] = _write_f32(args.out_dir / f"{name}.f32", captured[name])
    waveform = result.audios[0].reshape(-1)
    shapes["waveform"] = _write_f32(args.out_dir / "waveform.f32", waveform)
    for step, velocity in enumerate(velocities):
        shapes[f"velocity_step{step:03d}"] = _write_f32(
            args.out_dir / f"velocity_step{step:03d}.f32", velocity
        )
    metadata = {
        "text": args.text,
        "caption": args.caption,
        "reference": None if reference_path is None else str(reference_path),
        "seed": args.seed,
        "steps": args.steps,
        "sample_rate": result.sample_rate,
        "shapes": shapes,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"dumped matched clone fixture to {args.out_dir}")
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
