#!/usr/bin/env python3
"""Persistent warm PyTorch worker for the formal four-mode benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import wave
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
RESULT_PREFIX = "__IRO_RESULT__="


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--caption")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref", type=Path)
    args = parser.parse_args()
    if args.threads <= 0 or args.steps <= 0:
        parser.error("--threads and --steps must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint_candidate = Path(args.checkpoint).expanduser()
    checkpoint = (
        str(checkpoint_candidate.absolute())
        if checkpoint_candidate.is_file()
        else download_hf_checkpoint(args.checkpoint)
    )
    reference = args.ref.resolve() if args.ref is not None else None
    if reference is not None and not reference.is_file():
        parser.error(f"reference WAV does not exist: {reference}")

    key = RuntimeKey(
        checkpoint=checkpoint,
        model_device="cpu",
        codec_repo=CODEC_REPO,
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )
    runtime, _ = get_cached_runtime(key)
    request = SamplingRequest(
        text=args.text,
        caption=args.caption,
        ref_wav=None if reference is None else str(reference),
        no_ref=reference is None,
        num_steps=args.steps,
        seed=args.seed,
        cfg_guidance_mode="independent",
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0 if args.caption else 0.0,
        cfg_scale_speaker=0.0 if reference is None else 5.0,
    )

    def generate():
        return runtime.synthesize(request, log_fn=lambda _message: None)

    def generate_with_optional_noise(noise_out: Path | None):
        captured_noise: torch.Tensor | None = None
        original_sample = inference_runtime.sample_euler_rf_cfg

        def capture_sample(*wrapper_args, **wrapper_kwargs):
            nonlocal captured_noise
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
            captured_noise = noise[:1].detach().cpu().float().contiguous()
            return original_sample(*wrapper_args, **wrapper_kwargs)

        if noise_out is not None:
            inference_runtime.sample_euler_rf_cfg = capture_sample
        try:
            result = generate()
        finally:
            inference_runtime.sample_euler_rf_cfg = original_sample
        if noise_out is not None:
            if captured_noise is None:
                raise RuntimeError("failed to capture sampler noise")
            noise_out.parent.mkdir(parents=True, exist_ok=True)
            captured_noise.numpy().tofile(noise_out)
        return result

    print("__IRO_READY__", flush=True)
    with torch.inference_mode():
        for raw_line in sys.stdin:
            line = raw_line.rstrip("\r\n")
            if line == "QUIT":
                return 0
            fields = line.split("\t")
            if not fields or fields[0] not in {"WARM", "RUN"}:
                raise RuntimeError("expected WARM or RUN command")
            if fields[0] == "WARM":
                if len(fields) != 3:
                    raise RuntimeError("WARM expects output and noise paths")
                kind, output_text, noise_text = fields
                noise_out = Path(noise_text)
            else:
                if len(fields) != 2:
                    raise RuntimeError("RUN expects an output path")
                kind, output_text = fields
                noise_out = None

            begin = time.perf_counter()
            result = generate_with_optional_noise(noise_out)
            elapsed = time.perf_counter() - begin
            audio = result.audios[0].detach().cpu().float().reshape(-1).numpy()
            output = Path(output_text)
            write_wav(output, audio, result.sample_rate)
            stages = dict(result.stage_timings)
            sample_seconds = float(stages.get("sample_rf", 0.0))
            decode_seconds = float(stages.get("unpatchify_latent", 0.0)) + float(
                stages.get("decode_latent", 0.0)
            )
            encode_seconds = max(0.0, elapsed - sample_seconds - decode_seconds)
            payload = {
                "kind": kind,
                "elapsed_seconds": elapsed,
                "encode_seconds": encode_seconds,
                "sample_seconds": sample_seconds,
                "decode_seconds": decode_seconds,
                "output_samples": int(audio.size),
                "audio_seconds": float(audio.size / result.sample_rate),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "float32_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            }
            print(RESULT_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
