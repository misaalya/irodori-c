#!/usr/bin/env python3
"""Warm direct text-to-audio benchmark for the upstream PyTorch runtime.

Besides timing, this can export the exact initial Gaussian noise used by the
PyTorch sampler so the pure-C endpoint can be benchmarked with matched input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
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
DEFAULT_TEXT = "こんにちは、色とりどりの世界へようこそ。"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--caption")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default="Aratako/Irodori-TTS-v4.1-Small",
        help="HF repo id or an exact local model.safetensors path",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        help="optional reference WAV; enables the speaker-conditioned clone path",
    )
    parser.add_argument("--out", type=Path, help="optionally save the last result as PCM16 WAV")
    parser.add_argument(
        "--noise-out",
        type=Path,
        help="write the exact one-row initial sampler noise as raw float32",
    )
    parser.add_argument("--json-out", type=Path, help="write machine-readable benchmark metrics")
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "irodori-c" / "golden" / "seed42_steps8",
    )
    args = parser.parse_args()
    if args.threads <= 0 or args.steps <= 0 or args.repeats <= 0:
        parser.error("--threads, --steps, and --repeats must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint_candidate = Path(args.checkpoint).expanduser()
    checkpoint = (
        str(checkpoint_candidate.absolute())
        if checkpoint_candidate.is_file()
        else download_hf_checkpoint(args.checkpoint)
    )
    key = RuntimeKey(
        checkpoint=checkpoint,
        model_device="cpu",
        codec_repo=CODEC_REPO,
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )
    print(f"[python] load runtime; fp32 threads={args.threads}", flush=True)
    runtime, _ = get_cached_runtime(key)
    reference = args.ref.resolve() if args.ref is not None else None
    if reference is not None and not reference.is_file():
        parser.error(f"reference WAV does not exist: {reference}")
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

    if reference is not None and args.caption:
        mode = "clone+caption"
    elif reference is not None:
        mode = "clone"
    elif args.caption:
        mode = "caption-only"
    else:
        mode = "text-only"
    print(f"[python] warm-up direct {mode} -> audio", flush=True)

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

    with torch.inference_mode():
        if args.noise_out is not None:
            inference_runtime.sample_euler_rf_cfg = capture_sample
        try:
            generate()
        finally:
            inference_runtime.sample_euler_rf_cfg = original_sample
        if args.noise_out is not None:
            if captured_noise is None:
                raise RuntimeError("failed to capture sampler noise")
            args.noise_out.parent.mkdir(parents=True, exist_ok=True)
            captured_noise.numpy().tofile(args.noise_out)
        samples: list[float] = []
        hashes: list[str] = []
        result = None
        for run in range(args.repeats):
            begin = time.perf_counter()
            result = generate()
            elapsed = time.perf_counter() - begin
            samples.append(elapsed)
            run_audio = result.audios[0].detach().cpu().float().reshape(-1).numpy()
            hashes.append(hashlib.sha256(run_audio.tobytes()).hexdigest())
            print(f"  run {run + 1}: {elapsed:.3f} s", flush=True)

    assert result is not None
    audio = result.audios[0].detach().cpu().float().reshape(-1).numpy()
    golden_audio = None
    max_error = None
    if reference is None and args.caption is None:
        golden_audio = np.fromfile(args.golden / "waveform.f32", np.float32)
        compared = min(len(audio), len(golden_audio))
        max_error = float(np.max(np.abs(audio[:compared] - golden_audio[:compared])))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
        with wave.open(str(args.out), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(result.sample_rate)
            wav.writeframes(pcm.tobytes())
    duration = len(audio) / result.sample_rate
    median = statistics.median(samples)
    p95 = float(np.percentile(np.asarray(samples, dtype=np.float64), 95))
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stage_summary = ", ".join(
        f"{name}={seconds:.3f}s" for name, seconds in result.stage_timings
    )
    print(f"Python last-run stages: {stage_summary}", flush=True)
    print(
        f"Python direct {mode} fp32 threads={args.threads}: p50={median:.3f} s "
        f"p95={p95:.3f} s "
        f"best={min(samples):.3f} s | audio={duration:.3f} s "
        f"RTF={median / duration:.3f} | peak_rss_kib={rss_kib} | "
        f"samples={len(audio)}"
        + (f" max_error={max_error:.6g}" if max_error is not None else "")
        + (f" out={args.out}" if args.out is not None else ""),
        flush=True,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "python",
            "mode": mode,
            "checkpoint": checkpoint,
            "threads": args.threads,
            "steps": args.steps,
            "seed": args.seed,
            "text": args.text,
            "caption": args.caption,
            "reference": None if reference is None else str(reference),
            "samples_seconds": samples,
            "p50_seconds": median,
            "p95_seconds": p95,
            "best_seconds": min(samples),
            "audio_seconds": duration,
            "rtf_p50": median / duration,
            "peak_rss_kib": rss_kib,
            "sample_count": len(audio),
            "stage_timings": dict(result.stage_timings),
            "deterministic": len(set(hashes)) == 1,
            "float32_sha256": hashes[-1],
            "noise_path": None if args.noise_out is None else str(args.noise_out),
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if golden_audio is None:
        return 0
    return 0 if len(audio) == len(golden_audio) and max_error == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
