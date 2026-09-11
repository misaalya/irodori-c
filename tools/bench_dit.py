#!/usr/bin/env python3
"""Matched PyTorch CPU benchmark for the focused 8-step RF-DiT golden path."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from irodori_tts.inference_runtime import (  # noqa: E402
    RuntimeKey,
    download_hf_checkpoint,
    get_cached_runtime,
)

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


def load_f32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    array = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(array.reshape(shape).copy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "irodori-c" / "golden" / "seed42_steps8",
    )
    args = parser.parse_args()
    if args.threads <= 0 or args.steps <= 0:
        parser.error("--threads and --steps must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    checkpoint = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    key = RuntimeKey(
        checkpoint=checkpoint,
        model_device="cpu",
        codec_repo=CODEC_REPO,
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )
    print(f"[bench-dit] loading PyTorch runtime; threads={args.threads}")
    runtime, _ = get_cached_runtime(key)
    model = runtime.model
    golden = args.golden

    text_one = load_f32(golden / "text_state.f32", (1, 256, 512))
    token_count = (golden / "token_ids.i64").stat().st_size // 8
    text_cfg = torch.cat([text_one, torch.zeros_like(text_one)], dim=0)
    mask_cfg = torch.zeros((2, 256), dtype=torch.bool)
    mask_cfg[0, :token_count] = True
    mask_cond = mask_cfg[:1]
    speaker_cfg = torch.zeros((2, 1, 768), dtype=torch.float32)
    speaker_cond = speaker_cfg[:1]
    caption_cfg = torch.zeros((2, 1, 512), dtype=torch.float32)
    caption_cond = caption_cfg[:1]
    null_mask_cfg = torch.zeros((2, 1), dtype=torch.bool)
    null_mask_cond = null_mask_cfg[:1]

    with torch.inference_mode():
        cache_t0 = time.perf_counter()
        cache_cfg = model.build_context_kv_cache(text_cfg, speaker_cfg, caption_cfg)
        cache_cond = model.build_context_kv_cache(text_one, speaker_cond, caption_cond)
        cache_seconds = time.perf_counter() - cache_t0

        def forward_projected(step: int) -> tuple[torch.Tensor, float]:
            batch = 2 if step < 4 else 1
            x = load_f32(
                golden / f"in_proj_out_step{step:03d}.f32",
                (batch, 112, 1280),
            )
            cond = load_f32(
                golden / f"cond_embed_step{step:03d}.f32",
                (batch, 3840),
            )[:, None, :]
            if batch == 2:
                text, text_mask = text_cfg, mask_cfg
                speaker, caption = speaker_cfg, caption_cfg
                null_mask, cache = null_mask_cfg, cache_cfg
            else:
                text, text_mask = text_one, mask_cond
                speaker, caption = speaker_cond, caption_cond
                null_mask, cache = null_mask_cond, cache_cond
            freqs = model._rope_freqs(x.shape[1], x.device)
            start = time.perf_counter()
            for layer, block in enumerate(model.blocks):
                x = block(
                    x=x,
                    cond_embed=cond,
                    text_state=text,
                    text_mask=text_mask,
                    speaker_state=speaker,
                    speaker_mask=null_mask,
                    caption_state=caption,
                    caption_mask=null_mask,
                    freqs_cis=freqs,
                    context_kv=cache[layer],
                )
            velocity = model.out_proj(model.out_norm(x))
            return velocity, time.perf_counter() - start

        # Warm one CFG-shaped pass before measuring, matching the CBLAS warmup.
        forward_projected(0)
        total = 0.0
        worst = 0.0
        for step in range(args.steps):
            velocity, seconds = forward_projected(step)
            batch = velocity.shape[0]
            reference = load_f32(
                golden / f"velocity_step{step:03d}.f32",
                (batch, 112, 32),
            )
            error = (velocity - reference).abs().max().item()
            worst = max(worst, error)
            total += seconds
            print(
                f"  step {step:03d} batch={batch} velocity err {error:.6g} | "
                f"{seconds:.3f} s"
            )

    print(
        f"PyTorch fp32: CFG+cond caches {cache_seconds:.3f} s | "
        f"{args.steps} forwards {total:.3f} s | worst velocity err {worst:.6g}"
    )
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
