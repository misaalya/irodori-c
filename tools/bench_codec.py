#!/usr/bin/env python3
"""Matched warm PyTorch CPU benchmark for deterministic DACVAE decode."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from dacvae import DACVAE  # noqa: E402

DEFAULT_PTH = Path.home() / (
    ".cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/"
    "snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--weights", type=Path, default=DEFAULT_PTH)
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "irodori-c" / "golden" / "seed42_steps8",
    )
    args = parser.parse_args()
    if args.threads <= 0 or args.repeats <= 0:
        parser.error("--threads and --repeats must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    latent_tm = np.fromfile(
        args.golden / "codec_latent_final.f32", np.float32
    ).reshape(112, 32)
    latent = torch.from_numpy(latent_tm.T.copy()).unsqueeze(0)
    reference = torch.from_numpy(
        np.fromfile(args.golden / "codec_decoder_out.f32", np.float32).copy()
    ).reshape(1, 1, -1)
    model = DACVAE.load(str(args.weights)).eval()

    def decode() -> torch.Tensor:
        state = model.quantizer.out_proj(latent)
        for layer in model.decoder.model:
            state = layer(state)
        return model.decoder.wm_model.encoder_block.forward_no_conv(state)

    with torch.inference_mode():
        decode()
        samples: list[float] = []
        output = reference
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = decode()
            samples.append(time.perf_counter() - start)
    error = (output - reference).abs().max().item()
    print(
        f"PyTorch DACVAE fp32 threads={args.threads}: "
        f"median={statistics.median(samples):.3f} s "
        f"best={min(samples):.3f} s | max error {error:.6g}"
    )
    return 0 if error == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
