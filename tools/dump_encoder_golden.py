#!/usr/bin/env python3
"""Create a small deterministic per-stage golden for the DACVAE encoder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from dacvae import DACVAE

DEFAULT_PTH = Path.home() / (
    ".cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/"
    "snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth"
)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "golden" / "encoder_test"


def save(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().squeeze(0).transpose(0, 1).contiguous().cpu().numpy().tofile(path)


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PTH
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    model = DACVAE.load(source).eval()
    # Deliberately not divisible by hop_length=1920: this also gates the
    # right-side reflect padding used by the deterministic inference path.
    samples = 4000
    time = torch.arange(samples, dtype=torch.float32) / float(model.sample_rate)
    waveform = (0.31 * torch.sin(2 * torch.pi * 233 * time) +
                0.17 * torch.sin(2 * torch.pi * 3101 * time) +
                0.03 * torch.cos(2 * torch.pi * 47 * time)).reshape(1, 1, -1)
    waveform.numpy().reshape(-1).astype(np.float32).tofile(output / "waveform.f32")
    with torch.inference_mode():
        state = model._pad(waveform)  # noqa: SLF001 - exact runtime contract
        for index, layer in enumerate(model.encoder.block):
            state = layer(state)
            save(output / f"encoder_block{index}.f32", state)
        mean, _scale = model.quantizer.in_proj(state).chunk(2, dim=1)
        save(output / "encoder_mean.f32", mean)
    print(f"saved encoder golden: {output} ({samples} samples -> {mean.shape[-1]} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
