#!/usr/bin/env python3
"""Dump time-major DACVAE decoder checkpoints for the C implementation."""
from __future__ import annotations

import json
import sys
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


def load_f32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, np.float32).reshape(shape).copy())


def save_time_major(directory: Path, name: str, tensor: torch.Tensor) -> list[int]:
    time_major = tensor.detach().squeeze(0).transpose(0, 1).contiguous().float()
    time_major.numpy().tofile(directory / f"{name}.f32")
    shape = list(time_major.shape)
    print(f"  {name}: {shape}")
    return shape


def main() -> int:
    golden = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "irodori-c" / "golden" / "seed42_steps8"
    )
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PTH
    torch.set_num_threads(2)

    x_last = load_f32(golden / "x_t_step007.f32", (1, 112, 32))
    velocity_last = load_f32(golden / "velocity_step007.f32", (1, 112, 32))
    latent = x_last + velocity_last * (-0.999 / 8.0)
    latent.contiguous().numpy().tofile(golden / "codec_latent_final.f32")

    model = DACVAE.load(str(source)).eval()
    shapes: dict[str, list[int]] = {}
    with torch.inference_mode():
        state = model.quantizer.out_proj(latent.transpose(1, 2).contiguous())
        shapes["codec_quantizer_out"] = save_time_major(
            golden, "codec_quantizer_out", state
        )
        state = model.decoder.model[0](state)
        shapes["codec_decoder_initial"] = save_time_major(
            golden, "codec_decoder_initial", state
        )
        for index, stage in enumerate(model.decoder.model[1:]):
            state = stage(state)
            name = f"codec_decoder_stage{index}"
            shapes[name] = save_time_major(golden, name, state)
        output = model.decoder.wm_model.encoder_block.forward_no_conv(state)
        shapes["codec_decoder_out"] = save_time_major(
            golden, "codec_decoder_out_manual", output
        )

    reference = load_f32(golden / "codec_decoder_out.f32", tuple(output.shape))
    error = (output - reference).abs().max().item()
    print(f"  manual decoder vs synthesize hook: max error {error:.6g}")
    if error != 0.0:
        raise RuntimeError("manual decoder graph differs from runtime path")
    (golden / "codec_shapes.json").write_text(json.dumps(shapes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
