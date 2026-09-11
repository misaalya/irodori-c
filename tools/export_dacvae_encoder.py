#!/usr/bin/env python3
"""Export only the deterministic DACVAE reference-audio encoder graph."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

DEFAULT_PTH = Path.home() / (
    ".cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/"
    "snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth"
)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "weights" / "dacvae_encoder.safetensors"


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PTH
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    kwargs = payload.get("metadata", {}).get("kwargs", {})
    exported: dict[str, torch.Tensor] = {}

    def folded(base: str) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch._weight_norm(  # noqa: SLF001
            state[f"{base}.weight_v"].float(),
            state[f"{base}.weight_g"].float(),
            0,
        )
        return weight.permute(0, 2, 1).contiguous(), state[f"{base}.bias"].float().contiguous()

    def add_conv(base: str) -> None:
        weight, bias = folded(base)
        exported[f"{base}.weight"] = weight
        exported[f"{base}.bias"] = bias

    def add_alpha(name: str) -> None:
        exported[name] = state[name].float().contiguous()

    add_conv("encoder.block.0")
    for stage in range(1, 5):
        prefix = f"encoder.block.{stage}.block"
        for residual in range(3):
            residual_prefix = f"{prefix}.{residual}.block"
            add_alpha(f"{residual_prefix}.0.alpha")
            add_conv(f"{residual_prefix}.1")
            add_alpha(f"{residual_prefix}.2.alpha")
            add_conv(f"{residual_prefix}.3")
        add_alpha(f"{prefix}.3.alpha")
        add_conv(f"{prefix}.4")
    add_alpha("encoder.block.5.alpha")
    add_conv("encoder.block.6")

    mean_weight, mean_bias = folded("quantizer.in_proj")
    exported["quantizer.mean.weight"] = mean_weight[:32].contiguous()
    exported["quantizer.mean.bias"] = mean_bias[:32].contiguous()

    metadata = {
        "format": "irodori-dacvae-encoder-v1",
        "source": source.name,
        "config_json": json.dumps(kwargs, separators=(",", ":")),
        "sample_rate": "48000",
        "hop_length": "1920",
        "latent_dim": "32",
        "path": "deterministic-mean",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(exported, output, metadata=metadata)
    print(f"saved {len(exported)} tensors: {output} ({output.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
