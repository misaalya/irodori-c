#!/usr/bin/env python3
"""Export the deterministic DACVAE decoder in a C-runtime-friendly format.

The source checkpoint stores PyTorch weight-normalization parameters and many
encoder/watermark tensors unused by text-only decoding.  This one-time tool
folds weight normalization and writes only the active decoder graph.  Runtime
inference remains pure C and mmaps the resulting safetensors file.
"""
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
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent / "weights" / "dacvae_decoder.safetensors"
)


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PTH
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    kwargs = payload.get("metadata", {}).get("kwargs", {})
    exported: dict[str, torch.Tensor] = {}

    def add_weight_norm(base: str) -> None:
        vector = state[f"{base}.weight_v"].float()
        magnitude = state[f"{base}.weight_g"].float()
        # This is the exact operator used by torch.nn.utils.weight_norm.
        weight = torch._weight_norm(  # noqa: SLF001
            vector, magnitude, 0
        )
        # Pack the kernel tap before channels.  For Conv1d this changes
        # [Cout,Cin,K] -> [Cout,K,Cin], making im2col source copies contiguous.
        # For ConvTranspose1d, [Cin,Cout,K] -> [Cin,K,Cout] makes overlap-add
        # source vectors and time-major destination vectors contiguous.
        weight = weight.permute(0, 2, 1)
        exported[f"{base}.weight"] = weight.contiguous()
        exported[f"{base}.bias"] = state[f"{base}.bias"].float().contiguous()

    def add_alpha(name: str) -> None:
        exported[name] = state[name].float().contiguous()

    add_weight_norm("quantizer.out_proj")
    add_weight_norm("decoder.model.0")
    for stage in range(1, 5):
        prefix = f"decoder.model.{stage}.block"
        add_alpha(f"{prefix}.0.alpha")
        add_weight_norm(f"{prefix}.1")
        for residual in (4, 5, 8):
            residual_prefix = f"{prefix}.{residual}.block"
            add_alpha(f"{residual_prefix}.0.alpha")
            add_weight_norm(f"{residual_prefix}.1")
            add_alpha(f"{residual_prefix}.2.alpha")
            add_weight_norm(f"{residual_prefix}.3")
    add_alpha("decoder.wm_model.encoder_block.pre.0.alpha")
    add_weight_norm("decoder.wm_model.encoder_block.pre.1")

    metadata = {
        "format": "irodori-dacvae-decoder-v2",
        "source": source.name,
        "config_json": json.dumps(kwargs, separators=(",", ":")),
        "watermark": "disabled-alpha-zero",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(exported, output, metadata=metadata)
    gib = output.stat().st_size / (1024**3)
    print(f"saved {len(exported)} tensors: {output} ({gib:.3f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
