#!/usr/bin/env python3
"""Konversi DACVAE weights.pth (torch pickle) -> safetensors untuk mmap dari C.

Sekali jalankan offline; output dipakai engine C (D4 di PLAN.md).
Juga mencetak inventory struktur codec untuk MODEL.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Irodori-TTS"))

DEFAULT_PTH = Path.home() / (
    ".cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/"
    "snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth"
)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "weights" / "dacvae.safetensors"


def main() -> int:
    pth = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PTH
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    payload = torch.load(str(pth), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        print(f"payload tak terduga: {type(payload)} keys={list(payload)[:5] if isinstance(payload, dict) else ''}")
        return 1
    state = payload["state_dict"]
    meta = payload.get("metadata", {})
    kwargs = meta.get("kwargs", {}) if isinstance(meta, dict) else {}

    print(f"== {len(state)} tensor ==\n")
    for name, tensor in sorted(state.items()):
        print(f"{name:64s} {str(tuple(tensor.shape)):22s} {tensor.dtype}")

    out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, str(out))
    print(f"\nsaved: {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # kwargs arsitektur codec — dibutuhkan engine C untuk konstruksi decoder/encoder
    kwargs_out = out.with_suffix(".json")
    kwargs_out.write_text(json.dumps(kwargs, indent=2, default=str))
    print(f"saved: {kwargs_out}")
    print("\n== codec kwargs ==")
    print(json.dumps(kwargs, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
