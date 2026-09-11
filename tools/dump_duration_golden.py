#!/usr/bin/env python3
"""Build the 20-case text-only duration regression fixture.

The binary stores only valid-token text states, so the C duration component can
be tested independently without re-running the 25-layer backbone per case.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from irodori_tts.inference_runtime import (  # noqa: E402
    RuntimeKey,
    download_hf_checkpoint,
    get_cached_runtime,
)
from irodori_tts.text_normalization import normalize_text  # noqa: E402

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
MAGIC = b"IRODUR1\0"


def main() -> int:
    golden_dir = ROOT / "irodori-c" / "golden"
    rows = []
    for line in (golden_dir / "tokenizer_vectors.tsv").read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        raw = bytes.fromhex(fields[2]).decode("utf-8")
        rows.append((raw, normalize_text(raw)))
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 tokenizer vectors, found {len(rows)}")

    checkpoint = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    key = RuntimeKey(
        checkpoint=checkpoint,
        model_device="cpu",
        codec_repo=CODEC_REPO,
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )
    print("[duration-golden] loading runtime (fp32)...")
    runtime, _ = get_cached_runtime(key)
    model = runtime.model

    normalized = [row[1] for row in rows]
    max_len = max(int(runtime.tokenizer.encode(text).numel()) for text in normalized)
    ids, mask = runtime.tokenizer.batch_encode(normalized, max_length=max_len)
    ids = ids.to(runtime.model_device)
    mask = mask.to(runtime.model_device)
    with torch.inference_mode():
        projected = model.text_encoder(model.pretrained_text_backbone, ids, mask)
        text_state = model.text_norm(projected)
        predicted = model.duration_predictor(
            text_state,
            text_mask=mask,
            aux_features=torch.zeros(
                (len(rows), model.cfg.duration_aux_dim),
                device=runtime.model_device,
                dtype=text_state.dtype,
            ),
            speaker_state=None,
            speaker_mask=None,
            has_speaker=torch.zeros(len(rows), dtype=torch.bool, device=runtime.model_device),
            caption_state=None,
            caption_mask=None,
            has_caption=torch.zeros(len(rows), dtype=torch.bool, device=runtime.model_device),
        ).float()

    output = golden_dir / "duration_vectors.bin"
    manifest = []
    with output.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(rows), text_state.shape[-1]))
        for i, ((raw, norm), count) in enumerate(zip(rows, mask.sum(dim=1).tolist(), strict=True)):
            count = int(count)
            log_frames = float(predicted[i].item())
            state = text_state[i, :count].detach().cpu().contiguous().numpy()
            f.write(struct.pack("<If", count, log_frames))
            state.tofile(f)
            manifest.append(
                {
                    "raw": raw,
                    "normalized": norm,
                    "tokens": count,
                    "log_frames": log_frames,
                    "frames": round(math.expm1(log_frames)),
                }
            )
    (golden_dir / "duration_vectors.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(f"[duration-golden] {len(rows)} cases -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
