#!/usr/bin/env python3
"""Assemble a blind A/B listening package from int8_quality_corpus work dirs.

For every text in each work dir the FP32 and candidate WAVs are copied as
pairNN_A.wav / pairNN_B.wav in a seeded random order.  key.json (which side
is which) is written to a separate directory so the listener does not see it.
score_sheet.md lists the pairs with the target text and empty judgement
columns (prefer A / prefer B / same, and notes).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from int8_quality_corpus import DEFAULT_TEXTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dirs", type=Path, nargs="+")
    parser.add_argument("--candidate", default="int8-plain")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    args.key_dir.mkdir(parents=True, exist_ok=True)
    key, sheet = [], ["# Blind A/B: FP32 vs " + args.candidate, "",
                      "Dengarkan A dan B, lalu isi: `A`, `B`, atau `=` (sama baik). Catat artefak yang terdengar.", "",
                      "| pair | mode | teks | pilihan | catatan |", "|---|---|---|---|---|"]
    n = 0
    for work in args.work_dirs:
        mode = work.name.replace("quality-", "")
        for text_id, text in DEFAULT_TEXTS.items():
            ref, cand = work / f"{text_id}-fp32.wav", work / f"{text_id}-{args.candidate}.wav"
            if not ref.exists() or not cand.exists():
                continue
            n += 1
            flip = rng.random() < 0.5
            a, b = (cand, ref) if flip else (ref, cand)
            shutil.copy(a, args.out / f"pair{n:02d}_A.wav")
            shutil.copy(b, args.out / f"pair{n:02d}_B.wav")
            key.append({"pair": n, "mode": mode, "text": text_id, "A": args.candidate if flip else "fp32",
                        "B": "fp32" if flip else args.candidate})
            sheet.append(f"| {n:02d} | {mode} | {text} |  |  |")
    (args.key_dir / "key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False))
    (args.out / "score_sheet.md").write_text("\n".join(sheet) + "\n")
    print(f"{n} pairs -> {args.out} (key in {args.key_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
