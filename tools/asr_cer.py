#!/usr/bin/env python3
"""Reference-free intelligibility gate: transcribe WAVs with a Japanese
Whisper model and report character error rate against the intended text.

Distance-to-FP32 metrics saturate for any ~1% perturbation because the Euler
trajectory diverges, so they cannot rank int8 variants.  CER against the
*target text* measures whether the rendered speech still says the right
thing, independently of which valid sample the sampler landed on.  Compare
the FP32 and int8 columns of the same corpus; the ASR itself adds noise, so
judge aggregates (mean CER, count of clips above a threshold), not single
clips.

Model: kotoba-tech/kotoba-whisper-v2.0 (distil-whisper, Japanese).  Loaded
through the upstream venv's transformers; only the HF cache is touched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
PUNCT = re.compile(r"[\s、。，．,.!?！？「」『』（）()\-ー～〜:：;；\"'・…]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = PUNCT.sub("", text)
    return text.lower()


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return levenshtein(ref, hyp) / max(len(ref), 1)


def load_pipeline(threads: int):
    import torch
    from transformers import pipeline

    torch.set_num_threads(threads)
    return pipeline("automatic-speech-recognition", model=MODEL_ID, torch_dtype=torch.float32, device="cpu")


def transcribe(asr, path: Path) -> str:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    out = asr({"raw": audio, "sampling_rate": rate}, generate_kwargs={"language": "ja", "task": "transcribe"})
    return out["text"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="JSON list of {wav, text, label, group}")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    items = json.loads(args.manifest.read_text())
    asr = load_pipeline(args.threads)
    rows = []
    for item in items:
        hyp = transcribe(asr, Path(item["wav"]))
        value = cer(item["text"], hyp)
        rows.append({**item, "hypothesis": hyp, "cer": value})
        print(f"{item.get('group', ''):<16} {item.get('label', ''):<24} cer {value:6.3f}  {hyp}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    groups = sorted({r.get("group", "") for r in rows})
    print("\ngroup            n   mean_cer  median_cer  max_cer  clips>0.2")
    for g in groups:
        vals = [r["cer"] for r in rows if r.get("group", "") == g]
        print(f"{g:<16} {len(vals):>3}   {np.mean(vals):8.3f}  {np.median(vals):10.3f}  {max(vals):7.3f}  {sum(v > 0.2 for v in vals):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
