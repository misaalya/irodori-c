#!/usr/bin/env python3
"""Run the C engine over a small text corpus in several DiT precision
configurations and score every configuration against the FP32 reference
with tools/compare_audio_quality.py metrics.

Each configuration is a dict of environment overrides plus CLI precision.
The FP32 reference is generated with the same binary, seed and steps, so the
only difference is the DiT projection arithmetic.  A rounding-only FP32 pair
(MKL_CBWR=AVX512) is included as the numerical noise floor.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_audio_quality import metrics  # noqa: E402

DEFAULT_TEXTS = {
    "short": "おはようございます。",
    "greeting": "こんにちは、色とりどりの世界へようこそ。",
    "weather": "今日は朝から雨が降っていて、少し肌寒いですね。",
    "numbers": "会議は午後三時から、第二会議室で行います。",
    "long": "この製品は、日常のさまざまな場面で使えるように設計されており、初めての方でも簡単に操作できます。",
    "question": "週末はどこかへ出かける予定がありますか。",
}

CONFIGS = {
    "fp32": {"precision": "fp32", "env": {"MKL_CBWR": "AVX2"}},
    "fp32-avx512": {"precision": "fp32", "env": {"MKL_CBWR": "AVX512"}},
    # int8 GEMM sums are exact integers on every MKL branch; AVX512_E1 is the
    # VNNI branch the engine selects itself.  Only FP32 rounding differs.
    "int8": {"precision": "int8", "env": {"MKL_CBWR": "AVX512_E1"}},  # engine default policy
    "int8-plain": {"precision": "int8", "env": {"MKL_CBWR": "AVX512_E1", "IRO_INT8_RESIDUAL": "0"}},
    "int8-res-w2": {"precision": "int8", "env": {"MKL_CBWR": "AVX512_E1", "IRO_INT8_RESIDUAL": "8"}},
    "int8-res-w2-w13": {"precision": "int8", "env": {"MKL_CBWR": "AVX512_E1", "IRO_INT8_RESIDUAL": "12"}},
    "int8-res-all": {"precision": "int8", "env": {"MKL_CBWR": "AVX512_E1", "IRO_INT8_RESIDUAL": "15"}},
    # Codec int8 on an FP32 DiT isolates the decoder error (identical latent).
    "codec-int8": {"precision": "fp32", "codec": "int8", "env": {"MKL_CBWR": "AVX512_E1"}},
    "codec-int8-plain": {"precision": "fp32", "codec": "int8", "env": {"MKL_CBWR": "AVX512_E1", "IRO_CODEC_INT8_RESIDUAL": "0"}},
    "full-int8": {"precision": "int8", "codec": "int8", "env": {"MKL_CBWR": "AVX512_E1"}},
}


def run(binary: str, args: argparse.Namespace, text: str, out: Path, config: dict) -> dict:
    env = dict(os.environ)
    env.update(config["env"])
    env["IRO_NUM_THREADS"] = str(args.threads)
    cmd = [binary, "--text", text, "--model", args.model, "--tokenizer", args.tokenizer,
           "--decoder", args.decoder, "--steps", str(args.steps), "--seed", str(args.seed),
           "--dit-precision", config["precision"],
           "--codec-precision", config.get("codec", "fp32"), "--out", str(out)]
    if args.caption:
        cmd += ["--caption", args.caption]
    if args.ref:
        cmd += ["--ref", args.ref, "--encoder", args.encoder]
    if args.taskset:
        cmd = ["taskset", "-c", args.taskset] + cmd
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    if proc.returncode != 0:
        raise SystemExit(f"run failed: {' '.join(cmd)}\n{proc.stderr}")
    timing = {}
    for line in proc.stdout.splitlines():
        if line.startswith("encode "):
            parts = line.replace("|", " ").split()
            timing = {"encode": float(parts[1]), "euler": float(parts[4]), "decode": float(parts[7]), "total": float(parts[10])}
    timing["wall"] = wall
    return timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="weights/tokenizer.bin")
    parser.add_argument("--decoder", default="weights/dacvae_decoder.safetensors")
    parser.add_argument("--encoder", default="weights/dacvae_encoder.safetensors")
    parser.add_argument("--ref")
    parser.add_argument("--caption")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--taskset", default="0,1")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--texts", default=",".join(DEFAULT_TEXTS))
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configs = [c for c in args.configs.split(",") if c]
    texts = [t for t in args.texts.split(",") if t]
    results = {"args": vars(args) | {"work_dir": str(args.work_dir)}, "runs": []}
    for text_id in texts:
        text = DEFAULT_TEXTS[text_id]
        ref_wav = args.work_dir / f"{text_id}-fp32.wav"
        timing_ref = run(args.binary, args, text, ref_wav, CONFIGS["fp32"])
        results["runs"].append({"text": text_id, "config": "fp32", "timing": timing_ref})
        print(f"[{text_id}] fp32 euler {timing_ref.get('euler', 0):.2f}s", flush=True)
        for name in configs:
            if name == "fp32":
                continue
            out = args.work_dir / f"{text_id}-{name}.wav"
            timing = run(args.binary, args, text, out, CONFIGS[name])
            row = metrics(ref_wav, out, 0)
            row.pop("reference", None); row.pop("candidate", None)
            results["runs"].append({"text": text_id, "config": name, "timing": timing, "metrics": row})
            print(f"[{text_id}] {name:<16} euler {timing.get('euler', 0):.2f}s  snr {row['snr_db']:6.2f}  "
                  f"lsd {row['lsd_db']:.3f}  mcd {row['mcd_db']:.2f}  stoi {row['stoi']:.3f}", flush=True)
            (args.work_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    # aggregate
    import statistics
    print("\nconfig            n   snr_db(mean/min)   lsd_db(mean/max)   mcd(mean/max)   stoi(mean/min)   euler_s(median)")
    for name in configs:
        rows = [r for r in results["runs"] if r["config"] == name and "metrics" in r]
        if not rows:
            continue
        snr = [r["metrics"]["snr_db"] for r in rows]; lsd = [r["metrics"]["lsd_db"] for r in rows]
        mcd = [r["metrics"]["mcd_db"] for r in rows]; st = [r["metrics"]["stoi"] for r in rows]
        eul = [r["timing"].get("euler", 0) for r in rows]
        print(f"{name:<16} {len(rows):>3}   {statistics.mean(snr):7.2f}/{min(snr):6.2f}   "
              f"{statistics.mean(lsd):6.3f}/{max(lsd):6.3f}   {statistics.mean(mcd):6.2f}/{max(mcd):6.2f}   "
              f"{statistics.mean(st):.3f}/{min(st):.3f}   {statistics.median(eul):.2f}")
    eul = [r["timing"].get("euler", 0) for r in results["runs"] if r["config"] == "fp32"]
    print(f"{'fp32 (reference)':<16} {len(eul):>3}   {'':>14}   {'':>13}   {'':>13}   {'':>11}   {statistics.median(eul):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
