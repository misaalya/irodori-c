#!/usr/bin/env python3
"""Audio-level quality metrics between a reference WAV and candidate WAVs.

The FP32 golden gates compare intermediate tensors bit-for-bit-ish; they cannot
judge a numerically different path such as W8A8.  This tool compares the
*rendered audio* instead, using metrics that track what a listener perceives:

  snr_db      waveform SNR of candidate vs reference (higher is better)
  lsd_db      log-mel spectral distance, mean over frames (lower is better)
  mcd_db      mel-cepstral distortion on 13 MFCCs without c0 (lower is better)
  stoi        short-time objective intelligibility, reference = FP32 (0..1)
  corr        Pearson correlation of the waveforms

Interpretation needs calibration pairs produced by the same pipeline: an FP32
pair that differs only in rounding (e.g. MKL_CBWR=AVX2 vs AVX512) gives the
"noise floor" of numerically equivalent implementations; an FP32 pair with a
different seed gives the scale of "another valid sample".  A candidate that
sits between those and keeps stoi/mcd close to the floor has not changed the
audio in a way the FP32 gate would have protected against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def align(ref: np.ndarray, cand: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Trim both to a common length after a small cross-correlation lag search."""
    n = min(len(ref), len(cand))
    ref, cand = ref[:n], cand[:n]
    if max_lag <= 0 or n < 4 * max_lag:
        return ref, cand, 0
    best_lag, best = 0, -np.inf
    r = ref[max_lag:-max_lag]
    for lag in range(-max_lag, max_lag + 1):
        c = cand[max_lag + lag : n - max_lag + lag]
        score = float(np.dot(r, c))
        if score > best:
            best, best_lag = score, lag
    if best_lag:
        if best_lag > 0:
            ref, cand = ref[: n - best_lag], cand[best_lag:]
        else:
            ref, cand = ref[-best_lag:], cand[: n + best_lag]
    return ref, cand, best_lag


def log_mel(audio: np.ndarray, rate: int, n_fft: int = 2048, hop: int = 480, n_mels: int = 80):
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio, sr=rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )
    return 10.0 * np.log10(np.maximum(mel, 1e-10))


def mcd(ref: np.ndarray, cand: np.ndarray, rate: int) -> float:
    import librosa

    kwargs = dict(sr=rate, n_mfcc=13, n_fft=2048, hop_length=480, n_mels=80)
    a = librosa.feature.mfcc(y=ref, **kwargs)[1:]
    b = librosa.feature.mfcc(y=cand, **kwargs)[1:]
    n = min(a.shape[1], b.shape[1])
    diff = a[:, :n] - b[:, :n]
    # librosa MFCCs are computed on dB-scaled log-mel, so the usual
    # 10/ln(10) factor is already included in the coefficients.
    return float(np.mean(np.sqrt(2.0 * np.sum(diff * diff, axis=0))))


def metrics(ref_path: Path, cand_path: Path, max_lag: int) -> dict:
    ref, rate = load_mono(ref_path)
    cand, cand_rate = load_mono(cand_path)
    if rate != cand_rate:
        raise SystemExit(f"sample rate mismatch: {ref_path} {rate} vs {cand_path} {cand_rate}")
    length_ratio = len(cand) / max(len(ref), 1)
    ref_a, cand_a, lag = align(ref, cand, max_lag)
    noise = ref_a - cand_a
    sig = float(np.dot(ref_a, ref_a))
    err = float(np.dot(noise, noise))
    snr = 10.0 * np.log10(sig / err) if err > 0 else float("inf")
    corr = float(np.corrcoef(ref_a, cand_a)[0, 1]) if len(ref_a) > 1 else float("nan")
    lm_ref, lm_cand = log_mel(ref_a, rate), log_mel(cand_a, rate)
    n = min(lm_ref.shape[1], lm_cand.shape[1])
    lsd = float(np.mean(np.sqrt(np.mean((lm_ref[:, :n] - lm_cand[:, :n]) ** 2, axis=0))))
    try:
        from pystoi import stoi

        stoi_value = float(stoi(ref_a, cand_a, rate, extended=False))
    except Exception:  # pragma: no cover - optional dependency
        stoi_value = float("nan")
    return {
        "reference": str(ref_path),
        "candidate": str(cand_path),
        "reference_samples": int(len(ref)),
        "candidate_samples": int(len(cand)),
        "length_ratio": float(length_ratio),
        "lag_samples": int(lag),
        "snr_db": float(snr),
        "corr": corr,
        "lsd_db": lsd,
        "mcd_db": mcd(ref_a, cand_a, rate),
        "stoi": stoi_value,
        "peak_ref": float(np.max(np.abs(ref))),
        "peak_cand": float(np.max(np.abs(cand))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--labels", help="comma separated labels for candidates")
    parser.add_argument("--max-lag", type=int, default=0, help="lag search in samples (0 = assume aligned)")
    parser.add_argument("--json", type=Path, help="write all rows as JSON")
    args = parser.parse_args()
    labels = args.labels.split(",") if args.labels else [c.name for c in args.candidates]
    if len(labels) != len(args.candidates):
        raise SystemExit("--labels count must match candidates")
    rows = []
    print(f"{'candidate':<32} {'len':>6} {'lag':>4} {'snr_db':>8} {'corr':>7} {'lsd_db':>7} {'mcd_db':>7} {'stoi':>6}")
    for label, cand in zip(labels, args.candidates):
        row = metrics(args.reference, cand, args.max_lag)
        row["label"] = label
        rows.append(row)
        print(
            f"{label:<32} {row['length_ratio']:>6.3f} {row['lag_samples']:>4d} "
            f"{row['snr_db']:>8.2f} {row['corr']:>7.4f} {row['lsd_db']:>7.3f} "
            f"{row['mcd_db']:>7.3f} {row['stoi']:>6.3f}"
        )
    if args.json:
        args.json.write_text(json.dumps({"reference": str(args.reference), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
