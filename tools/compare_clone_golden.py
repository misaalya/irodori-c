#!/usr/bin/env python3
"""Compare pure-C conditioned end-to-end dumps against a PyTorch fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LIMITS = {
    "reference_latent": 1.0e-4,
    "speaker_state": 2.0e-4,
    "caption_state": 2.0e-4,
    "duration_log_frames": 1.0e-5,
    "noise": 0.0,
    "latent_final": 2.0e-4,
    "waveform": 1.0e-4,
}

CAPTION_COMPACT_LIMITS = {
    "latent_final": 1.0e-3,
    "waveform": 2.0e-3,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("golden", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    limits = dict(LIMITS)
    velocity_limit = 1.0e-3
    metadata_path = args.golden / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("caption"):
            # The C runtime intentionally removes 503 masked caption positions
            # before attention. This changes FP32 reduction order relative to
            # PyTorch SDPA while preserving the same mathematical context.
            limits.update(CAPTION_COMPACT_LIMITS)
            velocity_limit = 5.0e-3

    failed = False
    checks = list(limits.items())
    checks.extend(
        (path.stem, velocity_limit)
        for path in sorted(args.golden.glob("velocity_step*.f32"))
    )
    for name, limit in checks:
        expected_path = args.golden / f"{name}.f32"
        # Speaker/caption tensors are mode-dependent.  A single comparator is
        # intentionally shared by text-only, caption-only, clone, and
        # clone+caption fixtures.
        if not expected_path.exists():
            continue
        expected = np.fromfile(expected_path, dtype=np.float32)
        actual = np.fromfile(args.actual / f"{name}.f32", dtype=np.float32)
        if expected.size != actual.size or expected.size == 0:
            print(f"FAIL {name}: count expected={expected.size} actual={actual.size}")
            failed = True
            continue
        difference = np.abs(expected - actual)
        max_error = float(difference.max())
        mae = float(difference.mean())
        expected64 = expected.astype(np.float64)
        actual64 = actual.astype(np.float64)
        denominator = float(np.linalg.norm(expected64) * np.linalg.norm(actual64))
        cosine = float(np.dot(expected64, actual64) / denominator) if denominator else 1.0
        passed = max_error <= limit
        print(
            f"{'PASS' if passed else 'FAIL'} {name}: count={expected.size} "
            f"max={max_error:.8g} mae={mae:.8g} cosine={cosine:.10f} "
            f"limit={limit:.1g}"
        )
        failed |= not passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
