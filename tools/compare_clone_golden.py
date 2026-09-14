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

VALID_MODES = {"text-only", "caption-only", "clone", "clone+caption"}


def _legacy_mode(golden: Path, metadata: dict, shapes: dict) -> str | None:
    """Infer only enough to keep the checked-in pre-mode fixtures usable.

    Shape/path evidence is intentional: optional descriptive metadata alone
    must not be able to turn a clone/caption fixture into a weaker text-only
    comparison by omission.
    """
    name = golden.name.lower()
    has_ref_tensor = "reference_latent" in shapes or "speaker_state" in shapes
    has_caption_tensor = "caption_state" in shapes
    path_clone = "clone" in name
    path_caption = "caption" in name
    has_ref = has_ref_tensor or path_clone or bool(metadata.get("reference"))
    has_caption = has_caption_tensor or path_caption or bool(metadata.get("caption"))
    if has_ref and has_caption:
        return "clone+caption"
    if has_ref:
        return "clone"
    if has_caption:
        return "caption-only"
    # Legacy conditioned fixtures must carry some independent evidence of
    # their mode.  Do not silently downgrade an ambiguous directory.
    if name in {"seed42_steps8", "text-only", "text_only"}:
        return "text-only"
    return None


def _fixture_mode(golden: Path, metadata: dict, shapes: dict) -> str | None:
    mode = metadata.get("mode")
    if mode is None:
        return _legacy_mode(golden, metadata, shapes)
    if not isinstance(mode, str) or mode not in VALID_MODES:
        return None
    return mode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("golden", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    metadata_path = args.golden / "metadata.json"
    if not metadata_path.is_file():
        print(f"FAIL metadata: missing {metadata_path}")
        return 1
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL metadata: {exc}")
        return 1
    shapes = metadata.get("shapes")
    if not isinstance(shapes, dict) or not shapes:
        print("FAIL metadata: non-empty shapes object is required")
        return 1
    explicit_mode = metadata.get("mode") is not None
    mode = _fixture_mode(args.golden, metadata, shapes)
    if mode is None:
        print("FAIL metadata: valid mode is required (or unambiguous legacy fixture evidence)")
        return 1
    if not isinstance(metadata.get("steps"), int) or metadata["steps"] <= 0:
        print("FAIL metadata: positive integer steps is required")
        return 1
    if explicit_mode and ("caption" not in metadata or "reference" not in metadata):
        print("FAIL metadata: caption and reference fields are required")
        return 1

    expects_caption = mode in {"caption-only", "clone+caption"}
    expects_reference = mode in {"clone", "clone+caption"}
    if expects_caption != bool(metadata.get("caption")):
        print(f"FAIL metadata: mode {mode} contradicts caption field")
        return 1
    if expects_reference != bool(metadata.get("reference")):
        print(f"FAIL metadata: mode {mode} contradicts reference field")
        return 1

    limits = dict(LIMITS)
    velocity_limit = 1.0e-3
    if expects_caption:
        # The C runtime intentionally removes masked caption positions before
        # attention. This changes FP32 reduction order relative to PyTorch SDPA
        # while preserving the same mathematical context.
        limits.update(CAPTION_COMPACT_LIMITS)
        velocity_limit = 5.0e-3

    required = {"duration_log_frames", "noise", "latent_final", "waveform"}
    if expects_caption:
        required.add("caption_state")
    if expects_reference:
        required.update({"reference_latent", "speaker_state"})
    has_velocity = any(name.startswith("velocity_step") for name in shapes)
    if explicit_mode or has_velocity:
        required.update(f"velocity_step{step:03d}" for step in range(metadata["steps"]))
    missing_declared = sorted(required.difference(shapes))
    if missing_declared:
        print("FAIL metadata: missing required tensors: " + ", ".join(missing_declared))
        return 1

    failed = False
    checks: list[tuple[str, float]] = []
    for name in shapes:
        if name.startswith("velocity_step"):
            checks.append((name, velocity_limit))
        elif name in limits:
            checks.append((name, limits[name]))
    if not checks:
        print("FAIL metadata: no comparable tensors declared")
        return 1

    for name, limit in checks:
        expected_path = args.golden / f"{name}.f32"
        actual_path = args.actual / f"{name}.f32"
        shape = shapes.get(name)
        if not isinstance(shape, list) or not shape or any(
            not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            print(f"FAIL {name}: invalid shape metadata {shape!r}")
            failed = True
            continue
        expected_count = int(np.prod(np.asarray(shape, dtype=np.int64), dtype=np.int64))
        if not expected_path.is_file() or not actual_path.is_file():
            print(
                f"FAIL {name}: missing expected={expected_path.is_file()} "
                f"actual={actual_path.is_file()}"
            )
            failed = True
            continue
        expected = np.fromfile(expected_path, dtype=np.float32)
        actual = np.fromfile(actual_path, dtype=np.float32)
        if expected.size != expected_count or actual.size != expected_count:
            print(
                f"FAIL {name}: count metadata={expected_count} "
                f"expected={expected.size} actual={actual.size}"
            )
            failed = True
            continue
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            print(f"FAIL {name}: non-finite data")
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
