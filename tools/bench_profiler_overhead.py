#!/usr/bin/env python3
"""Measure opt-in DiT/GEMM profiler overhead with paired real Euler runs."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

from bench_four_modes import (
    affinity_uses_distinct_physical_cores,
    parse_cpus,
    wait_for_idle,
)


C_ROOT = Path(__file__).resolve().parents[1]
EULER_RE = re.compile(r"Euler full ([^:]+): ([0-9.]+) s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_once(
    *,
    binary: Path,
    model: Path,
    golden: Path,
    cpus: list[int] | None,
    threads: int,
    profile: bool,
    expected_backend: str,
    output_dir: Path,
    index: int,
) -> dict[str, object]:
    label = "on" if profile else "off"
    command = [str(binary), "--test-euler", str(model), str(golden)]
    if cpus is not None:
        command = ["taskset", "-c", ",".join(str(cpu) for cpu in cpus), *command]

    env = os.environ.copy()
    env.update(
        {
            "IRO_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
        }
    )
    if profile:
        env["IRO_DIT_PROFILE"] = "1"
        env["IRO_GEMM_PROFILE"] = "1"
    else:
        env.pop("IRO_DIT_PROFILE", None)
        env.pop("IRO_GEMM_PROFILE", None)

    completed = subprocess.run(
        command,
        cwd=C_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        check=False,
    )
    stem = f"{index:02d}-{label}"
    (output_dir / f"{stem}.stdout").write_text(completed.stdout, encoding="utf-8")
    (output_dir / f"{stem}.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{stem} exited with status {completed.returncode}")
    match = EULER_RE.search(completed.stdout)
    if not match:
        raise RuntimeError(f"{stem} did not report Euler timing")
    backend = match.group(1)
    if backend != expected_backend:
        raise RuntimeError(
            f"{stem} backend mismatch: expected={expected_backend!r} actual={backend!r}"
        )
    seconds = float(match.group(2))
    return {
        "index": index,
        "profile": label,
        "backend": backend,
        "euler_seconds": seconds,
        "stdout": str(output_dir / f"{stem}.stdout"),
        "stderr": str(output_dir / f"{stem}.stderr"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=C_ROOT / "irodori-blas")
    parser.add_argument("--golden", type=Path, default=C_ROOT / "golden/seed42_steps8")
    parser.add_argument("--backend", default="cblas-sgemm")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--cpus", default="auto")
    parser.add_argument("--idle-sample-seconds", type=float, default=2.0)
    parser.add_argument("--idle-wait-seconds", type=float, default=30.0)
    parser.add_argument("--min-idle-fraction", type=float, default=0.90)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.threads <= 0 or args.pairs < 3:
        parser.error("--threads must be positive and --pairs must be at least 3")
    if args.idle_sample_seconds <= 0 or args.idle_wait_seconds < 0:
        parser.error("--idle-sample-seconds must be positive and --idle-wait-seconds non-negative")
    if not 0.0 <= args.min_idle_fraction <= 1.0:
        parser.error("--min-idle-fraction must be between 0 and 1")
    try:
        cpus = parse_cpus(args.cpus, args.threads)
    except RuntimeError as exc:
        parser.error(str(exc))

    binary = args.binary.expanduser().resolve()
    model = args.model.expanduser().absolute()
    golden = args.golden.expanduser().resolve()
    if not binary.is_file() or not model.is_file() or not golden.is_dir():
        parser.error("--binary, --model, and --golden must exist")
    if args.work_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output_dir = C_ROOT.parent / "benchmark-results" / f"profiler-overhead-{stamp}"
    else:
        output_dir = args.work_dir.expanduser().absolute()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"--work-dir must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "status": "running",
        "model": str(model),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "golden": str(golden),
        "backend": args.backend,
        "threads": args.threads,
        "pairs_requested": args.pairs,
        "cpus": cpus,
        "distinct_physical_cores": affinity_uses_distinct_physical_cores(
            cpus, args.threads
        ),
        "minimum_idle_fraction": args.min_idle_fraction,
        "idle_wait_seconds": args.idle_wait_seconds,
        "idle_checks": [],
        "runs": [],
    }
    summary_path = output_dir / "summary.json"
    write_summary(summary_path, report)

    schedule: list[bool] = [False]  # discarded warm-up
    for pair in range(args.pairs):
        schedule.extend((False, True) if pair % 2 == 0 else (True, False))

    try:
        for index, profile in enumerate(schedule):
            idle = wait_for_idle(
                sample_seconds=args.idle_sample_seconds,
                minimum_fraction=args.min_idle_fraction,
                timeout_seconds=args.idle_wait_seconds,
            )
            idle["before_run"] = index
            report["idle_checks"].append(idle)
            fraction = idle.get("idle_fraction")
            if not idle.get("passed"):
                value = "unavailable" if fraction is None else f"{fraction:.4f}"
                raise RuntimeError(
                    f"idle fraction {value} below {args.min_idle_fraction:.4f} "
                    f"before run {index}"
                )
            run = run_once(
                binary=binary,
                model=model,
                golden=golden,
                cpus=cpus,
                threads=args.threads,
                profile=profile,
                expected_backend=args.backend,
                output_dir=output_dir,
                index=index,
            )
            run["warmup"] = index == 0
            report["runs"].append(run)
            write_summary(summary_path, report)
            print(
                f"run {index}: profile={run['profile']} "
                f"Euler={run['euler_seconds']:.3f}s",
                flush=True,
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report["status"] = "aborted"
        report["abort_reason"] = str(exc)
        write_summary(summary_path, report)
        print(f"profiler overhead benchmark aborted: {exc}", file=sys.stderr)
        print(f"summary: {summary_path}")
        return 2

    measured = report["runs"][1:]
    pairs: list[dict[str, float]] = []
    for offset in range(0, len(measured), 2):
        pair_runs = measured[offset : offset + 2]
        off = next(float(run["euler_seconds"]) for run in pair_runs if run["profile"] == "off")
        on = next(float(run["euler_seconds"]) for run in pair_runs if run["profile"] == "on")
        pairs.append({"off_seconds": off, "on_seconds": on, "overhead_percent": (on / off - 1.0) * 100.0})
    off_values = [pair["off_seconds"] for pair in pairs]
    on_values = [pair["on_seconds"] for pair in pairs]
    off_median = statistics.median(off_values)
    on_median = statistics.median(on_values)
    report.update(
        {
            "status": "complete",
            "pairs": pairs,
            "off_median_seconds": off_median,
            "on_median_seconds": on_median,
            "median_overhead_percent": (on_median / off_median - 1.0) * 100.0,
        }
    )
    write_summary(summary_path, report)
    print(
        f"median OFF={off_median:.3f}s ON={on_median:.3f}s "
        f"overhead={report['median_overhead_percent']:.3f}%"
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
