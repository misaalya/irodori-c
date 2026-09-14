#!/usr/bin/env python3
"""Profile warm C inference across CFG batches and representative text shapes."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from bench_four_modes import (
    ALL_MODES,
    DEFAULT_CAPTION,
    DEFAULT_TEXT,
    PersistentWorker,
    affinity_uses_distinct_physical_cores,
    parse_cpus,
    parse_modes,
    parse_steps,
    persist_provenance,
    selected_env,
    validate_worker_ready,
    wait_for_idle,
    worker_env,
)


C_ROOT = Path(__file__).resolve().parents[1]
PROFILE_TYPES = ("iro_dit_profile", "iro_gemm_profile")
DEFAULT_CORPUS = (
    ("short", "おはよう。"),
    ("medium", DEFAULT_TEXT),
    (
        "long",
        "今日は新しい技術について、できるだけ分かりやすく丁寧に説明します。"
        "最初に全体の目的を確認し、その後で具体的な手順と注意点を順番に見ていきましょう。",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_scenarios(
    modes: list[str], steps: list[int], corpus: tuple[tuple[str, str], ...] = DEFAULT_CORPUS
) -> list[dict[str, object]]:
    """Cover every CFG mode on medium text plus short/long text-only shapes."""
    texts = dict(corpus)
    scenarios: list[dict[str, object]] = []
    for step_count in steps:
        for mode in modes:
            scenarios.append(
                {
                    "id": f"medium-{mode}-s{step_count}",
                    "corpus_id": "medium",
                    "text": texts["medium"],
                    "mode": mode,
                    "steps": step_count,
                }
            )
        if "text-only" in modes:
            for corpus_id in ("short", "long"):
                scenarios.append(
                    {
                        "id": f"{corpus_id}-text-only-s{step_count}",
                        "corpus_id": corpus_id,
                        "text": texts[corpus_id],
                        "mode": "text-only",
                        "steps": step_count,
                    }
                )
    return scenarios


def worker_command(
    *, worker: Path, model: Path, tokenizer: Path, decoder: Path, encoder: Path,
    reference: Path | None, caption: str, text: str, seed: int, steps: int,
    mode: str,
) -> list[str]:
    command = [
        str(worker), "--model", str(model), "--tokenizer", str(tokenizer),
        "--decoder", str(decoder), "--text", text, "--seed", str(seed),
        "--steps", str(steps),
    ]
    if mode in {"clone", "clone+caption"}:
        if reference is None:
            raise RuntimeError(f"{mode} requires --ref")
        command += ["--encoder", str(encoder), "--ref", str(reference)]
    if mode in {"caption-only", "clone+caption"}:
        command += ["--caption", caption]
    return command


def parse_profile_log(path: Path) -> dict[str, list[dict[str, object]]]:
    records = {name: [] for name in PROFILE_TYPES}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = payload.get("type")
        if kind in records:
            payload["log_line"] = line_number
            records[kind].append(payload)
    return records


def summarize_profile(
    dit: dict[str, object], gemm: dict[str, object]
) -> dict[str, object]:
    bucket_totals: dict[str, dict[str, float | int]] = {}
    batches: set[int] = set()
    for cell in dit.get("cells", []):
        batch = int(cell["batch"])
        batches.add(batch)
        for name, values in cell.get("buckets", {}).items():
            total = bucket_totals.setdefault(name, {"calls": 0, "seconds": 0.0})
            total["calls"] = int(total["calls"]) + int(values["calls"])
            total["seconds"] = float(total["seconds"]) + float(values["seconds"])
    buckets = [
        {"name": name, "calls": values["calls"], "seconds": values["seconds"]}
        for name, values in bucket_totals.items()
    ]
    buckets.sort(key=lambda item: float(item["seconds"]), reverse=True)
    gemm_rows = sorted(
        gemm.get("rows", []), key=lambda item: float(item["seconds"]), reverse=True
    )
    return {
        "sequence_length": int(dit["sequence_length"]),
        "context_tokens": int(dit["context_tokens"]),
        "observed_batches": sorted(batches),
        "max_batch": max(batches, default=0),
        "dit_bucket_totals": buckets,
        "gemm_seconds": sum(float(row["seconds"]) for row in gemm_rows),
        "top_gemm_shapes": gemm_rows[:12],
        "dropped_gemm_shapes": int(gemm.get("dropped_shapes", 0)),
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# O1 four-mode profile corpus",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Scenario | Mode | Steps | S | Context | Batches | Sample (s) | Deterministic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["scenarios"]:
        summary = item["profile_summary"]
        batches = "/".join(str(value) for value in summary["observed_batches"])
        lines.append(
            f"| {item['id']} | {item['mode']} | {item['steps']} | "
            f"{summary['sequence_length']} | {summary['context_tokens']} | {batches} | "
            f"{item['run']['sample_seconds']:.3f} | {item['deterministic']} |"
        )
    lines += ["", "## Coverage", "", "```json", json.dumps(
        report.get("coverage", {}), indent=2, ensure_ascii=False
    ), "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=C_ROOT / "irodori-bench-worker")
    parser.add_argument("--tokenizer", type=Path, default=C_ROOT / "weights/tokenizer.bin")
    parser.add_argument("--decoder", type=Path, default=C_ROOT / "weights/dacvae_decoder.safetensors")
    parser.add_argument("--encoder", type=Path, default=C_ROOT / "weights/dacvae_encoder.safetensors")
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=parse_steps, default=[8, 40])
    parser.add_argument("--modes", type=parse_modes, default=list(ALL_MODES))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", default="auto")
    parser.add_argument("--backend", default="cblas-sgemm")
    parser.add_argument("--idle-sample-seconds", type=float, default=2.0)
    parser.add_argument("--idle-wait-seconds", type=float, default=30.0)
    parser.add_argument("--min-idle-fraction", type=float, default=0.90)
    parser.add_argument("--worker-timeout", type=float, default=900.0)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if (args.threads <= 0 or args.idle_sample_seconds <= 0
            or args.idle_wait_seconds < 0 or args.worker_timeout <= 0):
        parser.error(
            "--threads, --idle-sample-seconds, and --worker-timeout must be positive; "
            "--idle-wait-seconds must be non-negative"
        )
    if not 0.0 <= args.min_idle_fraction <= 1.0:
        parser.error("--min-idle-fraction must be between 0 and 1")
    try:
        cpus = parse_cpus(args.cpus, args.threads)
    except RuntimeError as exc:
        parser.error(str(exc))

    model = args.model.expanduser().absolute()
    worker = args.worker.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    decoder = args.decoder.expanduser().resolve()
    encoder = args.encoder.expanduser().resolve()
    reference = args.ref.expanduser().resolve() if args.ref else None
    required = [model, worker, tokenizer, decoder]
    if any(mode in {"clone", "clone+caption"} for mode in args.modes):
        required.append(encoder)
        if reference is None:
            parser.error("--ref is required when clone modes are selected")
        required.append(reference)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing required file(s): " + ", ".join(missing))

    if args.work_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        work_dir = C_ROOT.parent / "benchmark-results" / f"o1-profile-corpus-{stamp}"
    else:
        work_dir = args.work_dir.expanduser().absolute()
    if work_dir.exists() and any(work_dir.iterdir()):
        parser.error(f"--work-dir must be new or empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    env = worker_env(threads=args.threads, backend="c", overrides=[])
    env["IRO_DIT_PROFILE"] = "1"
    env["IRO_GEMM_PROFILE"] = "1"
    scenarios = build_scenarios(args.modes, args.steps)
    report: dict[str, object] = {
        "status": "running",
        "model": str(model),
        "worker": str(worker),
        "reference": None if reference is None else str(reference),
        "backend": args.backend,
        "threads": args.threads,
        "cpus": cpus,
        "distinct_physical_cores": affinity_uses_distinct_physical_cores(cpus, args.threads),
        "minimum_idle_fraction": args.min_idle_fraction,
        "idle_wait_seconds": args.idle_wait_seconds,
        "environment": selected_env(env),
        "scenario_plan": scenarios,
        "idle_checks": [],
        "scenarios": [],
    }
    report["provenance"] = persist_provenance(
        work_dir=work_dir,
        argv=sys.argv,
        files=[model, worker, tokenizer, decoder, encoder, Path(__file__).absolute(),
               C_ROOT / "tools/bench_c_worker.c"],
        python_env={}, c_env=env, cpus=cpus,
    )
    summary_path = work_dir / "summary.json"
    write_json(summary_path, report)

    try:
        for index, scenario in enumerate(scenarios, start=1):
            idle = wait_for_idle(
                sample_seconds=args.idle_sample_seconds,
                minimum_fraction=args.min_idle_fraction,
                timeout_seconds=args.idle_wait_seconds,
            )
            fraction = idle.get("idle_fraction")
            passed = bool(idle.get("passed"))
            idle["scenario"] = scenario["id"]
            report["idle_checks"].append(idle)
            write_json(summary_path, report)
            if not passed:
                value = "unavailable" if fraction is None else f"{fraction:.4f}"
                raise RuntimeError(
                    f"idle fraction {value} below {args.min_idle_fraction:.4f} "
                    f"before {scenario['id']}"
                )

            scenario_dir = work_dir / str(scenario["id"])
            scenario_dir.mkdir(parents=True, exist_ok=True)
            log_path = scenario_dir / "worker.log"
            command = worker_command(
                worker=worker, model=model, tokenizer=tokenizer, decoder=decoder,
                encoder=encoder, reference=reference, caption=args.caption,
                text=str(scenario["text"]), seed=args.seed,
                steps=int(scenario["steps"]), mode=str(scenario["mode"]),
            )
            with PersistentWorker(
                command, env=env, cpus=cpus, label=str(scenario["id"]),
                log_path=log_path, timeout_seconds=args.worker_timeout,
            ) as process:
                validate_worker_ready(
                    str(scenario["id"]), process.ready_payload, args.threads, args.backend
                )
                warm_path = scenario_dir / "warm.wav"
                run_path = scenario_dir / "run.wav"
                warm = process.request(f"WARM\t{warm_path}")
                run = process.request(f"RUN\t{run_path}")

            records = parse_profile_log(log_path)
            if len(records["iro_dit_profile"]) != 2 or len(records["iro_gemm_profile"]) != 2:
                raise RuntimeError(
                    f"{scenario['id']} expected two DiT and GEMM profile records; "
                    f"got {len(records['iro_dit_profile'])}/"
                    f"{len(records['iro_gemm_profile'])}"
                )
            dit = records["iro_dit_profile"][-1]
            gemm = records["iro_gemm_profile"][-1]
            if dit.get("status") != "ok" or gemm.get("status") != "ok":
                raise RuntimeError(f"{scenario['id']} profiler reported an error")
            if gemm.get("backend") != args.backend:
                raise RuntimeError(
                    f"{scenario['id']} GEMM backend mismatch: {gemm.get('backend')!r}"
                )
            deterministic = sha256(warm_path) == sha256(run_path)
            item = {
                **scenario,
                "worker_ready": process.ready_payload,
                "warm": warm,
                "run": run,
                "warm_wav_sha256": sha256(warm_path),
                "run_wav_sha256": sha256(run_path),
                "deterministic": deterministic,
                "profile_summary": summarize_profile(dit, gemm),
                "dit_profile": dit,
                "gemm_profile": gemm,
                "log": str(log_path),
            }
            if not deterministic:
                raise RuntimeError(f"{scenario['id']} warm/run WAV hashes differ")
            write_json(scenario_dir / "profile.json", item)
            report["scenarios"].append(item)
            write_json(summary_path, report)
            profile = item["profile_summary"]
            print(
                f"[{index}/{len(scenarios)}] {scenario['id']}: "
                f"S={profile['sequence_length']} context={profile['context_tokens']} "
                f"B={profile['observed_batches']} sample={run['sample_seconds']:.3f}s",
                flush=True,
            )
    except (OSError, RuntimeError, TimeoutError) as exc:
        report["status"] = "aborted"
        report["abort_reason"] = str(exc)
        write_json(summary_path, report)
        (work_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
        print(f"profile corpus aborted: {exc}", file=sys.stderr)
        print(f"summary: {summary_path}")
        return 2

    expected_max_batch = {
        "text-only": 2,
        "caption-only": 3,
        "clone": 3,
        "clone+caption": 4,
    }
    observed_mode_batches: dict[str, set[int]] = {mode: set() for mode in args.modes}
    observed_corpus: set[str] = set()
    for item in report["scenarios"]:
        observed_corpus.add(str(item["corpus_id"]))
        observed_mode_batches[str(item["mode"])].update(
            int(value) for value in item["profile_summary"]["observed_batches"]
        )
    report["coverage"] = {
        "modes": list(args.modes),
        "steps": list(args.steps),
        "corpus_ids": sorted(observed_corpus),
        "observed_batches_by_mode": {
            mode: sorted(values) for mode, values in observed_mode_batches.items()
        },
        "expected_max_batch_by_mode": {
            mode: expected_max_batch[mode] for mode in args.modes
        },
        "all_requested_modes": tuple(args.modes) == ALL_MODES,
        "eight_and_forty_steps": 8 in args.steps and 40 in args.steps,
        "short_medium_long": {"short", "medium", "long"}.issubset(observed_corpus),
        "batch_2_3_4": all(
            expected in observed_mode_batches[mode]
            for mode, expected in expected_max_batch.items() if mode in args.modes
        ),
        "all_deterministic": all(bool(item["deterministic"]) for item in report["scenarios"]),
        "all_gemm_shapes_recorded": all(
            int(item["profile_summary"]["dropped_gemm_shapes"]) == 0
            for item in report["scenarios"]
        ),
    }
    report["coverage"]["complete"] = all(
        bool(report["coverage"][key])
        for key in (
            "all_requested_modes", "eight_and_forty_steps", "short_medium_long",
            "batch_2_3_4", "all_deterministic", "all_gemm_shapes_recorded",
        )
    )
    report["status"] = "complete"
    write_json(summary_path, report)
    (work_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["coverage"], indent=2, ensure_ascii=False))
    print(f"summary: {summary_path}")
    return 0 if report["coverage"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
