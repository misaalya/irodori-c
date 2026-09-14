#!/usr/bin/env python3
"""Build the final FP32 quality/listening corpus for Python vs C.

The corpus is intentionally separate from performance acceptance.  Each case
uses one PyTorch generation to freeze sampler noise, then one C generation with
the exact same noise.  A plan-only mode freezes the matrix before expensive
generation and reports missing coverage such as a second reference voice.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from bench_four_modes import (
    C_ROOT,
    C_WORKER,
    DEFAULT_CAPTION,
    DEFAULT_CAPTION_B,
    DEFAULT_TEXT,
    PY_WORKER,
    UPSTREAM_PYTHON,
    PersistentWorker,
    c_worker_command,
    compare_wavs,
    parse_cpus,
    persist_provenance,
    python_worker_command,
    selected_env,
    sha256,
    validate_worker_ready,
    worker_env,
)


DEFAULT_SEEDS = (42, 314159, 271828)
DEFAULT_STEPS = (8, 40)
DEFAULT_TEXTS = (
    ("short", "おはよう。"),
    ("medium", DEFAULT_TEXT),
    (
        "long",
        "今日は新しい技術について、できるだけ分かりやすく丁寧に説明します。"
        "最初に全体の目的を確認し、その後で具体的な手順と注意点を順番に見ていきましょう。",
    ),
)


def parse_int_list(value: str, option: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must contain comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(f"{option} must contain positive integers")
    return list(dict.fromkeys(result))


def parse_reference(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, raw_path = value.split("=", 1)
    else:
        raw_path = value
        name = Path(raw_path).stem
    name = name.strip()
    if not name or any(char in name for char in "/\\\t\n"):
        raise argparse.ArgumentTypeError("reference name must be a simple non-empty label")
    return name, Path(raw_path).expanduser()


def build_scenarios(
    *,
    references: list[tuple[str, Path]],
    seeds: list[int],
    steps: list[int],
    captions: tuple[tuple[str, str], tuple[str, str]],
    texts: tuple[tuple[str, str], ...] = DEFAULT_TEXTS,
) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for text_id, text in texts:
        for seed in seeds:
            for step_count in steps:
                scenarios.append(
                    {
                        "id": f"{text_id}-seed{seed}-s{step_count}-text-only",
                        "text_id": text_id,
                        "text": text,
                        "seed": seed,
                        "steps": step_count,
                        "mode": "text-only",
                        "caption_id": None,
                        "caption": None,
                        "reference_id": None,
                        "reference": None,
                    }
                )
                for caption_id, caption in captions:
                    scenarios.append(
                        {
                            "id": f"{text_id}-seed{seed}-s{step_count}-caption-{caption_id}",
                            "text_id": text_id,
                            "text": text,
                            "seed": seed,
                            "steps": step_count,
                            "mode": "caption-only",
                            "caption_id": caption_id,
                            "caption": caption,
                            "reference_id": None,
                            "reference": None,
                        }
                    )
                for reference_id, reference in references:
                    scenarios.append(
                        {
                            "id": f"{text_id}-seed{seed}-s{step_count}-clone-{reference_id}",
                            "text_id": text_id,
                            "text": text,
                            "seed": seed,
                            "steps": step_count,
                            "mode": "clone",
                            "caption_id": None,
                            "caption": None,
                            "reference_id": reference_id,
                            "reference": str(reference),
                        }
                    )
                    for caption_id, caption in captions:
                        scenarios.append(
                            {
                                "id": (
                                    f"{text_id}-seed{seed}-s{step_count}-"
                                    f"clone-caption-{reference_id}-{caption_id}"
                                ),
                                "text_id": text_id,
                                "text": text,
                                "seed": seed,
                                "steps": step_count,
                                "mode": "clone+caption",
                                "caption_id": caption_id,
                                "caption": caption,
                                "reference_id": reference_id,
                                "reference": str(reference),
                            }
                        )
    return scenarios


def coverage_summary(
    scenarios: list[dict[str, object]], references: list[tuple[str, Path]]
) -> dict[str, object]:
    modes = sorted({str(item["mode"]) for item in scenarios})
    text_ids = sorted({str(item["text_id"]) for item in scenarios})
    seeds = sorted({int(item["seed"]) for item in scenarios})
    steps = sorted({int(item["steps"]) for item in scenarios})
    caption_ids = sorted(
        {str(item["caption_id"]) for item in scenarios if item["caption_id"] is not None}
    )
    reference_ids = [name for name, _ in references]
    requirements = {
        "three_texts": len(text_ids) >= 3,
        "three_seeds": len(seeds) >= 3,
        "eight_and_forty_steps": 8 in steps and 40 in steps,
        "four_modes": set(modes) == {"text-only", "caption-only", "clone", "clone+caption"},
        "two_captions": len(caption_ids) >= 2,
        "two_references": len(reference_ids) >= 2,
        "distinct_references": len(set(reference_ids)) == len(reference_ids),
    }
    expected_with_two_references = len(text_ids) * len(seeds) * len(steps) * 9
    return {
        "scenario_count": len(scenarios),
        "expected_scenario_count_with_two_references": expected_with_two_references,
        "texts": text_ids,
        "seeds": seeds,
        "steps": steps,
        "modes": modes,
        "captions": caption_ids,
        "references": reference_ids,
        "requirements": requirements,
        "complete": all(requirements.values()) and len(scenarios) == expected_with_two_references,
    }


def blind_order(scenario_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    return ("python", "c") if digest[0] & 1 == 0 else ("c", "python")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_case(
    *,
    scenario: dict[str, object],
    directory: Path,
    python: Path,
    worker: Path,
    model: Path,
    tokenizer: Path,
    decoder: Path,
    encoder: Path,
    fallback_reference: Path,
    threads: int,
    cpus: list[int] | None,
    python_env: dict[str, str],
    c_env: dict[str, str],
    worker_timeout: float,
    python_backend: str,
    c_backend: str,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    mode = str(scenario["mode"])
    reference = (
        Path(str(scenario["reference"])) if scenario["reference"] is not None else fallback_reference
    )
    caption = str(scenario["caption"] or "")
    text = str(scenario["text"])
    seed = int(scenario["seed"])
    steps = int(scenario["steps"])
    noise = directory / "noise.f32"
    python_wav = directory / "python.wav"
    c_wav = directory / "c.wav"
    py_command = python_worker_command(
        python=python,
        model=model,
        reference=reference,
        caption=caption,
        text=text,
        seed=seed,
        steps=steps,
        threads=threads,
        mode=mode,
    )
    with PersistentWorker(
        py_command,
        env=python_env,
        cpus=cpus,
        label=f"quality/python/{scenario['id']}",
        log_path=directory / "python-worker.log",
        timeout_seconds=worker_timeout,
    ) as py:
        validate_worker_ready("quality/python", py.ready_payload, threads, python_backend)
        py_metrics = py.request(f"WARM\t{python_wav}\t{noise}")

    c_command = c_worker_command(
        worker=worker,
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        encoder=encoder,
        reference=reference,
        caption=caption,
        text=text,
        seed=seed,
        steps=steps,
        mode=mode,
        noise=noise,
    )
    with PersistentWorker(
        c_command,
        env=c_env,
        cpus=cpus,
        label=f"quality/c/{scenario['id']}",
        log_path=directory / "c-worker.log",
        timeout_seconds=worker_timeout,
    ) as c:
        validate_worker_ready("quality/c", c.ready_payload, threads, c_backend)
        c_metrics = c.request(f"WARM\t{c_wav}")

    return {
        **scenario,
        "python_wav": str(python_wav),
        "c_wav": str(c_wav),
        "python_sha256": sha256(python_wav),
        "c_sha256": sha256(c_wav),
        "noise_sha256": sha256(noise),
        "python_metrics": py_metrics,
        "c_metrics": c_metrics,
        "pcm16_parity": compare_wavs(python_wav, c_wav),
    }


def create_listening_pack(work_dir: Path, cases: list[dict[str, object]]) -> None:
    listening_dir = work_dir / "listening"
    public: list[dict[str, object]] = []
    key: list[dict[str, object]] = []
    score_rows: list[list[object]] = []
    eligible = [item for item in cases if int(item["steps"]) == 40]
    for index, item in enumerate(eligible, start=1):
        pair_id = f"Q{index:04d}"
        first_backend, second_backend = blind_order(str(item["id"]))
        source = {
            "python": Path(str(item["python_wav"])),
            "c": Path(str(item["c_wav"])),
        }
        a_name = f"{pair_id}-A.wav"
        b_name = f"{pair_id}-B.wav"
        link_or_copy(source[first_backend], listening_dir / a_name)
        link_or_copy(source[second_backend], listening_dir / b_name)
        context = {
            "pair_id": pair_id,
            "scenario_id": item["id"],
            "text_id": item["text_id"],
            "text": item["text"],
            "seed": item["seed"],
            "mode": item["mode"],
            "caption_id": item["caption_id"],
            "caption": item["caption"],
            "reference_id": item["reference_id"],
            "file_a": a_name,
            "file_b": b_name,
        }
        public.append(context)
        key.append(
            {
                "pair_id": pair_id,
                "scenario_id": item["id"],
                "A": first_backend,
                "B": second_backend,
            }
        )
        score_rows.append([pair_id, "", "", "", "", "", "", "", "", ""])
    write_json(listening_dir / "manifest.json", public)
    # Keep the answer key outside the blind-listening directory so opening the
    # listening pack does not accidentally reveal which backend produced A/B.
    write_json(work_dir / "listening-backend-key.json", key)
    with (listening_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pair_id",
                "articulation_A_B_equal_or_winner",
                "speaker_identity_A_B_equal_or_winner",
                "prosody_caption_A_B_equal_or_winner",
                "clicks_A_B_none",
                "hiss_A_B_none",
                "clipping_A_B_none",
                "robotic_A_B_none",
                "overall_preference_A_B_equal",
                "notes",
            ]
        )
        writer.writerows(score_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ref", action="append", type=parse_reference, default=[])
    parser.add_argument("--python", type=Path, default=UPSTREAM_PYTHON)
    parser.add_argument("--c-worker", type=Path, default=C_WORKER)
    parser.add_argument("--tokenizer", type=Path, default=C_ROOT / "weights/tokenizer.bin")
    parser.add_argument("--decoder", type=Path, default=C_ROOT / "weights/dacvae_decoder.safetensors")
    parser.add_argument("--encoder", type=Path, default=C_ROOT / "weights/dacvae_encoder.safetensors")
    parser.add_argument("--python-backend", default="pytorch")
    parser.add_argument("--c-backend", default="cblas-sgemm")
    parser.add_argument("--caption-a", default=DEFAULT_CAPTION)
    parser.add_argument("--caption-b", default=DEFAULT_CAPTION_B)
    parser.add_argument("--seeds", default=",".join(str(item) for item in DEFAULT_SEEDS))
    parser.add_argument("--steps", default=",".join(str(item) for item in DEFAULT_STEPS))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", default="auto")
    parser.add_argument("--worker-timeout", type=float, default=900.0)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted generation from an existing --work-dir",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="run at most this many pending cases, preserving a resumable partial manifest",
    )
    args = parser.parse_args()
    if args.threads <= 0 or args.worker_timeout <= 0:
        parser.error("--threads and --worker-timeout must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.resume and args.plan_only:
        parser.error("--resume cannot be combined with --plan-only")
    if args.resume and args.work_dir is None:
        parser.error("--resume requires --work-dir")
    try:
        seeds = parse_int_list(args.seeds, "--seeds")
        steps = parse_int_list(args.steps, "--steps")
        cpus = parse_cpus(args.cpus, args.threads)
    except (argparse.ArgumentTypeError, RuntimeError) as exc:
        parser.error(str(exc))

    references: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for name, path in args.ref:
        resolved = path.resolve()
        if name in seen_names:
            parser.error(f"duplicate reference label: {name}")
        if not resolved.is_file():
            parser.error(f"reference WAV does not exist: {resolved}")
        if resolved in seen_paths:
            parser.error(f"duplicate reference path: {resolved}")
        digest = sha256(resolved)
        if digest in seen_hashes:
            parser.error(f"reference audio duplicates an earlier file: {resolved}")
        seen_names.add(name)
        seen_paths.add(resolved)
        seen_hashes.add(digest)
        references.append((name, resolved))

    captions = (("A", args.caption_a), ("B", args.caption_b))
    scenarios = build_scenarios(
        references=references,
        seeds=seeds,
        steps=steps,
        captions=captions,
    )
    coverage = coverage_summary(scenarios, references)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    work_dir = (
        args.work_dir.expanduser().absolute()
        if args.work_dir is not None
        else (C_ROOT.parent / "benchmark-results" / f"quality-corpus-{stamp}").absolute()
    )
    manifest_path = work_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"--resume requires an existing manifest: {manifest_path}")
    else:
        if work_dir.exists() and any(work_dir.iterdir()):
            parser.error(f"--work-dir must be new or empty: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)

    reference_metadata = [
        {"id": name, "path": str(path), "sha256": sha256(path)} for name, path in references
    ]
    expected_plan: dict[str, object] = {
        "status": "planned" if args.plan_only else "running",
        "purpose": "FP32 quality corpus and blind listening; not performance timing",
        "seeds": seeds,
        "steps": steps,
        "texts": [{"id": name, "text": text} for name, text in DEFAULT_TEXTS],
        "captions": [{"id": name, "caption": caption} for name, caption in captions],
        "references": reference_metadata,
        "coverage": coverage,
        "scenarios": scenarios,
        "cases": [],
    }
    if args.resume:
        try:
            report = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read resume manifest {manifest_path}: {exc}")
        for key in ("purpose", "seeds", "steps", "texts", "captions", "references", "scenarios"):
            if report.get(key) != expected_plan[key]:
                parser.error(f"resume manifest does not match current {key}")
        existing_cases = report.get("cases")
        if not isinstance(existing_cases, list):
            parser.error("resume manifest has invalid cases list")
        case_ids = [str(item.get("id", "")) for item in existing_cases if isinstance(item, dict)]
        valid_ids = {str(item["id"]) for item in scenarios}
        if len(case_ids) != len(existing_cases) or len(set(case_ids)) != len(case_ids):
            parser.error("resume manifest contains invalid or duplicate case ids")
        if not set(case_ids).issubset(valid_ids):
            parser.error("resume manifest contains cases outside the frozen scenario matrix")
        report["status"] = "running"
        report.pop("abort_reason", None)
    else:
        report = expected_plan
        write_json(manifest_path, report)
    if args.plan_only:
        print(json.dumps({"work_dir": str(work_dir), "coverage": coverage}, ensure_ascii=False))
        return 0

    if args.model is None:
        parser.error("--model is required unless --plan-only is used")
    if not references:
        parser.error("at least one --ref is required for generation")
    model = args.model.expanduser().absolute()
    python = args.python.expanduser().absolute()
    worker = args.c_worker.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    decoder = args.decoder.expanduser().resolve()
    encoder = args.encoder.expanduser().resolve()
    required = [model, python, worker, PY_WORKER, tokenizer, decoder, encoder]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing required file(s): " + ", ".join(missing))

    python_env = worker_env(threads=args.threads, backend="python", overrides=[])
    c_env = worker_env(threads=args.threads, backend="c", overrides=[])
    runtime_config = {
        "model": {"path": str(model), "sha256": sha256(model)},
        "python": str(python),
        "c_worker": {"path": str(worker), "sha256": sha256(worker)},
        "python_worker": {"path": str(PY_WORKER), "sha256": sha256(PY_WORKER)},
        "quality_harness": {
            "path": str(Path(__file__).absolute()),
            "sha256": sha256(Path(__file__).absolute()),
        },
        "tokenizer": {"path": str(tokenizer), "sha256": sha256(tokenizer)},
        "decoder": {"path": str(decoder), "sha256": sha256(decoder)},
        "encoder": {"path": str(encoder), "sha256": sha256(encoder)},
        "threads": args.threads,
        "cpus": cpus,
        "python_backend": args.python_backend,
        "c_backend": args.c_backend,
        "python_env": selected_env(python_env),
        "c_env": selected_env(c_env),
    }
    if args.resume:
        if report.get("runtime_config") != runtime_config:
            parser.error("resume runtime configuration/checksums do not match the original run")
        report.setdefault("resume_events", []).append(
            {
                "timestamp_local": datetime.now().astimezone().isoformat(),
                "argv": sys.argv,
            }
        )
    else:
        report["runtime_config"] = runtime_config
        report["provenance"] = persist_provenance(
            work_dir=work_dir,
            argv=sys.argv,
            files=[model, python, worker, tokenizer, decoder, encoder, Path(__file__).absolute()],
            python_env=python_env,
            c_env=c_env,
            cpus=cpus,
        )
    write_json(manifest_path, report)
    fallback_reference = references[0][1]
    completed_ids = {str(item["id"]) for item in report["cases"]}
    pending = [item for item in scenarios if str(item["id"]) not in completed_ids]
    executed = 0
    try:
        for scenario in pending:
            if args.max_cases is not None and executed >= args.max_cases:
                break
            ordinal = len(report["cases"]) + 1
            print(f"[{ordinal}/{len(scenarios)}] {scenario['id']}", flush=True)
            case = run_case(
                scenario=scenario,
                directory=work_dir / "cases" / str(scenario["id"]),
                python=python,
                worker=worker,
                model=model,
                tokenizer=tokenizer,
                decoder=decoder,
                encoder=encoder,
                fallback_reference=fallback_reference,
                threads=args.threads,
                cpus=cpus,
                python_env=python_env,
                c_env=c_env,
                worker_timeout=args.worker_timeout,
                python_backend=args.python_backend,
                c_backend=args.c_backend,
            )
            report["cases"].append(case)
            executed += 1
            write_json(manifest_path, report)
    except (OSError, RuntimeError, TimeoutError) as exc:
        report["status"] = "aborted"
        report["abort_reason"] = str(exc)
        report["remaining_cases"] = len(scenarios) - len(report["cases"])
        write_json(manifest_path, report)
        print(f"quality corpus aborted: {exc}", file=sys.stderr)
        return 2

    remaining = len(scenarios) - len(report["cases"])
    report["remaining_cases"] = remaining
    if remaining:
        report["status"] = "partial"
        write_json(manifest_path, report)
        print(f"quality corpus partial: {work_dir}")
        print(f"completed={len(report['cases'])} remaining={remaining}")
        return 0

    report["status"] = "complete" if coverage["complete"] else "incomplete-coverage"
    write_json(manifest_path, report)
    create_listening_pack(work_dir, report["cases"])
    print(f"quality corpus: {work_dir}")
    print(f"coverage_complete={coverage['complete']}")
    return 0 if coverage["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
