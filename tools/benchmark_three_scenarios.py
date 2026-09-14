#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_four_modes import (  # noqa: E402
    C_ROOT,
    DEFAULT_CAPTION,
    DEFAULT_TEXT,
    UPSTREAM_PYTHON,
    parse_cpus,
    run_interleaved_mode,
    sample_idle_fraction,
    worker_env,
)


SCENARIOS = (
    ("text-only", "Text only"),
    ("caption-only", "Caption only"),
    ("clone+caption", "Clone + caption"),
)


def write_chart(
    path: Path,
    *,
    title: str,
    ylabel: str,
    labels: list[str],
    python_values: list[float],
    c_values: list[float],
) -> None:
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    python_bars = ax.bar([item - width / 2 for item in x], python_values, width, label="Python")
    c_bars = ax.bar([item + width / 2 for item in x], c_values, width, label="C")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(python_bars, fmt="%.2f", padding=3)
    ax.bar_label(c_bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=UPSTREAM_PYTHON)
    parser.add_argument("--c-worker", type=Path, default=C_ROOT / "irodori-bench-worker-mkl")
    parser.add_argument("--c-backend", default="torch-mkl-sgemm")
    parser.add_argument("--tokenizer", type=Path, default=C_ROOT / "weights/tokenizer.bin")
    parser.add_argument("--decoder", type=Path, default=C_ROOT / "weights/dacvae_decoder.safetensors")
    parser.add_argument("--encoder", type=Path, default=C_ROOT / "weights/dacvae_encoder.safetensors")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", default="0,1")
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats <= 0 or args.threads <= 0:
        parser.error("--steps, --repeats, and --threads must be positive")

    output = args.output.expanduser().absolute()
    if output.exists() and any(output.iterdir()):
        parser.error(f"--output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir()

    model = args.model.expanduser().absolute()
    reference = args.ref.expanduser().resolve()
    python = args.python.expanduser().absolute()  # keep venv symlink intact
    worker = args.c_worker.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    decoder = args.decoder.expanduser().resolve()
    encoder = args.encoder.expanduser().resolve()
    for path in (model, reference, python, worker, tokenizer, decoder, encoder):
        if not path.is_file():
            parser.error(f"missing required file: {path}")

    cpus = parse_cpus(args.cpus, args.threads)
    python_env = worker_env(threads=args.threads, backend="python", overrides=[])
    c_env = worker_env(threads=args.threads, backend="c", overrides=[])
    idle = sample_idle_fraction(3.0)

    rows: list[dict[str, object]] = []
    raw: dict[str, object] = {
        "benchmark_kind": "diagnostic direct warm-inference Python vs C",
        "formal_acceptance": False,
        "idle_sample": idle,
        "steps": args.steps,
        "repeats": args.repeats,
        "threads": args.threads,
        "cpus": cpus,
        "seed": args.seed,
        "scenarios": {},
    }

    for mode, label in SCENARIOS:
        print(f"\n=== {label} ===", flush=True)
        result = run_interleaved_mode(
            python=python,
            c_worker=worker,
            mode=mode,
            model=model,
            tokenizer=tokenizer,
            decoder=decoder,
            encoder=encoder,
            reference=reference,
            caption=args.caption,
            text=args.text,
            seed=args.seed,
            steps=args.steps,
            threads=args.threads,
            repeats=args.repeats,
            cpus=cpus,
            directory=output / "raw" / mode,
            python_env=python_env,
            c_env=c_env,
            worker_timeout=900.0,
            python_backend="pytorch",
            c_backend=args.c_backend,
        )
        py = result["python"]
        c = result["c"]
        py_audio = audio_dir / f"{mode}-python.wav"
        c_audio = audio_dir / f"{mode}-c.wav"
        shutil.copy2(Path(str(py["wav_path"])), py_audio)
        shutil.copy2(Path(str(c["wav_path"])), c_audio)
        row = {
            "scenario": label,
            "mode": mode,
            "python_p50_seconds": float(py["p50_seconds"]),
            "c_p50_seconds": float(c["p50_seconds"]),
            "speedup_python_over_c": float(result["speedup_python_over_c"]),
            "python_rtf": float(py["rtf_p50"]),
            "c_rtf": float(c["rtf_p50"]),
            "python_peak_ram_mib": int(py["peak_rss_kib"]) / 1024.0,
            "c_peak_ram_mib": int(c["peak_rss_kib"]) / 1024.0,
            "ram_ratio_c_over_python": int(c["peak_rss_kib"]) / int(py["peak_rss_kib"]),
            "audio_seconds": float(py["audio_seconds"]),
            "pcm16_max_lsb": int(result["pcm16_parity"]["max_lsb"]),
            "pcm16_mae_lsb": float(result["pcm16_parity"]["mae_lsb"]),
            "python_audio": str(py_audio),
            "c_audio": str(c_audio),
        }
        rows.append(row)
        raw["scenarios"][mode] = result

    (output / "raw-summary.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    labels = [str(row["scenario"]) for row in rows]
    write_chart(
        output / "latency-p50.png",
        title=f"Direct inference latency ({args.steps} steps, p50 of {args.repeats} runs)",
        ylabel="Seconds (lower is better)",
        labels=labels,
        python_values=[float(row["python_p50_seconds"]) for row in rows],
        c_values=[float(row["c_p50_seconds"]) for row in rows],
    )
    write_chart(
        output / "peak-ram.png",
        title="Peak resident memory during direct inference",
        ylabel="Peak RSS (MiB, lower is better)",
        labels=labels,
        python_values=[float(row["python_peak_ram_mib"]) for row in rows],
        c_values=[float(row["c_peak_ram_mib"]) for row in rows],
    )
    write_chart(
        output / "rtf.png",
        title="Real-time factor",
        ylabel="RTF (lower is better; <1 is realtime)",
        labels=labels,
        python_values=[float(row["python_rtf"]) for row in rows],
        c_values=[float(row["c_rtf"]) for row in rows],
    )

    markdown = [
        "# Direct Python vs C — 3 scenarios",
        "",
        f"Diagnostic warm-inference benchmark: {args.steps} steps, {args.repeats} matched-noise runs, "
        f"{args.threads} threads on CPUs {cpus}. Not a formal acceptance run.",
        "",
        "| Scenario | Python p50 | C p50 | Speedup | Python RAM | C RAM | Python RTF | C RTF | PCM16 max LSB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['scenario']} | {row['python_p50_seconds']:.3f}s | {row['c_p50_seconds']:.3f}s | "
            f"{row['speedup_python_over_c']:.2f}x | {row['python_peak_ram_mib']:.1f} MiB | "
            f"{row['c_peak_ram_mib']:.1f} MiB | {row['python_rtf']:.2f} | {row['c_rtf']:.2f} | "
            f"{row['pcm16_max_lsb']} |"
        )
    markdown += [
        "",
        "## Charts",
        "",
        "![Latency](latency-p50.png)",
        "",
        "![Peak RAM](peak-ram.png)",
        "",
        "![RTF](rtf.png)",
        "",
        "## Audio",
        "",
    ]
    for row in rows:
        markdown.append(
            f"- {row['scenario']}: [Python](audio/{row['mode']}-python.wav) · "
            f"[C](audio/{row['mode']}-c.wav)"
        )
    (output / "REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(output), "idle": idle, "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
