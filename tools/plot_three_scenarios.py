#!/usr/bin/env python3
"""Consistent, publication-style figures for the three-scenario Python-vs-C
benchmark (text-only, caption-only, clone+caption).

Input: one or more bench_four_modes.py summary.json files, each an
interleaved Python/C run.  The C arm of every run is labelled by --labels
(e.g. "C fp32", "C int8"); the Python arm of every run is kept as an
independent replicate so run-to-run drift of the host is visible.

Figures (all "lower is better" unless stated):
  latency.png   p50 warm latency per scenario with min-max whiskers and the
                individual repeats overlaid; speedup vs Python annotated
  stages.png    stacked encode / sample / decode of the p50 run
  rtf.png       real-time factor with the 1.0 line
  ram.png       lifetime peak RSS
  repeats.png   every timed repeat as a strip plot (consistency check)
Tables: summary.md and summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALL_SCENARIOS = [("text-only", "Text only"), ("caption-only", "Caption only"), ("clone", "Clone"), ("clone+caption", "Clone + caption")]
SCENARIOS = [s for s in ALL_SCENARIOS if s[0] != "clone"]
PALETTE = {"python": "#7f7f7f", "python2": "#b0b0b0", "c0": "#1f77b4", "c1": "#ff7f0e", "c2": "#2ca02c"}


def load(path: Path, steps: str) -> dict:
    rep = json.loads(path.read_text())
    modes = rep["step_runs"][steps]["modes"]
    return {"report": rep, "modes": modes}


def arm_rows(runs: list[tuple[str, dict]]) -> list[dict]:
    """One row per (scenario, arm)."""
    rows = []
    for index, (label, run) in enumerate(runs):
        for mode, title in SCENARIOS:
            m = run["modes"].get(mode)
            if not m:
                continue
            for backend, name, color in (("python", f"Python fp32 (run {index + 1})", PALETTE["python"] if index == 0 else PALETTE["python2"]),
                                         ("c", label, PALETTE[f"c{index}"])):
                d = m[backend]
                samples = d["samples_seconds"]
                rows.append({
                    "scenario": title, "mode": mode, "arm": name, "backend": backend, "run": index + 1,
                    "color": color, "samples": samples, "p50": d["p50_seconds"], "p95": d["p95_seconds"],
                    "min": min(samples), "max": max(samples), "mean": statistics.mean(samples),
                    "cv_pct": 100 * statistics.pstdev(samples) / statistics.mean(samples),
                    "audio": d["audio_seconds"], "rtf": d["rtf_p50"], "rss_mib": d["peak_rss_kib"] / 1024,
                    "stages": d.get("stage_timings", {}), "deterministic": d.get("deterministic"),
                    "python_p50": m["python"]["p50_seconds"],
                    "parity": m.get("pcm16_parity", {}).get("max_lsb") if backend == "c" else None,
                    "gate": m.get("gate"),
                })
    return rows


def grouped(ax, rows, key, ylabel, title, whisk=True, annotate_speedup=False, hline=None, unit="s"):
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    n = len(arms); width = 0.8 / n
    for i, arm in enumerate(arms):
        vals, lo, hi, xs, colors = [], [], [], [], []
        for j, (mode, title_) in enumerate(SCENARIOS):
            r = next((r for r in rows if r["arm"] == arm and r["mode"] == mode), None)
            if not r:
                continue
            xs.append(j + (i - (n - 1) / 2) * width); vals.append(r[key]); colors.append(r["color"])
            lo.append(r["p50"] - r["min"] if key == "p50" else 0); hi.append(r["max"] - r["p50"] if key == "p50" else 0)
            if key == "p50":
                ax.scatter([xs[-1]] * len(r["samples"]), r["samples"], s=9, color="black", alpha=0.55, zorder=3)
        bars = ax.bar(xs, vals, width, label=arm, color=colors[0] if colors else None,
                      yerr=[lo, hi] if whisk and key == "p50" else None, capsize=3, edgecolor="black", linewidth=0.5)
        for b, v, x in zip(bars, vals, xs):
            txt = f"{v:.2f}" if unit == "s" else f"{v:.0f}"
            r = next(r for r in rows if r["arm"] == arm and abs(r[key] - v) < 1e-9)
            if annotate_speedup and r["backend"] == "c":
                txt += f"\n{r['python_p50'] / r['p50']:.2f}× vs Py"
            ax.text(x, b.get_height(), txt, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(SCENARIOS))); ax.set_xticklabels([t for _, t in SCENARIOS])
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1, label="real time (RTF = 1)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", type=Path, nargs="+")
    ap.add_argument("--labels", required=True, help="comma separated C-arm labels, one per summary")
    ap.add_argument("--steps", default="8")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Irodori-TTS v4.1 Small: Python (PyTorch) vs engine C")
    ap.add_argument("--scenarios", help="comma separated mode subset (default text-only,caption-only,clone+caption)")
    args = ap.parse_args()
    if args.scenarios:
        wanted = args.scenarios.split(",")
        SCENARIOS[:] = [sc for sc in ALL_SCENARIOS if sc[0] in wanted]
    labels = args.labels.split(",")
    runs = [(lab, load(p, args.steps)) for lab, p in zip(labels, args.summaries)]
    rows = arm_rows(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    proto = runs[0][1]["report"].get("protocol", {})
    foot = (f"{args.steps} Euler steps, seed 42, 2 threads on 2 physical cores, 1 warm-up + {runs[0][1]['report'].get('repeats', proto.get('repeats', '?'))} timed repeats per backend/scenario, "
            "Python and C interleaved, identical initial noise,\nhost ≥90% idle before every scenario. Whiskers = min–max, dots = individual repeats.")

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped(ax, rows, "p50", "warm end-to-end latency, p50 (s) — lower is better", f"{args.title}: end-to-end latency", annotate_speedup=True)
    fig.text(0.01, 0.005, foot, fontsize=6.5, ha="left"); fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(args.out / "latency.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    grouped(ax, rows, "rtf", "RTF, p50 (latency / audio duration) — lower is better", "Real-time factor", whisk=False, hline=1.0, unit="x")
    fig.tight_layout(); fig.savefig(args.out / "rtf.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    grouped(ax, rows, "rss_mib", "peak RSS (MiB) — lower is better", "Peak process memory (getrusage ru_maxrss, includes init and warm-up)", whisk=False, unit="MiB")
    fig.tight_layout(); fig.savefig(args.out / "ram.png", dpi=150); plt.close(fig)

    # stages
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    fig, ax = plt.subplots(figsize=(10, 5.2)); n = len(arms); width = 0.8 / n
    stage_colors = {"encode": "#9ecae1", "sample": "#3182bd", "decode": "#08519c"}
    sub_ticks, sub_labels = [], []
    for i, arm in enumerate(arms):
        for j, (mode, _) in enumerate(SCENARIOS):
            r = next((r for r in rows if r["arm"] == arm and r["mode"] == mode), None)
            if not r or not r["stages"]:
                continue
            x = j + (i - (n - 1) / 2) * width; bottom = 0.0
            for stage in ("encode", "sample", "decode"):
                v = r["stages"].get(stage, 0.0)
                ax.bar(x, v, width, bottom=bottom, color=stage_colors[stage], edgecolor="black", linewidth=0.4,
                       label=stage if (i == 0 and j == 0) else None)
                if v > 1.0:
                    ax.text(x, bottom + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=6, color="white" if stage != "encode" else "black")
                bottom += v
            ax.text(x, bottom, f"{bottom:.1f}", ha="center", va="bottom", fontsize=6.5)
            sub_ticks.append(x); sub_labels.append(arm.replace(" (run ", "\n(run ").replace(" (torch", "\n(torch").replace(" (oneMKL", "\n(oneMKL"))
    ax.set_xticks(sub_ticks); ax.set_xticklabels(sub_labels, fontsize=6, rotation=0)
    for j, (_, title_) in enumerate(SCENARIOS):
        ax.text(j, -0.16 * ax.get_ylim()[1], title_, ha="center", va="top", fontsize=10, fontweight="bold")
    ax.set_ylabel("seconds (p50 run) — lower is better"); ax.set_title("Stage breakdown: encode (front-end) / sample (DiT) / decode (codec)", fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=8); ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(args.out / "stages.png", dpi=150); plt.close(fig)

    # repeats strip
    fig, ax = plt.subplots(figsize=(10, 4.2))
    tick, ticks, labels_ = 0, [], []
    for mode, title in SCENARIOS:
        for arm in arms:
            r = next((r for r in rows if r["arm"] == arm and r["mode"] == mode), None)
            if not r:
                continue
            ax.scatter([tick] * len(r["samples"]), r["samples"], color=r["color"], s=18, edgecolor="black", linewidth=0.4, zorder=3)
            ax.plot([tick - 0.3, tick + 0.3], [r["p50"]] * 2, color="black", linewidth=1)
            ticks.append(tick); labels_.append(f"{title}\n{arm}"); tick += 1
        tick += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels(labels_, fontsize=5.5, rotation=90)
    ax.set_ylabel("latency per repeat (s)"); ax.set_title("Every timed repeat (line = p50): run-to-run consistency", fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(args.out / "repeats.png", dpi=150); plt.close(fig)

    # tables
    md = ["| Scenario | Arm | p50 s | p95 s | min–max s | CV % | Speedup vs Python | Latency cut | RTF | Peak RSS MiB | RAM cut | PCM16 max LSB | deterministic |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    with open(args.out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["scenario", "arm", "p50_s", "p95_s", "min_s", "max_s", "cv_pct", "speedup_vs_python", "rtf", "peak_rss_mib", "pcm16_max_lsb", "deterministic", "samples"])
        for r in rows:
            py = next(x for x in rows if x["mode"] == r["mode"] and x["backend"] == "python" and x["run"] == r["run"])
            sp = py["p50"] / r["p50"]; cut = 100 * (1 - r["p50"] / py["p50"]); ramcut = 100 * (1 - r["rss_mib"] / py["rss_mib"])
            md.append(f"| {r['scenario']} | {r['arm']} | {r['p50']:.3f} | {r['p95']:.3f} | {r['min']:.2f}–{r['max']:.2f} | {r['cv_pct']:.1f} | "
                      f"{sp:.2f}× | {cut:.1f}% | {r['rtf']:.2f} | {r['rss_mib']:.0f} | {ramcut:.1f}% | {r['parity'] if r['parity'] is not None else '–'} | {r['deterministic']} |")
            w.writerow([r["scenario"], r["arm"], f"{r['p50']:.6f}", f"{r['p95']:.6f}", f"{r['min']:.6f}", f"{r['max']:.6f}", f"{r['cv_pct']:.3f}", f"{sp:.4f}", f"{r['rtf']:.4f}", f"{r['rss_mib']:.1f}", r["parity"], r["deterministic"], " ".join(f"{s:.4f}" for s in r["samples"])])
    (args.out / "summary.md").write_text("\n".join(md) + "\n\n" + foot + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
