#!/usr/bin/env python3
"""Render tables and Matplotlib figures for the int8 DiT evaluation from the
JSON artifacts written by int8_quality_corpus.py, asr_cer.py and
bench_speed_tradeoff.py.  Prints Markdown to stdout and writes PNGs."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def quality_tables(root: Path, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines, modes, series = [], [], {}
    for work in sorted(root.glob("quality-*")):
        res_path = work / "results.json"
        if not res_path.exists():
            continue
        runs = json.loads(res_path.read_text())["runs"]
        mode = work.name.replace("quality-", "")
        modes.append(mode)
        lines += [f"### {mode}", "", "| config | n | SNR dB mean/min | LSD dB mean/max | MCD mean/max | STOI mean/min | Euler s median (diagnostik) |", "|---|---:|---:|---:|---:|---:|---:|"]
        configs = []
        for r in runs:
            if r["config"] not in configs:
                configs.append(r["config"])
        for cfg in configs:
            rows = [r for r in runs if r["config"] == cfg]
            eul = statistics.median(r["timing"].get("euler", 0) for r in rows)
            if cfg == "fp32":
                lines.append(f"| fp32 (referensi) | {len(rows)} | – | – | – | – | {eul:.2f} |")
                continue
            m = [r["metrics"] for r in rows]
            snr = [x["snr_db"] for x in m]; lsd = [x["lsd_db"] for x in m]; mcd = [x["mcd_db"] for x in m]; st = [x["stoi"] for x in m]
            series.setdefault(cfg, {})[mode] = (statistics.mean(lsd), statistics.mean(st), min(st))
            lines.append(f"| {cfg} | {len(rows)} | {statistics.mean(snr):.2f}/{min(snr):.2f} | {statistics.mean(lsd):.3f}/{max(lsd):.3f} | {statistics.mean(mcd):.2f}/{max(mcd):.2f} | {statistics.mean(st):.3f}/{min(st):.3f} | {eul:.2f} |")
        lines.append("")
    if series:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        width = 0.8 / max(len(series), 1)
        for i, (cfg, per_mode) in enumerate(series.items()):
            xs = [j + i * width for j in range(len(modes))]
            axes[0].bar(xs, [per_mode.get(m, (0, 0, 0))[0] for m in modes], width, label=cfg)
            axes[1].bar(xs, [per_mode.get(m, (0, 0, 0))[1] for m in modes], width, label=cfg)
        for ax, title in zip(axes, ["Log-mel spectral distance vs FP32 (dB, lebih rendah lebih dekat)", "STOI vs FP32 (lebih tinggi lebih dekat)"]):
            ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(modes))])
            ax.set_xticklabels(modes)
            ax.set_title(title, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.35)
        axes[0].legend(fontsize=7, ncol=2, loc="upper left")
        fig.suptitle("Jarak audio ke FP32, 8-step, 6 teks per mode (fp32-avx512 = batas rounding)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "quality-distance.png", dpi=130)
        lines.append(f"![quality]({(out / 'quality-distance.png').name})\n")
    return lines


def asr_table(root: Path, out: Path) -> list[str]:
    rows = []
    for p in sorted(root.glob("asr-*.json")):
        rows += [r for r in json.loads(p.read_text()) if "cer" in r]
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({r["group"] for r in rows})
    lines = ["| grup | n | CER mean | CER median | CER max | klip CER>0.2 |", "|---|---:|---:|---:|---:|---:|"]
    means = {}
    for g in groups:
        v = [r["cer"] for r in rows if r["group"] == g]
        means[g] = statistics.mean(v)
        lines.append(f"| {g} | {len(v)} | {means[g]:.3f} | {statistics.median(v):.3f} | {max(v):.3f} | {sum(x > 0.2 for x in v)} |")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(groups)), [means[g] for g in groups], color=["#4c72b0" if "fp32" in g else "#dd8452" for g in groups])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CER rata-rata (lebih rendah lebih baik)")
    ax.set_title("ASR CER terhadap teks target (kotoba-whisper-v2.0)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "asr-cer.png", dpi=130)
    lines.append(f"\n![asr]({(out / 'asr-cer.png').name})\n")
    return lines


def speed_table(root: Path, out: Path) -> list[str]:
    lines = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for summary in sorted(root.glob("speed-*/summary.json")):
        rep = json.loads(summary.read_text())
        rows = rep.get("rows", [])
        if not rows:
            continue
        lines += [f"### {summary.parent.name} (status {rep.get('status')})", "",
                  "| mode | steps | pair | baseline s | candidate s | speedup | baseline sample s | candidate sample s | RSS base MiB | RSS cand MiB | idle |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        keyed = {}
        for r in rows:
            keyed.setdefault((r["mode"], r["steps"], r["pair"]), {})[r["variant"]] = r
        labels, base, cand = [], [], []
        for (mode, step, pair), v in sorted(keyed.items()):
            if "baseline" not in v or "candidate" not in v:
                continue
            b, c = v["baseline"]["run"], v["candidate"]["run"]
            labels.append(f"{mode}\ns{step} p{pair}"); base.append(b["elapsed_seconds"]); cand.append(c["elapsed_seconds"])
            lines.append(f"| {mode} | {step} | {pair} | {b['elapsed_seconds']:.3f} | {c['elapsed_seconds']:.3f} | {b['elapsed_seconds']/c['elapsed_seconds']:.3f}x | {b['sample_seconds']:.3f} | {c['sample_seconds']:.3f} | {b['peak_rss_kib']/1024:.0f} | {c['peak_rss_kib']/1024:.0f} | {v['candidate']['idle_run']['idle_fraction']:.0%} |")
        if base:
            import statistics as _st
            groups = {}
            for (mode, step, pair), v in sorted(keyed.items()):
                if "baseline" in v and "candidate" in v:
                    groups.setdefault((mode, step), ([], []))
                    groups[(mode, step)][0].append(v["baseline"]["run"]["elapsed_seconds"])
                    groups[(mode, step)][1].append(v["candidate"]["run"]["elapsed_seconds"])
            min_idle = min(r["idle_run"]["idle_fraction"] for r in rows)
            status = f"formal, idle >= {min_idle:.0%}" if min_idle >= 0.9 else f"diagnostik, idle min {min_idle:.0%}"
            names = [f"{m}\ns{st}" for (m, st) in groups]
            fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4))
            xs = range(len(names)); w = 0.38
            for i, (label, color) in enumerate([("baseline fp32", "#4c72b0"), ("candidate int8", "#dd8452")]):
                med = [_st.median(g[i]) for g in groups.values()]
                lo = [med[j] - min(g[i]) for j, g in enumerate(groups.values())]
                hi = [max(g[i]) - med[j] for j, g in enumerate(groups.values())]
                ax.bar([x + (i - 0.5) * w for x in xs], med, w, yerr=[lo, hi], capsize=3, label=label, color=color)
                for x, m in zip(xs, med):
                    ax.text(x + (i - 0.5) * w, m, f"{m:.1f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=8)
            ax.set_ylabel("warm E2E p50 s (lebih rendah lebih baik)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
            ax.set_title(f"{summary.parent.name}: {len(next(iter(groups.values()))[0])} pair, {status}", fontsize=9)
            fig.tight_layout(); png = out / f"{summary.parent.name}.png"; fig.savefig(png, dpi=130)
            lines.append(f"\n![speed]({png.name})\n")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    out = args.root
    parts = ["## Kualitas: jarak audio ke FP32", ""] + quality_tables(args.root, out)
    asr = asr_table(args.root, out)
    if asr:
        parts += ["## Kualitas absolut: ASR CER", ""] + asr
    speed = speed_table(args.root, out)
    if speed:
        parts += ["## Kecepatan A/B (harness bench_speed_tradeoff)", ""] + speed
    sys.stdout.write("\n".join(parts) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
