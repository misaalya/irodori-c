#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_summaries(name: str):
    rows = []
    with (ROOT / name).open() as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("type") == "summary":
                rows.append(item)
    return rows


def label(row):
    return f"S{row['stage']} R{row['residual']}\n{row['channels']}ch d{row['dilation']}"


def plot_pair(rows, candidate_key: str, candidate_label: str, output: str):
    labels = [label(row) for row in rows]
    baseline = np.array([row["baseline_median"] for row in rows])
    candidate = np.array([row[candidate_key] for row in rows])
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, baseline, width, label="Baseline torch-MKL AVX2")
    ax.bar(x + width / 2, candidate, width, label=candidate_label)
    ax.set_ylabel("Median layer time (s) — lower is better")
    ax.set_title("Residual Conv7 diagnostic (host busy; not acceptance timing)")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / output, dpi=180)
    plt.close(fig)


batch = load_summaries("batch-conv7-diagnostic.jsonl")
avx512 = load_summaries("avx512-conv7-diagnostic.jsonl")

plot_pair(batch, "batch_median", "MKL sgemm_batch + deterministic reduce",
          "batch-conv7-diagnostic.png")
if avx512:
    plot_pair(avx512, "avx512_median", "Custom fused AVX-512",
              "avx512-conv7-diagnostic.png")

