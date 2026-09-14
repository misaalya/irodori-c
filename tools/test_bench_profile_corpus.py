#!/usr/bin/env python3
"""Unit tests for profile-corpus scenario and JSON aggregation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_profile_corpus as profile


class ProfileCorpusTests(unittest.TestCase):
    def test_default_matrix_covers_modes_steps_and_corpus_edges(self) -> None:
        scenarios = profile.build_scenarios(
            ["text-only", "caption-only", "clone", "clone+caption"], [8, 40]
        )
        self.assertEqual(len(scenarios), 12)
        keys = {(item["corpus_id"], item["mode"], item["steps"]) for item in scenarios}
        for steps in (8, 40):
            for mode in ("text-only", "caption-only", "clone", "clone+caption"):
                self.assertIn(("medium", mode, steps), keys)
            self.assertIn(("short", "text-only", steps), keys)
            self.assertIn(("long", "text-only", steps), keys)

    def test_profile_parser_and_summary(self) -> None:
        dit = {
            "type": "iro_dit_profile", "sequence_length": 117,
            "context_tokens": 23,
            "cells": [
                {"layer": 0, "batch": 3, "buckets": {
                    "w1_w3": {"calls": 4, "seconds": 1.25},
                    "softmax": {"calls": 80, "seconds": 0.1},
                }},
                {"layer": 0, "batch": 1, "buckets": {
                    "w1_w3": {"calls": 4, "seconds": 0.5},
                }},
            ],
        }
        gemm = {
            "type": "iro_gemm_profile", "dropped_shapes": 0,
            "rows": [
                {"M": 351, "N": 3680, "K": 1280, "calls": 4, "seconds": 2.0},
                {"M": 117, "N": 1280, "K": 3680, "calls": 4, "seconds": 1.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            log.write_text(
                "ready\n" + json.dumps(dit) + "\nnoise\n" + json.dumps(gemm) + "\n",
                encoding="utf-8",
            )
            parsed = profile.parse_profile_log(log)
        self.assertEqual(len(parsed["iro_dit_profile"]), 1)
        self.assertEqual(len(parsed["iro_gemm_profile"]), 1)
        summary = profile.summarize_profile(dit, gemm)
        self.assertEqual(summary["observed_batches"], [1, 3])
        self.assertEqual(summary["dit_bucket_totals"][0]["name"], "w1_w3")
        self.assertAlmostEqual(summary["dit_bucket_totals"][0]["seconds"], 1.75)
        self.assertEqual(summary["top_gemm_shapes"][0]["M"], 351)


if __name__ == "__main__":
    unittest.main()
