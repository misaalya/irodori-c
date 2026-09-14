#!/usr/bin/env python3
"""Fast fault-path tests for the four-mode benchmark harness.

These tests intentionally avoid loading model weights.  They exercise the
protocol/error handling that must fail closed before a long benchmark run can
be treated as evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import bench_four_modes as bench


TOOLS = Path(__file__).resolve().parent
COMPARE = TOOLS / "compare_clone_golden.py"


class HarnessFaultPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="irodori-bench-harness-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _script(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def test_worker_startup_error_is_reported(self) -> None:
        script = self._script(
            "startup_error.py",
            """
            import sys
            print("dummy startup failure", flush=True)
            raise SystemExit(7)
            """,
        )
        with self.assertRaisesRegex(RuntimeError, r"exited before __IRO_READY__ .*code=7"):
            bench.PersistentWorker(
                [sys.executable, str(script)],
                env=dict(**__import__("os").environ),
                cpus=None,
                label="startup-error",
                log_path=self.root / "startup-error.log",
                timeout_seconds=1.0,
            )

    def test_worker_readiness_timeout_is_reported(self) -> None:
        script = self._script(
            "timeout.py",
            """
            import time
            time.sleep(10)
            """,
        )
        with self.assertRaisesRegex(TimeoutError, r"timed out waiting for __IRO_READY__"):
            bench.PersistentWorker(
                [sys.executable, str(script)],
                env=dict(**__import__("os").environ),
                cpus=None,
                label="timeout",
                log_path=self.root / "timeout.log",
                timeout_seconds=0.05,
            )

    def test_worker_backend_mismatch_is_rejected(self) -> None:
        payload = {
            "backend": "wrong-backend",
            "requested_threads": 2,
            "backend_threads": 2,
        }
        with self.assertRaisesRegex(RuntimeError, r"backend mismatch"):
            bench.validate_worker_ready("dummy", payload, 2, "cblas-sgemm")

    def test_nonfinite_metric_is_rejected(self) -> None:
        sample = {
            "elapsed_seconds": float("nan"),
            "encode_seconds": 0.1,
            "sample_seconds": 0.2,
            "decode_seconds": 0.3,
            "output_samples": 48000,
            "peak_rss_kib": 1,
        }
        with self.assertRaisesRegex(RuntimeError, r"non-finite/negative elapsed_seconds"):
            bench.summarize_worker_samples(
                backend="dummy",
                mode="text-only",
                steps=8,
                threads=2,
                seed=42,
                samples=[sample],
                hashes=["x"],
                output=self.root / "unused.wav",
                noise=None,
            )

    def test_empty_output_metric_is_rejected(self) -> None:
        sample = {
            "elapsed_seconds": 1.0,
            "encode_seconds": 0.1,
            "sample_seconds": 0.2,
            "decode_seconds": 0.3,
            "output_samples": 0,
            "peak_rss_kib": 1,
        }
        with self.assertRaisesRegex(RuntimeError, r"empty output"):
            bench.summarize_worker_samples(
                backend="dummy",
                mode="text-only",
                steps=8,
                threads=2,
                seed=42,
                samples=[sample],
                hashes=["x"],
                output=self.root / "unused.wav",
                noise=None,
            )

    def test_partial_summary_is_persisted_after_one_pair(self) -> None:
        directory = self.root / "partial"
        directory.mkdir()
        bench.persist_pair_progress(
            directory,
            mode="clone",
            steps=8,
            completed_pairs=1,
            requested_pairs=5,
            python_samples=[{"elapsed_seconds": 2.0}],
            c_samples=[{"elapsed_seconds": 1.0}],
            python_ready={"backend": "pytorch"},
            c_ready={"backend": "cblas-sgemm"},
        )
        progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
        partial = json.loads((directory / "partial-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(progress, partial)
        self.assertEqual(progress["completed_pairs"], 1)
        self.assertEqual(progress["requested_pairs"], 5)

    def test_only_three_modes_cannot_be_formal(self) -> None:
        common = dict(
            repeats=5,
            steps=[8, 40],
            threads=2,
            distinct_physical_cores=True,
            idle_pass=True,
            idle_checks=[{"passed": True}],
        )
        self.assertFalse(
            bench.formal_protocol_met(
                modes=("text-only", "caption-only", "clone"),
                **common,
            )
        )
        self.assertTrue(bench.formal_protocol_met(modes=bench.ALL_MODES, **common))

    def test_idle_wait_records_failed_attempt_before_pass(self) -> None:
        samples = [
            {"available": True, "idle_fraction": 0.5},
            {"available": True, "idle_fraction": 0.95},
        ]
        with mock.patch.object(bench, "sample_idle_fraction", side_effect=samples):
            result = bench.wait_for_idle(
                sample_seconds=0.01, minimum_fraction=0.90, timeout_seconds=1.0
            )
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertFalse(result["attempts"][0]["passed"])
        self.assertTrue(result["attempts"][1]["passed"])

    def test_abort_summary_preserves_partial_results(self) -> None:
        report_path = self.root / "summary.json"
        report = {"step_runs": {"8": {"modes": {"text-only": {"done": True}}}}}
        bench.record_abort(report, report_path, "dummy abort")
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(saved["step_runs"]["8"]["modes"]["text-only"]["done"])
        self.assertTrue(saved["acceptance"]["aborted"])
        self.assertFalse(saved["acceptance"]["formal_protocol_met"])
        self.assertEqual(saved["acceptance"]["abort_reason"], "dummy abort")

    def test_missing_and_empty_golden_are_rejected(self) -> None:
        actual = self.root / "actual"
        actual.mkdir()
        missing = self.root / "does-not-exist"
        empty = self.root / "empty-golden"
        empty.mkdir()
        for golden in (missing, empty):
            completed = subprocess.run(
                [sys.executable, str(COMPARE), str(golden), str(actual)],
                cwd=bench.C_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("FAIL metadata", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
