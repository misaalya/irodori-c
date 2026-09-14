#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_quality_corpus as quality


class QualityCorpusTests(unittest.TestCase):
    def test_two_reference_matrix_has_162_cases_and_full_coverage(self) -> None:
        refs = [("r1", Path("/tmp/r1.wav")), ("r2", Path("/tmp/r2.wav"))]
        scenarios = quality.build_scenarios(
            references=refs,
            seeds=list(quality.DEFAULT_SEEDS),
            steps=list(quality.DEFAULT_STEPS),
            captions=(("A", "caption a"), ("B", "caption b")),
        )
        coverage = quality.coverage_summary(scenarios, refs)
        self.assertEqual(len(scenarios), 162)
        self.assertEqual(coverage["expected_scenario_count_with_two_references"], 162)
        self.assertTrue(coverage["complete"])

    def test_one_reference_is_explicitly_incomplete(self) -> None:
        refs = [("r1", Path("/tmp/r1.wav"))]
        scenarios = quality.build_scenarios(
            references=refs,
            seeds=list(quality.DEFAULT_SEEDS),
            steps=list(quality.DEFAULT_STEPS),
            captions=(("A", "caption a"), ("B", "caption b")),
        )
        coverage = quality.coverage_summary(scenarios, refs)
        self.assertEqual(len(scenarios), 108)
        self.assertFalse(coverage["requirements"]["two_references"])
        self.assertFalse(coverage["complete"])

    def test_blind_order_is_stable_and_balanced_over_default_matrix(self) -> None:
        refs = [("r1", Path("/tmp/r1.wav")), ("r2", Path("/tmp/r2.wav"))]
        scenarios = quality.build_scenarios(
            references=refs,
            seeds=list(quality.DEFAULT_SEEDS),
            steps=list(quality.DEFAULT_STEPS),
            captions=(("A", "caption a"), ("B", "caption b")),
        )
        orders = [quality.blind_order(str(item["id"])) for item in scenarios]
        self.assertEqual(orders, [quality.blind_order(str(item["id"])) for item in scenarios])
        first_python = sum(order[0] == "python" for order in orders)
        self.assertGreater(first_python, len(orders) // 3)
        self.assertLess(first_python, len(orders) * 2 // 3)

    def test_listening_answer_key_stays_outside_blind_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python_wav = root / "python.wav"
            c_wav = root / "c.wav"
            python_wav.write_bytes(b"python-audio")
            c_wav.write_bytes(b"c-audio")
            cases = [
                {
                    "id": "case-40",
                    "text_id": "short",
                    "text": "hello",
                    "seed": 42,
                    "steps": 40,
                    "mode": "text-only",
                    "caption_id": None,
                    "caption": None,
                    "reference_id": None,
                    "python_wav": str(python_wav),
                    "c_wav": str(c_wav),
                }
            ]
            quality.create_listening_pack(root, cases)
            self.assertTrue((root / "listening" / "manifest.json").is_file())
            self.assertTrue((root / "listening" / "scores.csv").is_file())
            self.assertTrue((root / "listening-backend-key.json").is_file())
            self.assertFalse((root / "listening" / "backend-key.json").exists())


if __name__ == "__main__":
    unittest.main()
