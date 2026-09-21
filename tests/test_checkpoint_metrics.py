"""CPU contracts for full-location diagnostic bookkeeping."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.diagnostics.checkpoint_metrics import location_key, summarize_records  # noqa: E402


class CheckpointMetricsTest(unittest.TestCase):
    def test_summary_keeps_unevaluable_locations_and_groups_by_view_and_pc(self) -> None:
        records = [
            {"view": "SAX", "slice_id": "s1", "eligible": True, "cardiac_target_fraction": .2, "same_peak_exact": True, "selected_cardiac_pc": 1},
            {"view": "SAX", "slice_id": "s2", "eligible": True, "cardiac_target_fraction": .6, "same_peak_exact": False, "selected_cardiac_pc": 2},
            {"view": "2CH", "slice_id": "s3", "eligible": False, "skip_reason": "fewer_than_3_valid_frames", "cardiac_target_fraction": None, "same_peak_exact": None, "selected_cardiac_pc": None},
        ]
        result = summarize_records(records, continuous=("cardiac_target_fraction",), boolean=("same_peak_exact",))
        self.assertEqual("SAX/s1", location_key(records[0]))
        self.assertEqual(3, result["overall"]["n_locations"])
        self.assertEqual(2, result["overall"]["continuous"]["cardiac_target_fraction"]["n"])
        self.assertAlmostEqual(.4, result["overall"]["continuous"]["cardiac_target_fraction"]["mean"])
        self.assertEqual(1, result["overall"]["boolean"]["same_peak_exact"]["count"])
        self.assertEqual(2, result["by_view"]["SAX"]["n_locations"])
        self.assertEqual(1, result["by_selected_pc"]["1"]["n_locations"])


if __name__ == "__main__":
    unittest.main()
