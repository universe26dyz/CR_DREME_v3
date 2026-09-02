"""Regression tests for the final Phase-1 provenance, privacy and consensus review."""
from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.config import config_from_mapping
from cardioresp4d.frequency.frequency_bands import aggregate_frequency_bands
from cardioresp4d.geometry.coordinate_normalization import WorldNormalizer
from cardioresp4d.geometry.world_geometry import DicomPlane
from tests.test_data import nested_config_mapping


def candidate_result(freq, reliable=True, slice_key="SAX/s1"):
    candidate = {"frequency_hz": freq, "selected_pc": 1 if freq else None,
                 "peak_power": 1.0 if freq else None, "dominance": 3.0 if freq else None,
                 "reliable": reliable, "reason": "ok" if reliable else "null"}
    return {"slice_key": slice_key, "median_dt_s": .171, "duration_span_s": 8.363,
            "duration_n_dt_s": 8.55, "df_hz": .1, "nyquist_hz": 2.9,
            "explained_variance": [.5], "respiratory_candidate": candidate,
            "cardiac_candidate": dict(candidate)}


class FinalReviewTest(unittest.TestCase):
    def test_unknown_keys_are_rejected_at_root_and_nested_levels(self):
        with tempfile.TemporaryDirectory() as d:
            mapping = nested_config_mapping(Path(d)/"dicom", Path(d)/"results")
            mapping["surprise"] = {}
            with self.assertRaisesRegex(ValueError, "unknown"):
                config_from_mapping(mapping, Path(d))
            mapping = nested_config_mapping(Path(d)/"dicom", Path(d)/"results")
            mapping["frequency"]["dominence_threshold"] = 2.2
            with self.assertRaisesRegex(ValueError, "unknown"):
                config_from_mapping(mapping, Path(d))

    def test_frequency_verified_band_requires_half_of_all_eligible_slices(self):
        one = candidate_result(.3, True, "a")
        null = candidate_result(None, False, "b")
        bands = aggregate_frequency_bands([one, null], consensus_min_slice_fraction=.5)
        self.assertIsNone(bands["respiratory"]["verified_band_hz"])
        bands = aggregate_frequency_bands([one, null, null], consensus_min_slice_fraction=.5)
        self.assertIsNone(bands["respiratory"]["verified_band_hz"])
        support = bands["respiratory"]["resolution_bin_support"]
        self.assertEqual(1, support[0]["count"])
        self.assertEqual(3, support[0]["total"])
        self.assertAlmostEqual(1/3, support[0]["fraction"])

    def test_scalar_nan_and_inf_are_rejected_before_affine_math(self):
        plane = DicomPlane([0,0,0], [1,0,0], [0,1,0], [1,1], 2, 2)
        for bad in ([np.nan, 0, 0], [np.inf, 0, 0]):
            with self.assertRaises(ValueError): plane.world_to_pixel(bad)
        normalizer = WorldNormalizer.from_corners([[0,0,0], [1,1,1]])
        for bad in ([np.nan, 0, 0], [np.inf, 0, 0]):
            with self.assertRaises(ValueError): normalizer.normalize(bad)


if __name__ == "__main__": unittest.main()
