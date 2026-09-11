"""Red/green DREME frequency contracts with real Phase-1 location identity."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.frequency.training_prior import load_training_frequency_prior  # noqa: E402
from cardioresp4d.losses.frequency_loss import dreme_cardiac_leakage_in_resp, nonuniform_dft_at_frequencies  # noqa: E402


class FrequencyContractTest(unittest.TestCase):
    def test_reliable_phase1_slice_key_selects_location_prior_and_unreliable_uses_explicit_global_fallback(self) -> None:
        payload = {
            "schema_version": 1,
            "respiratory": {"verified_band_hz": [[.2, .3]]},
            "cardiac": {
                "union_resolution_bins_hz": [[1.0, 1.2], [1.8, 2.0]],
                "per_slice_candidates": [
                    {"slice_key": "SAX/s1", "reliable": True, "frequency_hz": 1.1, "df_hz": .2},
                    {"slice_key": "SAX/s2", "reliable": False, "frequency_hz": None, "df_hz": .2},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frequency_bands.json"
            path.write_text(json.dumps(payload))
            prior = load_training_frequency_prior(path)
        local = prior.for_location("SAX", "s1")
        fallback = prior.for_location("SAX", "s2")
        self.assertAlmostEqual(1.0, local.cardiac_bands_hz[0][0])
        self.assertAlmostEqual(1.2, local.cardiac_bands_hz[0][1])
        self.assertEqual("phase1_per_location", local.source)
        self.assertEqual([(1.0, 1.2), (1.8, 2.0)], fallback.cardiac_bands_hz)
        self.assertEqual("phase1_global_fallback", fallback.source)
        self.assertEqual(1, len(local.cardiac_baseline_pairs))

    def test_eq8_uses_complex_paired_subtraction_not_difference_of_magnitudes(self) -> None:
        timestamps = torch.tensor([0., .17, .43, .91], dtype=torch.float64)
        scores = torch.tensor([[1.], [.3], [-.7], [.2]])
        pairs = [{"cardiac_band_hz": [.8, .8], "baseline_band_hz": [1.7, 1.7], "resolution_hz": 1.}]
        value = dreme_cardiac_leakage_in_resp(scores, timestamps, pairs)
        cardiac = nonuniform_dft_at_frequencies(scores, timestamps, [.8])
        baseline = nonuniform_dft_at_frequencies(scores, timestamps, [1.7])
        torch.testing.assert_close(value, (cardiac - baseline).abs().square().mean())

    def test_nudft_is_timestamp_offset_and_sample_count_stable_and_centres_dc(self) -> None:
        times = torch.linspace(0., 1., 8, dtype=torch.float64)
        signal = torch.sin(2 * torch.pi * times)[:, None] + 5.
        one = nonuniform_dft_at_frequencies(signal, times, [1.])
        two = nonuniform_dft_at_frequencies(signal, times + 40000., [1.])
        torch.testing.assert_close(one, two, atol=1e-5, rtol=1e-5)
        constant = nonuniform_dft_at_frequencies(torch.ones(8, 1), times, [1.])
        self.assertLess(float(constant.abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
