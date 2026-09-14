"""Regression coverage for v3_change4 local Phase-1 frequency evidence."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.frequency.training_prior import load_training_frequency_prior  # noqa: E402
from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402
import cardioresp4d.training.trainer as trainer_module  # noqa: E402


def _payload() -> dict:
    return {
        "schema_version": 1,
        "respiratory": {
            "verified_band_hz": [[.2923976608, .4093567251]],
            "per_slice_candidates": [
                {"slice_key": "SAX/s001", "reliable": True, "frequency_hz": .233918, "df_hz": .116959},
                {"slice_key": "SAX/s002", "reliable": True, "frequency_hz": .350877, "df_hz": .116959},
            ],
        },
        "cardiac": {
            "union_resolution_bins_hz": [[1.0, 1.2]],
            "per_slice_candidates": [
                {"slice_key": "SAX/s001", "reliable": True, "frequency_hz": 1.20, "df_hz": .20},
                {"slice_key": "SAX/s002", "reliable": True, "frequency_hz": 1.40, "df_hz": .20},
            ],
        },
    }


def _prior(payload: dict | None = None):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frequency_bands.json"
        path.write_text(json.dumps(_payload() if payload is None else payload), encoding="utf-8")
        return load_training_frequency_prior(path)


def _observation(frame: int, timestamp: float, *, view: str = "SAX") -> DynamicObservation:
    return DynamicObservation(
        image=torch.rand(1, 4, 4), view=view, slice_id="s001", dynamic_frame_id=frame,
        center_mm=torch.zeros(3), row_direction=torch.tensor([1., 0., 0.]),
        column_direction=torch.tensor([0., 1., 0.]), normal=torch.tensor([0., 0., 1.]),
        pixel_spacing_mm=torch.ones(2), slice_thickness_mm=4., qc_valid=True,
        qc_reason="valid", timestamp_s=timestamp,
    )


def _model() -> SourceFirstDynamicModel:
    return SourceFirstDynamicModel(
        torch.tensor([-4., -4., -4.]), torch.tensor([4., 4., 4.]),
        cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]),
        n_dynamic_frames=3, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8,
        respiratory_grid_shapes=((4, 4, 4), (4, 4, 4), (4, 4, 4)), cardiac_grid_shape=(4, 4, 4), psf_samples=1,
    )


class Change4FrequencyPriorTest(unittest.TestCase):
    def test_location_prior_keeps_independent_respiratory_and_cardiac_provenance(self) -> None:
        prior = _prior()
        first = prior.for_location("SAX", "s001")
        second = prior.for_location("SAX", "s002")
        self.assertAlmostEqual(.1754385, first.respiratory_bands_hz[0][0], places=7)
        self.assertAlmostEqual(.2923975, first.respiratory_bands_hz[0][1], places=7)
        self.assertAlmostEqual(.2923975, second.respiratory_bands_hz[0][0], places=7)
        self.assertAlmostEqual(.4093565, second.respiratory_bands_hz[0][1], places=7)
        self.assertNotEqual(first.respiratory_bands_hz, second.respiratory_bands_hz)
        self.assertEqual("phase1_per_location", first.respiratory_source)
        self.assertEqual("phase1_per_location", first.cardiac_source)
        self.assertAlmostEqual(.233918 - .116959 / 2., first.respiratory_bands_hz[0][0], places=12)
        self.assertAlmostEqual(.233918 + .116959 / 2., first.respiratory_bands_hz[0][1], places=12)
        self.assertAlmostEqual(1.1, first.cardiac_bands_hz[0][0], places=12)
        self.assertAlmostEqual(1.3, first.cardiac_bands_hz[0][1], places=12)
        self.assertIn("respiratory_source", asdict(prior)["locations"]["SAX/s001"])

    def test_local_modalities_fallback_independently_and_invalid_reliable_candidates_fail(self) -> None:
        payload = _payload()
        payload["respiratory"]["per_slice_candidates"][1]["reliable"] = False
        payload["cardiac"]["per_slice_candidates"][0]["reliable"] = False
        prior = _prior(payload)
        resp_local_card_global = prior.for_location("SAX", "s001")
        resp_global_card_local = prior.for_location("SAX", "s002")
        self.assertEqual("phase1_per_location", resp_local_card_global.respiratory_source)
        self.assertEqual("phase1_global_fallback", resp_local_card_global.cardiac_source)
        self.assertEqual("phase1_global_fallback", resp_global_card_local.respiratory_source)
        self.assertEqual("phase1_per_location", resp_global_card_local.cardiac_source)
        payload = _payload()
        payload["respiratory"]["per_slice_candidates"][0].pop("df_hz")
        with self.assertRaisesRegex(ValueError, "reliable"):
            _prior(payload)

    def test_trainer_eq9_passes_resolved_location_respiratory_bands(self) -> None:
        prior = _prior()
        sequence = [_observation(index, timestamp) for index, timestamp in enumerate((0., .4, .9))]
        sampler_rows = sequence + [_observation(0, 0., view="2CH"), _observation(1, .4, view="4CH")]
        trainer = UnifiedProgressiveTrainer(_model(), ViewLocationBalancedSampler(sampler_rows), pixel_samples=1, frequency_prior=prior)
        trainer.sampler.temporal_batch = lambda **_: sequence  # type: ignore[method-assign]
        captured: list[list[tuple[float, float]]] = []
        original = trainer_module.dreme_respiratory_leakage_in_card

        def capture(scores, timestamps, bands):
            captured.append(list(bands))
            return original(scores, timestamps, bands)

        trainer_module.dreme_respiratory_leakage_in_card = capture
        try:
            trainer._temporal_components("stage3b")
        finally:
            trainer_module.dreme_respiratory_leakage_in_card = original
        self.assertEqual(1, len(captured))
        self.assertEqual(1, len(captured[0]))
        self.assertAlmostEqual(.233918 - .116959 / 2., captured[0][0][0], places=12)
        self.assertAlmostEqual(.233918 + .116959 / 2., captured[0][0][1], places=12)


if __name__ == "__main__":
    unittest.main()
