"""Behavioral regression coverage for read-only Change4 diagnostics."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("diagnose_change4_checkpoint", ROOT / "scripts" / "diagnose_change4_checkpoint.py")
assert SPEC and SPEC.loader
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)

from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation  # noqa: E402


class DiagnoseChange4CheckpointTest(unittest.TestCase):
    def test_dense_nudft_peak_recovers_known_respiratory_and_cardiac_sinusoids(self) -> None:
        times = torch.arange(50, dtype=torch.float64) * .17
        respiratory = torch.sin(2 * torch.pi * .35 * times)[:, None, None]
        cardiac = torch.sin(2 * torch.pi * 1.41 * times)[:, None, None]
        self.assertAlmostEqual(.35, diagnostic.dominant_peak_hz(respiratory, times), delta=.02)
        self.assertAlmostEqual(1.41, diagnostic.dominant_peak_hz(cardiac, times), delta=.02)

    def test_diagnostic_eval_mode_restores_only_canonical_inr_latent_api_without_updates(self) -> None:
        model = SourceFirstDynamicModel(torch.tensor([-4., -4., -4.]), torch.tensor([4., 4., 4.]), cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]), n_dynamic_frames=1, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
        observation = DynamicObservation(torch.rand(1, 4, 4), "SAX", "s", 0, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", 0.)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        model.eval()
        diagnostic.prepare_read_only_diagnostic_model(model)
        self.assertTrue(model.canonical.inr.training)
        self.assertFalse(model.film_encoder.training)
        with torch.no_grad():
            output = model.predict(observation, torch.tensor([[0., 0.]]), "stage3c")
        self.assertIn("uncertainty", output)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_frequency_record_reports_change5a_concentration_without_mutating_scores(self) -> None:
        times = torch.arange(50, dtype=torch.float64) * .2
        card = torch.sin(2 * torch.pi * (6 / 9.8) * times)[:, None, None].repeat(1, 1, 3)
        resp = torch.sin(2 * torch.pi * .2 * times)[:, None, None].repeat(1, 1, 3)
        before = card.clone()
        local = SimpleNamespace(cardiac_bands_hz=[(.60, .63)], respiratory_bands_hz=[(.15, .25)], respiratory_source="phase1_per_location", cardiac_source="phase1_per_location")
        pca_prior = SimpleNamespace(match=lambda view, slice_id, requested, **kwargs: SimpleNamespace(waveform=torch.sin(2 * torch.pi * .6 * requested), selected_pc=2))
        record = diagnostic.frequency_semantics_record("SAX", "s", resp, card, times, local, pca_waveform_prior=pca_prior)
        self.assertIn("cardiac_target_fraction", record)
        self.assertIn("cardiac_target_concentration", record)
        self.assertIn("cardiac_pca_waveform_r2", record)
        self.assertIn("cardiac_pca_waveform_loss", record)
        self.assertIn("cardiac_pca_matched_frames", record)
        self.assertEqual(2, record["selected_cardiac_pc"])
        self.assertTrue(torch.equal(before, card))


if __name__ == "__main__":
    unittest.main()
