"""CPU contracts for Change5B local PCA waveform weak supervision."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformPrior  # noqa: E402
from cardioresp4d.losses.motion_loss import cardiac_pca_waveform_subspace_loss  # noqa: E402
from cardioresp4d.training.source_first_config import validate_source_first_config  # noqa: E402
from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402


def _write_location(root: Path, view: str, slice_id: str, timestamps: np.ndarray, pcs: np.ndarray, selected_pc: int, *, reliable: bool = True) -> Path:
    artifact = root / view / slice_id
    artifact.mkdir(parents=True)
    np.savez_compressed(artifact / "pca_psd.npz", timestamps_s=timestamps, temporal_pcs=pcs)
    payload = {"schema_version": 1, "cardiac": {"per_slice_candidates": [{"slice_key": f"{view}/{slice_id}", "reliable": reliable, "selected_pc": selected_pc}]}}
    path = root / "frequency_bands.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class PCAWaveformPriorTest(unittest.TestCase):
    def test_selected_pc_is_one_based_and_unreliable_has_no_global_substitute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); times = np.arange(10, dtype=float) * .2
            pcs = np.stack((times + 1., 10. - times), axis=1)
            path = _write_location(root, "SAX", "s1", times, pcs, 2)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cardiac"]["per_slice_candidates"].append({"slice_key": "2CH/unreliable", "reliable": False, "selected_pc": 1})
            path.write_text(json.dumps(payload), encoding="utf-8")
            prior = PCAWaveformPrior.load(path)
            match = prior.match("SAX", "s1", torch.tensor(times[[1, 4, 8]], dtype=torch.float64))
            self.assertTrue(torch.equal(match.waveform.cpu(), torch.tensor(pcs[[1, 4, 8], 1], dtype=torch.float64)))
            self.assertEqual(2, match.selected_pc)
            self.assertIsNone(prior.match("2CH", "other", torch.tensor(times[:8])))

    def test_timestamp_subset_is_strict_and_preserves_current_valid_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); times = np.arange(10, dtype=float) * .2; pcs = np.stack((times, times ** 2), axis=1)
            prior = PCAWaveformPrior.load(_write_location(root, "SAX", "s1", times, pcs, 1))
            matched = prior.match("SAX", "s1", torch.tensor([1.0, .2, 1.6], dtype=torch.float64))
            self.assertEqual([1.0, .2, 1.6], matched.timestamps_s.cpu().tolist())
            with self.assertRaises(ValueError):
                prior.match("SAX", "s1", torch.tensor([.201], dtype=torch.float64))

    def test_invalid_selected_pc_fails_without_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); times = np.arange(10, dtype=float); pcs = np.ones((10, 2))
            path = _write_location(root, "SAX", "s1", times, pcs, 3)
            with self.assertRaises(ValueError):
                PCAWaveformPrior.load(path)


class PCAWaveformSubspaceLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.scores = torch.randn(20, 3)
        self.target = self.scores @ torch.tensor([.4, -1.2, .7])

    def test_linear_target_has_sign_rotation_and_permutation_invariant_near_perfect_r2(self) -> None:
        direct = cardiac_pca_waveform_subspace_loss(self.scores, self.target)
        flipped = cardiac_pca_waveform_subspace_loss(self.scores, -self.target)
        rotated = cardiac_pca_waveform_subspace_loss(self.scores @ torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]), self.target)
        self.assertGreater(float(direct["r2"]), .999)
        self.assertAlmostEqual(float(direct["r2"]), float(flipped["r2"]), places=5)
        self.assertAlmostEqual(float(direct["r2"]), float(rotated["r2"]), places=5)

    def test_unrelated_constant_and_rank_deficient_inputs_are_finite(self) -> None:
        unrelated = cardiac_pca_waveform_subspace_loss(self.scores, torch.arange(20, dtype=torch.float32))
        constant = cardiac_pca_waveform_subspace_loss(torch.zeros(20, 3), self.target)
        rank_deficient = cardiac_pca_waveform_subspace_loss(self.scores[:, :1].repeat(1, 3), self.target)
        self.assertLess(float(unrelated["r2"]), .9)
        for result in (constant, rank_deficient):
            self.assertTrue(torch.isfinite(result["loss"]))
            self.assertLessEqual(float(result["r2"]), 1.)

    def test_backward_is_finite_and_minimum_frames_skip_is_explicit(self) -> None:
        scores = self.scores.clone().requires_grad_()
        result = cardiac_pca_waveform_subspace_loss(scores, self.target)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        skipped = cardiac_pca_waveform_subspace_loss(self.scores[:5], self.target[:5], min_frames=8)
        self.assertEqual(1., float(skipped["skipped"]))


class Change5BConfigTest(unittest.TestCase):
    def test_change5b_config_and_validation(self) -> None:
        baseline = yaml.safe_load((ROOT / "configs" / "source_first_change5a.yaml").read_text())
        validate_source_first_config(baseline, ROOT)
        config = yaml.safe_load((ROOT / "configs" / "source_first_change5b.yaml").read_text())
        validate_source_first_config(config, ROOT)
        self.assertEqual(.003, config["training"]["loss_weights"]["cardiac_target_concentration"])
        self.assertEqual(.001, config["training"]["loss_weights"]["cardiac_pca_waveform"])
        config["training"]["loss_weights"]["cardiac_pca_waveform"] = math.nan
        with self.assertRaises(ValueError):
            validate_source_first_config(config, ROOT)
        config = yaml.safe_load((ROOT / "configs" / "source_first_change5b.yaml").read_text())
        config["training"]["temporal_auxiliary"]["pca_waveform_ridge"] = math.nan
        with self.assertRaises(ValueError):
            validate_source_first_config(config, ROOT)
        config = yaml.safe_load((ROOT / "configs" / "source_first_change5b.yaml").read_text())
        config["training"]["temporal_auxiliary"]["pca_waveform_min_frames"] = 2
        with self.assertRaises(ValueError):
            validate_source_first_config(config, ROOT)


class Change5BTrainerTest(unittest.TestCase):
    def test_stage3a_pca_component_reaches_only_cardiac_film_head(self) -> None:
        model = SourceFirstDynamicModel(torch.tensor([-4., -4., -4.]), torch.tensor([4., 4., 4.]), cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]), n_dynamic_frames=24, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
        items = []
        for view in ("SAX", "2CH", "4CH"):
            for frame in range(8):
                items.append(DynamicObservation(torch.rand(1, 4, 4), view, "s", len(items), torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(frame)))
        local = type("Local", (), {"cardiac_bands_hz": [(.45, .55)], "respiratory_bands_hz": [(.1, .2)], "cardiac_baseline_pairs": [{"cardiac_band_hz": [.45, .55], "baseline_band_hz": [.25, .35]}]})()
        pca = type("PCA", (), {"match": lambda self, view, slice_id, timestamps, **kwargs: type("Match", (), {"waveform": torch.arange(timestamps.numel(), device=kwargs["device"], dtype=kwargs["dtype"]), "selected_pc": 1})()})()
        trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(items, seed=0), pixel_samples=1, frequency_prior=type("Prior", (), {"for_location": lambda self, view, slice_id: local})(), pca_waveform_prior=pca, loss_weights={"cardiac_pca_waveform": .1})
        trainer._configure_stage("stage3a")
        components = trainer._temporal_components("stage3a")
        self.assertGreater(float(components["cardiac_pca_waveform_matched_frames"]), 0.)
        self.assertNotIn("cardiac_pca_waveform", trainer._temporal_components("stage3b"))
        components["cardiac_pca_waveform"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.film_encoder.card.parameters()))
        for module in (model.canonical, model.respiratory_mbc, model.film_encoder.image, model.film_encoder.film, model.film_encoder.geometry_mlp, model.film_encoder.head, model.film_encoder.resp, model.uncertainty):
            self.assertFalse(any(parameter.grad is not None for parameter in module.parameters()))


if __name__ == "__main__":
    unittest.main()
