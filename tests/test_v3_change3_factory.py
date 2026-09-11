"""Red/green runtime factory contracts: requested config must be actual objects."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.training.build_model import build_source_first_model, effective_model_config  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402


class SourceFirstFactoryTest(unittest.TestCase):
    def test_config_perturbations_change_constructed_source_objects_and_effective_config(self) -> None:
        config = yaml.safe_load((ROOT / "configs" / "source_first.yaml").read_text())
        config = copy.deepcopy(config)
        config["model"]["psf"]["n_samples"] = 3
        config["model"]["canonical"]["latent_dim"] = 5
        config["model"]["respiratory_mbc"]["hidden_dim"] = 11
        config["model"]["respiratory_mbc"]["cps"] = 3
        config["model"]["cardiac_mbc"]["taper_mm"] = 2.5
        config["model"]["film"] = {"channels": 9, "position_bands": 2}
        config["model"]["uncertainty"].update({"frame_embedding_dim": 6, "width": 10, "depth": 1})
        config["training"]["motion_regularization"] = {
            "respiratory_evaluation_grid": {"shape": [7, 8, 9]},
            "cardiac_evaluation_grid": {"shape": [5, 6, 7]},
        }
        domain = {"world_min_mm": [-10., -11., -12.], "world_max_mm": [10., 11., 12.], "cardiac_box": {"min_mm": [-3., -4., -5.], "max_mm": [3., 4., 5.]}}
        subject = build_source_first_model(config, domain, n_dynamic_frames=4, device=torch.device("cpu"))
        effective = effective_model_config(subject)
        self.assertEqual(3, subject.psf.n_samples)
        self.assertEqual(5, subject.canonical.args.n_features_z)
        self.assertEqual(11, subject.respiratory_mbc.levels[0].siren.layers[0].out_features)
        self.assertEqual((3, 3, 3), subject.respiratory_mbc.levels[0].cps)
        self.assertEqual(2.5, subject.cardiac_mbc.taper_mm)
        self.assertEqual(9, subject.film_encoder.image[0].out_channels)
        self.assertEqual(6, subject.uncertainty.frame_embedding.embedding_dim)
        self.assertEqual([7, 8, 9], effective["smoothness"]["respiratory_shape"])
        self.assertEqual([5, 6, 7], effective["smoothness"]["cardiac_shape"])

    def test_stage_optimizer_groups_apply_configured_lrs_and_keep_canonical_state(self) -> None:
        subject = build_source_first_model(yaml.safe_load((ROOT / "configs" / "source_first.yaml").read_text()), {"world_min_mm": [-4., -4., -4.], "world_max_mm": [4., 4., 4.], "cardiac_box": {"min_mm": [-2., -2., -2.], "max_mm": [2., 2., 2.]}}, n_dynamic_frames=3, device=torch.device("cpu"))
        rows = [DynamicObservation(torch.rand(1, 3, 3), view, view, index, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(index)) for index, view in enumerate(("SAX", "2CH", "4CH"))]
        lrs = {"stage1": {"canonical_lr": .004}, "stage2": {"canonical_lr": .003, "film_lr": .002, "respiratory_mbc_lr": .001}, "stage3": {"canonical_lr": .0007, "film_lr": .0006, "respiratory_mbc_lr": .0005, "cardiac_mbc_lr": .0004, "uncertainty_lr": .0003}}
        trainer = UnifiedProgressiveTrainer(subject, ViewLocationBalancedSampler(rows), pixel_samples=1, optimizer_config=lrs)
        trainer.run_stage("stage1", steps=1)
        canonical_state = next(iter(trainer.optimizer.state.values()))["exp_avg"].clone()
        trainer._configure_stage("stage2a")
        groups = {group["name"]: group["lr"] for group in trainer.optimizer.param_groups}
        self.assertEqual({"canonical": .003, "film": .002, "respiratory_mbc": .001}, groups)
        self.assertTrue(any(torch.equal(state["exp_avg"], canonical_state) for state in trainer.optimizer.state.values()))


if __name__ == "__main__":
    unittest.main()
