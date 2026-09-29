"""CPU contracts for final Change5C pairing, provenance, and summaries."""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.diagnostics.change5c import (  # noqa: E402
    aggregate_ablation_by_view,
    diagnostic_rng,
    paired_reconstruction_seeds,
    provenance_matches,
    quiver_subsample_indices,
    select_visualization_locations_paired,
    world_to_grid_index,
    stable_diagnostic_seed,
)


class Change5CHotfixTest(unittest.TestCase):
    def test_psf_seed_is_stable_and_experiment_label_independent(self) -> None:
        first = stable_diagnostic_seed(7, "SAX", "SAX_s026", 42, purpose="psf")
        self.assertEqual(first, stable_diagnostic_seed(7, "SAX", "SAX_s026", 42, purpose="psf"))
        self.assertNotEqual(first, stable_diagnostic_seed(8, "SAX", "SAX_s026", 42, purpose="psf"))
        self.assertEqual((first, first), paired_reconstruction_seeds(7, "CTRL_2500", "SAX", "SAX_s026", 42))

    def test_rng_context_pairs_realizations_and_restores_global_rng(self) -> None:
        torch.manual_seed(123); random.seed(456)
        expected_torch, expected_python = torch.rand(3), random.random()
        torch.manual_seed(123); random.seed(456)
        with diagnostic_rng(9, torch.device("cpu")):
            resp = torch.rand(4)
        with diagnostic_rng(9, torch.device("cpu")):
            joint = torch.rand(4)
        self.assertTrue(torch.equal(resp, joint))
        self.assertTrue(torch.equal(expected_torch, torch.rand(3)))
        self.assertEqual(expected_python, random.random())

    def test_per_view_summary_separates_views(self) -> None:
        rows = [
            {"view": "SAX", "relative_mse_improvement_percent": 2., "mean_abs_joint_minus_resp": .2, "resp_only_mse": 1., "resp_plus_card_mse": .8},
            {"view": "2CH", "relative_mse_improvement_percent": -2., "mean_abs_joint_minus_resp": .1, "resp_only_mse": 1., "resp_plus_card_mse": 1.2},
        ]
        self.assertEqual(1, aggregate_ablation_by_view(rows)["SAX"]["n_positive_gain"])
        self.assertEqual(1, aggregate_ablation_by_view(rows)["2CH"]["n_negative_gain"])

    def test_reconstruction_extremes_use_paired_ctrl_to_c5b_delta(self) -> None:
        semantic = [{"view": "SAX", "slice_id": "a", "cardiac_pca_waveform_r2": .1}, {"view": "SAX", "slice_id": "b", "cardiac_pca_waveform_r2": .2}]
        ctrl = [{"view": "SAX", "slice_id": "a", "relative_mse_improvement_percent": 10., "mean_abs_joint_minus_resp": .1}, {"view": "SAX", "slice_id": "b", "relative_mse_improvement_percent": 0., "mean_abs_joint_minus_resp": .2}]
        c5b = [{"view": "SAX", "slice_id": "a", "relative_mse_improvement_percent": 5., "mean_abs_joint_minus_resp": .3}, {"view": "SAX", "slice_id": "b", "relative_mse_improvement_percent": 8., "mean_abs_joint_minus_resp": 0.}]
        result = select_visualization_locations_paired(semantic, semantic, ctrl, c5b, fixed=[("4CH", "missing")])
        selected = {(row["view"], row["slice_id"]): row["reasons"] for row in result["locations"]}
        self.assertIn("largest_c5b_reconstruction_gain_improvement_vs_ctrl", selected[("SAX", "b")])
        self.assertEqual("fixed_location_not_valid_in_all_inputs", result["omitted_fixed"][0]["reason"])

    def test_provenance_reuse_requires_complete_exact_match(self) -> None:
        expected = {"git_commit": "a", "seed": 0, "inputs": {"checkpoint": "x"}}
        self.assertTrue(provenance_matches(expected, {**expected, "completion": True}))
        self.assertFalse(provenance_matches(expected, {"git_commit": "b", "seed": 0, "inputs": {"checkpoint": "x"}, "completion": True}))
        self.assertFalse(provenance_matches(expected, expected))

    def test_world_center_and_quiver_sampling_are_deterministic(self) -> None:
        self.assertEqual((2, 2, 2), world_to_grid_index(torch.tensor([5., 5., 5.]), torch.zeros(3), torch.full((3,), 10.), (5, 5, 5)))
        self.assertEqual([0, 3, 6, 9], quiver_subsample_indices(10, 4))


if __name__ == "__main__":
    unittest.main()
