"""CPU contracts for Change5C read-only audit helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.diagnostics.change5c import (  # noqa: E402
    ablation_record,
    aggregate_ablation,
    cardiac_effect,
    choose_representative_indices,
    cosine_similarity,
    deterministic_pixel_indices,
    jacobian_summary,
    paired_ablation_comparison,
    select_visualization_locations,
    summarize_gradient_vectors,
)


class Change5CDiagnosticsTest(unittest.TestCase):
    def test_representative_indices_are_full_span_and_deterministic(self) -> None:
        self.assertEqual([0, 3, 7, 10], choose_representative_indices(11, 4))
        self.assertEqual([0, 1, 2], choose_representative_indices(3, 9))

    def test_pixel_sampling_has_reproducible_80_20_priority_global_policy(self) -> None:
        first = deterministic_pixel_indices(100, torch.tensor([0, 1]), count=10, cardiac_fraction=.8, seed=12)
        second = deterministic_pixel_indices(100, torch.tensor([0, 1]), count=10, cardiac_fraction=.8, seed=12)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(10, first.numel())
        self.assertEqual(8, int(((first == 0) | (first == 1)).sum()))

    def test_gradient_summary_uses_vectors_and_marks_zero_cosine_undefined(self) -> None:
        summary = summarize_gradient_vectors({"data": torch.tensor([3., 4.]), "aux": torch.tensor([0., 0.])})
        self.assertEqual(5., summary["data"]["l2_norm"])
        self.assertIsNone(cosine_similarity(torch.ones(2), torch.zeros(2)))

    def test_cardiac_effect_is_joint_minus_resp_only(self) -> None:
        self.assertTrue(torch.equal(torch.tensor([.5, -.5]), cardiac_effect(torch.tensor([1., 2.]), torch.tensor([1.5, 1.5]))))

    def test_ablation_aggregation_and_pairing_are_exact_keyed(self) -> None:
        left = [ablation_record("SAX", "s1", 1., .5, torch.tensor([.1, .2])), ablation_record("2CH", "s2", 1., 1., torch.tensor([0., 0.]))]
        right = [ablation_record("SAX", "s1", 1., .25, torch.tensor([.2, .4])), ablation_record("4CH", "s3", 1., .5, torch.tensor([.1]))]
        aggregate = aggregate_ablation(left, tolerance=1e-8)
        self.assertEqual(1, aggregate["n_positive_gain"])
        self.assertEqual(1, aggregate["n_zero_gain"])
        comparison = paired_ablation_comparison(left, right)
        self.assertEqual(["SAX/s1"], comparison["paired_location_keys"])
        self.assertEqual(["2CH/s2"], comparison["missing_from_right"])

    def test_jacobian_and_location_selection_are_structured_and_deduplicated(self) -> None:
        summary = jacobian_summary(torch.tensor([-1., 0., 1., 2., 3.]))
        self.assertAlmostEqual(.4, summary["fraction_leq_zero"])
        chosen = select_visualization_locations(
            [{"view": "SAX", "slice_id": "s1", "cardiac_pca_waveform_r2": .2}, {"view": "2CH", "slice_id": "s2", "cardiac_pca_waveform_r2": .1}],
            [{"view": "SAX", "slice_id": "s1", "cardiac_pca_waveform_r2": .3}, {"view": "2CH", "slice_id": "s2", "cardiac_pca_waveform_r2": .0}],
            [{"view": "SAX", "slice_id": "s1", "relative_mse_improvement_percent": 5., "mean_abs_joint_minus_resp": .3}, {"view": "2CH", "slice_id": "s2", "relative_mse_improvement_percent": -3., "mean_abs_joint_minus_resp": 0.}],
            fixed=[("SAX", "s1")],
        )
        self.assertEqual(2, len(chosen))
        sax = next(item for item in chosen if item["view"] == "SAX")
        self.assertIn("fixed_representative", sax["reasons"])


if __name__ == "__main__":
    unittest.main()
