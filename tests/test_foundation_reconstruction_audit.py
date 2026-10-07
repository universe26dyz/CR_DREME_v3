"""Read-only foundation-reconstruction audit contracts."""
from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace

import torch


class _Observation:
    def __init__(self) -> None:
        self.image = torch.tensor([[[1., 2.], [3., 4.]]])
        self.row_direction = torch.tensor([1., 0., 0.])
        self.column_direction = torch.tensor([0., 1., 0.])
        self.normal = torch.tensor([0., 0., 1.])
        self.pixel_spacing_mm = torch.ones(2)
        self.slice_thickness_mm = 1.
        self.view, self.slice_id, self.dynamic_frame_id = "SAX", "s001", 5
        self.timestamp_s = 0.


class _Canonical(torch.nn.Module):
    def forward(self, points, return_features: bool = False):
        values = points[..., 0] + 10 * points[..., 1]
        return (values, torch.zeros(*values.shape, 1)) if return_features else values


class _PSF:
    def __init__(self) -> None:
        self.motions: list[object] = []
        self.random: list[torch.Tensor] = []

    def __call__(self, canonical, centers, resolution, *, row_direction, column_direction, normal, motion=None):
        self.motions.append(motion); self.random.append(torch.rand(2))
        return {"predicted_intensity": canonical(centers)}


class _Model:
    def __init__(self) -> None:
        self.canonical, self.psf, self.calls = _Canonical(), _PSF(), []

    @staticmethod
    def _pixel_world(observation, pixels):
        return torch.cat((pixels, torch.zeros(len(pixels), 1)), dim=-1)

    def predict(self, observation, pixels, stage):
        self.calls.append((stage, torch.rand(2)))
        value = self.canonical(self._pixel_world(observation, pixels))
        return {"predicted_intensity": value + (100. if stage == "stage3a" else 10.)}


class FoundationReconstructionAuditTest(unittest.TestCase):
    def test_foundation_predictions_keep_direct_psf_resp_and_joint_contracts(self) -> None:
        module = importlib.import_module("cardioresp4d.diagnostics.foundation")
        subject, observation = _Model(), _Observation()
        result = module.foundation_predictions(subject, observation, joint_stage="stage3a", seed=7, slice_chunk_size=2)
        self.assertTrue(torch.equal(result["canonical_direct"], torch.tensor([[0., 1.], [10., 11.]])))
        self.assertTrue(torch.equal(result["canonical_psf"], result["canonical_direct"]))
        self.assertTrue(torch.equal(result["resp_only"], result["canonical_direct"] + 10.))
        self.assertTrue(torch.equal(result["resp_plus_card"], result["canonical_direct"] + 100.))
        self.assertEqual([None, None], subject.psf.motions)
        self.assertEqual(["stage2c", "stage2c", "stage3a", "stage3a"], [stage for stage, _ in subject.calls])
        self.assertTrue(torch.equal(subject.calls[0][1], subject.calls[2][1]))
        self.assertTrue(torch.equal(subject.calls[1][1], subject.calls[3][1]))

    def test_frame_and_temporal_metrics_keep_regions_and_timestamp_order(self) -> None:
        module = importlib.import_module("cardioresp4d.diagnostics.foundation")
        acquired = torch.tensor([[1., 3.], [2., 4.]])
        predicted = torch.tensor([[1., 1.], [2., 2.]])
        metrics = module.image_metrics(acquired, predicted, torch.tensor([[True, False], [True, False]]))
        self.assertEqual(0., metrics["cardiac_intersection"]["mse"])
        self.assertGreater(metrics["whole_fov"]["mse"], 0.)
        temporal = module.temporal_metrics(torch.stack((acquired, acquired + 1., acquired + 3.)), torch.stack((predicted, predicted + 1., predicted + 3.)), torch.tensor([[True, False], [True, False]]))
        self.assertGreater(temporal["whole_fov"]["mean_temporal_std"], 0.)
        self.assertAlmostEqual(1., temporal["whole_fov"]["delta_correlation"], places=6)

    def test_shared_psf_seed_keeps_static_canonical_temporally_constant(self) -> None:
        module = importlib.import_module("cardioresp4d.diagnostics.foundation")
        first, second = _Observation(), _Observation()
        second.dynamic_frame_id, second.timestamp_s = 6, 1.
        model = _Model()
        shared_seed = 123
        one = module.foundation_predictions(model, first, joint_stage="stage3a", seed=7, psf_seed=shared_seed)
        two = module.foundation_predictions(model, second, joint_stage="stage3a", seed=7, psf_seed=shared_seed)
        self.assertEqual(0., float(module.temporal_std_map(torch.stack((one["canonical_psf"], two["canonical_psf"]))).max()))


if __name__ == "__main__":
    unittest.main()
