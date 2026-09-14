"""Red/green contracts for the v3_change3 stage-aware motion mainline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.losses.motion_loss import dreme_mbc_normalization  # noqa: E402
from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402
from cardioresp4d.geometry.world_geometry import DicomPlane  # noqa: E402
from cardioresp4d.models.film_motion_encoder import GeometryFiLMMotionEncoder  # noqa: E402


def observation(view: str, location: str, frame: int) -> DynamicObservation:
    return DynamicObservation(
        torch.rand(1, 4, 4), view, location, frame, torch.zeros(3),
        torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]),
        torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(frame),
    )


def model() -> SourceFirstDynamicModel:
    return SourceFirstDynamicModel(
        torch.tensor([-8., -8., -8.]), torch.tensor([8., 8., 8.]),
        cardiac_lower_world_mm=torch.tensor([-3., -3., -3.]),
        cardiac_upper_world_mm=torch.tensor([3., 3., 3.]), n_dynamic_frames=6,
        inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8,
        respiratory_grid_shapes=((4, 4, 4), (5, 5, 5), (6, 6, 6)),
        cardiac_grid_shape=(4, 4, 4), psf_samples=1,
    )


class StageContractTest(unittest.TestCase):
    def test_eq6_preserves_level_and_xyz_axes_before_scalar_reduction(self) -> None:
        # [batch, level, spatial, xyz]; each level/component has a unique RMS.
        values = torch.tensor(
            [[[[1., 2., .5], [1., 2., .5]], [[2., .5, 1.], [2., .5, 1.]]]],
        )
        expected = ((values.square().mean(dim=(0, 2)) - 1.).square()).mean()
        torch.testing.assert_close(dreme_mbc_normalization(values), expected)

    def test_active_raw_basis_skips_inactive_upstream_level_calls(self) -> None:
        subject = model()
        calls = [0, 0, 0]
        for index, level in enumerate(subject.respiratory_mbc.levels):
            original = level.forward
            def counted(points, *, _index=index, _original=original):
                calls[_index] += 1
                return _original(points)
            level.forward = counted  # type: ignore[method-assign]
        subject.respiratory_mbc.raw_active(torch.zeros(1, 2, 3), active_levels=1)
        self.assertEqual([1, 0, 0], calls)
        self.assertFalse(any("level_gates" in name for name, _ in subject.respiratory_mbc.named_parameters()))

    def test_stage1_calls_no_film_or_sinr_and_has_data_image_only(self) -> None:
        subject = model()
        calls = {"film": 0, "resp": 0, "card": 0}
        for name, module in (("film", subject.film_encoder), ("resp", subject.respiratory_mbc), ("card", subject.cardiac_mbc)):
            original = module.forward
            def counted(*args, _name=name, _original=original, **kwargs):
                calls[_name] += 1
                return _original(*args, **kwargs)
            module.forward = counted  # type: ignore[method-assign]
        items = [observation("SAX", "s", 0), observation("2CH", "t", 1), observation("4CH", "f", 2)]
        report = UnifiedProgressiveTrainer(subject, ViewLocationBalancedSampler(items), pixel_samples=2).run_stage("stage1", steps=1)
        self.assertEqual({"film": 0, "resp": 0, "card": 0}, calls)
        self.assertEqual({"data", "image"}, set(report["loss_components"]))

    def test_progressive_stage_call_ownership_never_runs_inactive_branches(self) -> None:
        subject = model(); calls = [0, 0, 0]; cardiac = [0]
        for index, level in enumerate(subject.respiratory_mbc.levels):
            original = level.forward
            def counted(points, *, _index=index, _original=original):
                calls[_index] += 1; return _original(points)
            level.forward = counted  # type: ignore[method-assign]
        original_cardiac = subject.cardiac_mbc.forward
        def counted_cardiac(*args, **kwargs):
            cardiac[0] += 1; return original_cardiac(*args, **kwargs)
        subject.cardiac_mbc.forward = counted_cardiac  # type: ignore[method-assign]
        item = observation("SAX", "s", 0)
        subject.predict(item, torch.tensor([[1., 1.]]), "stage2a")
        self.assertGreater(calls[0], 0); self.assertEqual([calls[0], 0, 0], calls); self.assertEqual(0, cardiac[0])
        calls[:] = [0, 0, 0]
        subject.predict(item, torch.tensor([[1., 1.]]), "stage2b")
        self.assertGreater(calls[0], 0); self.assertGreater(calls[1], 0); self.assertEqual(0, calls[2]); self.assertEqual(0, cardiac[0])
        calls[:] = [0, 0, 0]
        subject.predict(item, torch.tensor([[1., 1.]]), "stage3c")
        self.assertTrue(all(value > 0 for value in calls)); self.assertGreater(cardiac[0], 0)

    def test_oblique_dicom_plane_and_model_pixel_world_and_projected_extents_agree(self) -> None:
        row = torch.tensor([2**-.5, 0., 2**-.5])
        column = torch.tensor([0., 1., 0.])
        plane = DicomPlane([10., -2., 4.], row.tolist(), column.tolist(), [2., 3.], 5., 9, 11)
        center = torch.tensor(plane.pixel_to_world(5., 4.))
        item = DynamicObservation(torch.zeros(1, 9, 11), "SAX", "s", 0, center, row, column, torch.linalg.cross(row, column), torch.tensor([2., 3.]), 5., True, "valid")
        actual = model()._pixel_world(item, torch.tensor([[0., 0.], [10., 8.]]))
        expected = torch.from_numpy(np.stack((plane.pixel_to_world(0., 0.), plane.pixel_to_world(10., 8.)))).float()
        torch.testing.assert_close(actual, expected)
        encoder = GeometryFiLMMotionEncoder(channels=4, canonical_lower_world_mm=torch.tensor([-10., -10., -10.]), canonical_upper_world_mm=torch.tensor([10., 10., 10.]))
        scalars = encoder.acquisition_scalars(center[None], row[None], column[None], torch.linalg.cross(row, column)[None], torch.tensor([[2., 3.]]), torch.tensor([[5.]]))
        extent = torch.tensor([20., 20., 20.])
        expected_scalars = torch.log(torch.tensor([[3. / (row.abs() * extent).sum(), 2. / (column.abs() * extent).sum(), 5. / (torch.linalg.cross(row, column).abs() * extent).sum()]]))
        torch.testing.assert_close(scalars, expected_scalars)


if __name__ == "__main__":
    unittest.main()
