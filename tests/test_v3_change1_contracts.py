"""Second-round source-first contracts: semantics, not merely import success."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.adapters.nesvor_uncertainty import NeSVoRDynamicFrameUncertainty
from cardioresp4d.adapters.sinr_mbc import RespiratorySINRMBCAdapter, SINRFFDBasis
from cardioresp4d.losses.stage_aware import frequency_leakage
from cardioresp4d.models.film_motion_encoder import GeometryFiLMMotionEncoder
from cardioresp4d.training.source_first_config import validate_source_first_config


class V3Change1ContractsTest(unittest.TestCase):
    def test_vendored_source_lock_and_default_paths(self) -> None:
        lock = json.loads((ROOT / "third_party" / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual("vendored", lock["SINR"]["state"])
        self.assertEqual("1a524ca7ae453b55310595fe957245088a108233", lock["SINR"]["commit"])
        from cardioresp4d.adapters._upstream import add_upstream_to_path
        self.assertEqual(ROOT / "third_party" / "SINR", add_upstream_to_path("SINR"))

    def test_geometry_encoding_separates_units_and_validates_orientation(self) -> None:
        encoder = GeometryFiLMMotionEncoder(
            channels=8, canonical_lower_world_mm=torch.tensor([-100., -80., -60.]),
            canonical_upper_world_mm=torch.tensor([100., 80., 60.]), position_bands=3,
        )
        center = torch.tensor([[-100., -80., -60.], [0., 0., 0.], [100., 80., 60.]], requires_grad=True)
        row = torch.tensor([[1., 0., 0.]]).expand(3, -1)
        col = torch.tensor([[0., 1., 0.]]).expand(3, -1)
        normal = torch.tensor([[0., 0., 1.]]).expand(3, -1)
        encoded = encoder.encode_geometry(center, row, col, normal, torch.ones(3, 2), torch.ones(3, 1) * 6)
        torch.testing.assert_close(encoded[:, :3], torch.tensor([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]))
        wider = GeometryFiLMMotionEncoder(channels=8, canonical_lower_world_mm=torch.tensor([-1000., -800., -600.]), canonical_upper_world_mm=torch.tensor([1000., 800., 600.]), position_bands=3)
        torch.testing.assert_close(encoded[:, encoder.position_feature_dim:encoder.position_feature_dim + 9], wider.encode_geometry(center * 10, row, col, normal, torch.ones(3, 2), torch.ones(3, 1) * 6)[:, wider.position_feature_dim:wider.position_feature_dim + 9])
        with self.assertRaises(ValueError):
            encoder.encode_geometry(center[:1], row[:1] * 2, col[:1], normal[:1], torch.ones(1, 2), torch.ones(1, 1))
        encoded.sum().backward()

    def test_oblique_psf_variances_follow_dicom_basis(self) -> None:
        psf = NeSVoRPSFAdapter(n_samples=8192)
        centers = torch.zeros(1, 3)
        row = torch.tensor([[2**-0.5, 0., 2**-0.5]])
        column = torch.tensor([[0., 1., 0.]])
        normal = torch.linalg.cross(row, column)
        resolution = torch.tensor([[2., 3., 8.]])
        torch.manual_seed(7)
        offsets = psf.sample(centers, resolution, row_direction=row, column_direction=column, normal=normal)[0]
        projected = torch.stack((offsets @ row[0], offsets @ column[0], offsets @ normal[0]), dim=-1)
        expected = psf.sigma_mm(resolution)[0].square()
        torch.testing.assert_close(projected.var(0, unbiased=True), expected, rtol=0.08, atol=0.02)
        self.assertEqual(int(projected.var(0).argmax()), 2)

    def test_sinr_logical_control_shape_is_not_dense_img_size(self) -> None:
        basis = SINRFFDBasis(torch.tensor([-8., -8., -8.]), torch.tensor([8., 8., 8.]), logical_control_shape=(8, 8, 8), cps=2, hidden_dim=8)
        self.assertEqual((8, 8, 8), basis.logical_control_shape)
        self.assertEqual((10, 10, 10), basis.padded_control_shape)
        self.assertEqual((16, 16, 16), basis.dense_evaluation_shape)
        self.assertEqual((16, 16, 16), tuple(basis.ffd.img_size))
        resp = RespiratorySINRMBCAdapter(torch.tensor([-8., -8., -8.]), torch.tensor([8., 8., 8.]), hidden_dim=8)
        self.assertEqual([(8, 8, 8), (12, 12, 12), (16, 16, 16)], [x.logical_control_shape for x in resp.levels])
        self.assertEqual(0, int(resp.active_level_count))
        self.assertFalse(any("level_gates" in name for name, _ in resp.named_parameters()))

    def test_uncertainty_aggregates_scale_before_squaring(self) -> None:
        uncertainty = NeSVoRDynamicFrameUncertainty(latent_dim=2, n_dynamic_frames=1, frame_embedding_dim=1, width=4, depth=0)
        uncertainty.sigma_net = torch.nn.Sequential(torch.nn.Linear(3, 1, bias=False))
        uncertainty.sigma_net[0].weight.data.copy_(torch.tensor([[1., 0., 0.]]))
        uncertainty.frame_embedding.weight.data.zero_()
        uncertainty.log_var_frame.data.fill_(torch.log(torch.tensor(0.5)))
        latent = torch.tensor([[[0.0, 0.0], [torch.log(torch.tensor(3.0)), 0.0]]])
        output = uncertainty(latent, torch.tensor([0]))
        expected = ((1.0 + 3.0) / 2.0) ** 2 + 0.5
        torch.testing.assert_close(output["variance"], torch.tensor([expected]))

    def test_irregular_timestamp_loss_is_finite_for_empty_and_valid_bands(self) -> None:
        scores = torch.randn(4, 3, requires_grad=True)
        timestamps = torch.tensor([0.0, 0.11, 0.31, 0.57])
        zero = frequency_leakage(scores, timestamps, (9.0, 10.0))
        self.assertEqual(float(zero), 0.0)
        value = frequency_leakage(scores, timestamps, (1.0, 3.0))
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNotNone(scores.grad)

    def test_config_rejects_cardiac_crop_and_stage2_late_uncertainty(self) -> None:
        import yaml
        config = yaml.safe_load((ROOT / "configs" / "source_first.yaml").read_text(encoding="utf-8")); config["domain"]["cardiac_box_is_crop"] = True
        with self.assertRaises(ValueError):
            validate_source_first_config(config, ROOT)


if __name__ == "__main__":
    unittest.main()
