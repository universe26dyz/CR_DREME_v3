"""One synthetic zero-motion Phase-2 forward/backward smoke test."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cardioresp4d.models.hash_inr import CanonicalINR  # noqa: E402
from cardioresp4d.models.bspline_mbc import RespiratoryMBC, CardiacMBC  # noqa: E402
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion  # noqa: E402
from cardioresp4d.models.uncertainty import SliceUncertainty  # noqa: E402
from cardioresp4d.rendering.thick_slice_renderer import ThickSliceRenderer  # noqa: E402


class Phase2SmokeTest(unittest.TestCase):
    def test_zero_motion_renderer_uncertainty_backward(self):
        inr = CanonicalINR(levels=2, features_per_level=2, hash_size=32, min_resolution=4, max_resolution=8, hidden_dim=12, latent_dim=6)
        lower, upper = torch.tensor([-100., -100., -100.]), torch.tensor([100., 100., 100.])
        respiration = ScoreWeightedMBCField(RespiratoryMBC(lower, upper, resolutions=(4, 4, 4)), torch.zeros(1, 3, 3))
        cardiac = ScoreWeightedMBCField(CardiacMBC(lower, upper, resolution=4), torch.zeros(1, 1, 3))
        motion = SequentialPullbackMotion(respiration, cardiac)
        renderer = ThickSliceRenderer(torch.eye(4)); uncertainty = SliceUncertainty(6, num_frames=1, embedding_dim=2)
        output = renderer(inr, center_mm=torch.zeros(1, 3), row_direction=torch.tensor([[1.,0.,0.]]), column_direction=torch.tensor([[0.,1.,0.]]), normal=torch.tensor([[0.,0.,1.]]), pixel_spacing_mm=torch.ones(1,2), thickness_mm=torch.tensor([8.]), height=4, width=4, motion=motion)
        variance = uncertainty(output["latent_samples"], torch.tensor([0]), output["quadrature_weights"])
        loss = output["predicted_slice"].square().mean() + variance["total_variance"].mean(); loss.backward()
        self.assertTrue(torch.isfinite(output["predicted_slice"]).all()); self.assertTrue(torch.isfinite(variance["total_variance"]).all())
        self.assertIsNotNone(inr.intensity_head.weight.grad)


if __name__ == "__main__": unittest.main()
