import torch

from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import PhantomCanonicalAdapter


def test_asymmetric_phantom_has_distinct_intensities_at_mirrored_points() -> None:
    """Would fail if the canonical phantom regressed to a symmetric shape."""
    phantom = AnalyticCanonicalPhantom()
    points = torch.tensor([[[18.0, -8.0, 4.0], [-18.0, -8.0, 4.0]]])

    values = phantom(points)

    assert values.shape == (1, 2, 1)
    assert torch.isfinite(values).all()
    assert float((values[0, 0] - values[0, 1]).abs()) > 1e-3
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0


def test_identity_motion_preserves_nesvor_psf_rendering() -> None:
    """Would fail if the synthetic renderer changed identity pullback coordinates."""
    canonical = PhantomCanonicalAdapter(AnalyticCanonicalPhantom())
    adapter = NeSVoRPSFAdapter(n_samples=2)
    centers = torch.tensor([[3.0, -2.0, 1.0], [-4.0, 5.0, -3.0]])
    resolution = torch.full((2, 3), 1.5)
    row = torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1)
    column = torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1)
    normal = torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1)

    torch.manual_seed(13)
    reference = adapter(canonical, centers, resolution, row_direction=row, column_direction=column, normal=normal)["predicted_intensity"]
    torch.manual_seed(13)
    identity = adapter(canonical, centers, resolution, row_direction=row, column_direction=column, normal=normal, motion=lambda points: {"reference_points_mm": points})["predicted_intensity"]

    torch.testing.assert_close(identity, reference)
