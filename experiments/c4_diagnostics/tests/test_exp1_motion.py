import torch

from cardioresp4d.models.cardioresp_motion import SequentialPullbackMotion
from experiments.c4_diagnostics.synthetic.metrics import location_spectral_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion


def test_known_respiratory_sinusoid_has_configured_dominant_frequency() -> None:
    """Would fail if time units or respiratory frequency were wired incorrectly."""
    motion = GroundTruthMotion(respiratory_frequency_hz=0.25, cardiac_frequency_hz=1.10)
    timestamps = torch.arange(0.0, 40.0, 0.05)

    peak_hz = location_spectral_metrics(motion.respiratory_score(timestamps)[:, None], timestamps, [(0.22, 0.28)])["dominant_frequency_hz"]

    assert abs(peak_hz - 0.25) < 0.03


def test_cardiac_field_is_negligible_outside_cardiac_roi() -> None:
    """Would fail if the cardiac taper leaked into the thoracic background."""
    motion = GroundTruthMotion(cardiac_center_mm=(0.0, 0.0, 0.0), cardiac_radius_mm=(10.0, 12.0, 14.0))
    far_points = torch.tensor([[[35.0, 0.0, 0.0], [0.0, -38.0, 0.0]]])

    field = motion.cardiac_dvf_mm(far_points, torch.tensor([0.3]))

    assert float(field.abs().max()) < 1e-5


def test_ground_truth_pullback_matches_production_sequential_convention() -> None:
    """Would fail if diagnostic rendering reverses C4's cardiac-then-respiratory pullback."""
    motion = GroundTruthMotion()
    points = torch.tensor([[[2.0, -3.0, 4.0], [-5.0, 7.0, -2.0]]])
    timestamp = torch.tensor([0.4])
    production = SequentialPullbackMotion(
        lambda x: motion.respiratory_dvf_mm(x, timestamp),
        lambda x: motion.cardiac_dvf_mm(x, timestamp),
    )

    result = production(points)

    torch.testing.assert_close(result["reference_points_mm"], motion.pullback_reference_mm(points, timestamp))
