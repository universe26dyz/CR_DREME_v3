import torch

from cardioresp4d.losses.frequency_loss import dreme_cardiac_leakage_in_resp, dreme_respiratory_leakage_in_card
from cardioresp4d.losses.motion_loss import dreme_zero_mean_scores
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import build_experiment_schedule, preflight_observations, training_observations_per_update
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics, location_spectral_metrics, temporal_dynamics_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, rescale_frame_for_rendering
from cardioresp4d.training.sampler import DynamicObservation


def test_synthetic_prior_drives_nonzero_differentiable_production_temporal_losses() -> None:
    """Would fail if Exp1 again constructs the production trainer without GT frequency evidence."""
    timestamps = torch.tensor([0.0, 0.17, 0.36, 0.58, 0.83, 1.11, 1.44, 1.76, 2.08, 2.43], dtype=torch.float64)
    prior = build_synthetic_frequency_prior(0.25, 1.10, (("SAX", "loc0"),))
    time = timestamps.to(torch.float32)
    resp = (0.3 + torch.sin(2 * torch.pi * 1.10 * time))[:, None, None].expand(-1, 3, 3).clone().requires_grad_()
    card = (0.2 + torch.sin(2 * torch.pi * 0.25 * time))[:, None, None].expand(-1, 1, 3).clone().requires_grad_()
    local = prior.for_location("SAX", "loc0")
    zero_mean = dreme_zero_mean_scores(torch.cat((resp, card), dim=1))
    eq8 = dreme_cardiac_leakage_in_resp(resp, timestamps, local.cardiac_baseline_pairs)
    eq9 = dreme_respiratory_leakage_in_card(card, timestamps, local.respiratory_bands_hz)

    total = zero_mean + eq8 + eq9
    total.backward()

    assert float(zero_mean) > 0.0 and float(eq8) > 0.0 and float(eq9) > 0.0
    assert resp.grad is not None and card.grad is not None


def test_motion_only_schedule_keeps_canonical_frozen_and_smoke_is_explicitly_small() -> None:
    """Would fail if Exp1 directly trains stage3b or silently unfreezes the GT canonical."""
    formal = build_experiment_schedule("motion_only", smoke=False)
    smoke = build_experiment_schedule("motion_only", smoke=True)

    assert [segment.name for segment in formal] == ["s2a_init", "s2a_joint", "s2b_init", "s2b_joint", "s2c_init", "s2c_joint", "s3a", "s3b_full"]
    assert all(segment.train_canonical is False for segment in formal)
    assert all(segment.steps == 1 for segment in smoke)
    assert [segment.name for segment in build_experiment_schedule("full_joint", smoke=False)] == ["s1a", "s1b", "s2a_init", "s2a_joint", "s2b_init", "s2b_joint", "s2c_init", "s2c_joint", "s3a", "s3b_full"]
    assert training_observations_per_update({"training": {"observations_per_update": 32}}, smoke=False) == 32
    assert training_observations_per_update({"training": {"observations_per_update": 32}}, smoke=True) == 3


def test_reduced_rendering_preserves_manifest_physical_fov() -> None:
    """Would fail if a 32x32 synthetic render retained original high-resolution spacing."""
    frame = GeometryFrame("SAX", "loc0", torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.2, 1.5]), 6.0, 0.0, 121, 181)

    reduced = rescale_frame_for_rendering(frame, height=31, width=46)

    assert torch.allclose(reduced.pixel_spacing_mm, torch.tensor([1.2 * 120 / 30, 1.5 * 180 / 45]))


def test_location_metrics_keep_sequences_separate_and_use_per_channel_nudft() -> None:
    """Would fail if temporal metrics or spectra concatenate different fixed locations or average score channels first."""
    timestamps = torch.tensor([0.0, 0.17, 0.36, 0.58, 0.83, 1.11, 1.44, 1.76], dtype=torch.float64)
    signal = torch.sin(2 * torch.pi * .25 * timestamps.float())
    metrics_a = temporal_dynamics_metrics(signal[:, None], signal[:, None])
    metrics_b = temporal_dynamics_metrics((10 + signal)[:, None], (10 + signal)[:, None])
    aggregate = aggregate_location_metrics({"SAX/a": metrics_a, "SAX/b": metrics_b})
    scores = torch.stack((signal, .4 * signal, torch.zeros_like(signal)), dim=-1)
    spectrum = location_spectral_metrics(scores, timestamps, [(0.18, 0.32)])

    assert abs(metrics_a["temporal_std_gt"] - metrics_b["temporal_std_gt"]) < 1e-6
    assert abs(aggregate["temporal_std_gt"]["mean"] - metrics_a["temporal_std_gt"]) < 1e-6
    assert spectrum["channel_count"] == 3 and spectrum["target_band_power"] > 0.0


def test_preflight_rejects_cardiac_views_without_gt_support() -> None:
    """Would fail if formal runs start even though long-axis synthetic slices miss the cardiac support."""
    motion = GroundTruthMotion.from_domain({"world_min_mm": [-20., -20., -20.], "world_max_mm": [20., 20., 20.], "cardiac_box": {"min_mm": [-5., -5., -5.], "max_mm": [5., 5., 5.]}})
    observation = DynamicObservation(torch.linspace(0, 1, 16).reshape(1, 4, 4), "2CH", "miss", 0, torch.tensor([30., 0., 0.]), torch.tensor([1., 0., 0.]), torch.tensor([0., 0., 1.]), torch.tensor([0., -1., 0.]), torch.tensor([1., 1.]), 4.0, True, "synthetic_gt", 0.0)

    report = preflight_observations([observation], motion)

    assert report["cardiac_view_intersection_fraction"] == 0.0
    assert report["valid"] is False


def test_phantom_and_cardiac_motion_follow_shifted_canonical_domain() -> None:
    """Would fail if the synthetic anatomy or cardiac field reintroduced a hard-coded world origin."""
    domain = {"world_min_mm": [100., 200., 300.], "world_max_mm": [180., 280., 390.], "cardiac_box": {"min_mm": [125., 225., 325.], "max_mm": [155., 255., 365.]}}
    phantom = AnalyticCanonicalPhantom(domain)
    motion = GroundTruthMotion.from_domain(domain)
    center = torch.tensor(motion.cardiac_center_mm).reshape(1, 1, 3)

    intensity = phantom(center)
    field = motion.cardiac_dvf_mm(center + torch.tensor([[[1., 0., 0.]]]), torch.tensor([.25]))

    assert float(intensity) > .1
    assert float(field.abs().max()) > 0.0
