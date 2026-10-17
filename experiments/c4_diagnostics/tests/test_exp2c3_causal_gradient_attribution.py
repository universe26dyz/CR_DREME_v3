from __future__ import annotations

import pytest
import torch
import yaml

from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.sampler import DynamicObservation
from experiments.c4_diagnostics.exp2c3_causal_gradient_attribution import (
    _assert_stage3b_data_connectivity,
    cardiac_data_support,
    _gradient_vectors,
    _loss_terms,
    directional_effect,
    gradient_cosine,
    gradient_vector,
    parameter_groups,
    score_state_metrics,
)
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior
from experiments.c4_diagnostics.synthetic.render import PhantomCanonicalAdapter
from experiments.c4_diagnostics.exp2_checkpoint_common import assert_parameters_unchanged, parameter_snapshot


def _domain() -> dict[str, object]:
    return {
        "world_min_mm": [-4.0, -4.0, -4.0],
        "world_max_mm": [4.0, 4.0, 4.0],
        "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]},
    }


@pytest.mark.parametrize("config_name", ["source_first_change4_paperaligned.yaml", "source_first_change4_s2vstyle.yaml"])
def test_parameter_groups_are_nonempty_and_leaf_groups_do_not_duplicate_parameters(config_name: str) -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "configs" / config_name).read_text())
    groups = parameter_groups(build_source_first_model(config, _domain(), n_dynamic_frames=3, device=torch.device("cpu")))
    required = {"encoder_backbone", "encoder_all", "respiratory_mbc.level0", "respiratory_mbc.level1", "respiratory_mbc.level2", "respiratory_mbc_all", "cardiac_mbc", "motion_all"}
    assert required <= set(groups)
    assert all(values for values in groups.values())
    leaf_parameters = [parameter for name, values in groups.items() if name not in {"encoder_backbone", "encoder_all", "respiratory_mbc_all", "motion_all"} for parameter in values]
    assert len({id(parameter) for parameter in leaf_parameters}) == len(leaf_parameters)


def test_gradient_vector_is_deterministic_and_leaves_parameter_grad_unset() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    first = gradient_vector((parameter.square()).sum(), [parameter])
    second = gradient_vector((parameter.square()).sum(), [parameter])
    torch.testing.assert_close(first, torch.tensor([4.0, -6.0]))
    torch.testing.assert_close(first, second)
    assert parameter.grad is None


def test_gradient_cosine_and_directional_effect_follow_loss_descent_sign() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    metric = parameter
    loss = parameter.square()
    metric_gradient = gradient_vector(metric, [parameter])
    loss_gradient = gradient_vector(loss, [parameter])
    assert gradient_cosine(metric_gradient, loss_gradient) == pytest.approx(1.0)
    effect = directional_effect(metric_gradient, loss_gradient, current_metric=float(metric.detach()))
    assert effect["directional_effect_per_unit_loss_gradient"] == pytest.approx(-1.0)
    epsilon = 1e-4
    finite_difference = float(((parameter.detach() - epsilon) - parameter.detach()) / epsilon)
    assert finite_difference < 0 and effect["directional_effect_per_unit_loss_gradient"] < 0


def test_scale_consistent_score_state_ratio_uses_rms_not_channel_l2() -> None:
    scores = torch.tensor([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]]])
    values = score_state_metrics(scores)
    assert values["static_rms"] == pytest.approx(2.0)
    assert values["temporal_rms"] == pytest.approx(1.0)
    assert values["static_to_temporal_rms_ratio"] == pytest.approx(2.0)


def test_production_data_gradient_reaches_encoder_and_respiratory_mbc_while_score_terms_do_not_reach_mbc() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text())
    model = build_source_first_model(config, _domain(), n_dynamic_frames=3, device=torch.device("cpu"))
    model.canonical = PhantomCanonicalAdapter(AnalyticCanonicalPhantom(_domain()))
    sequence = [DynamicObservation(
        image=torch.rand(1, 8, 8), view="SAX", slice_id="one", dynamic_frame_id=index,
        center_mm=torch.tensor([0.0, 0.0, 0.0]), row_direction=torch.tensor([1.0, 0.0, 0.0]),
        column_direction=torch.tensor([0.0, 1.0, 0.0]), normal=torch.tensor([0.0, 0.0, 1.0]),
        pixel_spacing_mm=torch.ones(2), slice_thickness_mm=2.0, qc_valid=True, qc_reason="valid", timestamp_s=.1 * index,
    ) for index in range(3)]
    prior = build_synthetic_frequency_prior(.25, 1.1, {("SAX", "one")})
    weights = {"image": 2e-6, "mbc_normalization": 1e-2, "smooth_resp": 0., "smooth_card": 0., "zero_mean_score": 1e-4, "cardiac_leakage_in_resp": .1, "respiratory_leakage_in_card": .05, "cardiac_target_concentration": 0., "cardiac_pca_waveform": 0.}
    before = parameter_snapshot(model)
    terms, _ = _loss_terms(model, sequence, objective_view="stage3b_joint", data_frame_count=2, pixels_per_frame=8, prior=prior, weights=weights, data_psf_seed=7)
    vectors = _gradient_vectors(terms["weighted_data_mse"], parameter_groups(model))
    assert float(vectors["encoder_all"].norm()) > 0.
    assert float(vectors["respiratory_mbc_all"].norm()) > 0.
    mbc_vectors = _gradient_vectors(terms["weighted_mbc_normalization"], parameter_groups(model))
    zero_vectors = _gradient_vectors(terms["weighted_zero_mean_score"], parameter_groups(model))
    assert torch.equal(mbc_vectors["encoder_all"], torch.zeros_like(mbc_vectors["encoder_all"]))
    assert torch.equal(zero_vectors["respiratory_mbc_all"], torch.zeros_like(zero_vectors["respiratory_mbc_all"]))
    assert_parameters_unchanged(model, before)


def test_cardiac_support_uses_the_same_deterministic_data_pixels_and_requires_cardiac_connectivity() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text())
    model = build_source_first_model(config, _domain(), n_dynamic_frames=1, device=torch.device("cpu"))
    observation = DynamicObservation(
        image=torch.ones(1, 8, 8), view="SAX", slice_id="supported", dynamic_frame_id=0,
        center_mm=torch.zeros(3), row_direction=torch.tensor([1.0, 0.0, 0.0]),
        column_direction=torch.tensor([0.0, 1.0, 0.0]), normal=torch.tensor([0.0, 0.0, 1.0]),
        pixel_spacing_mm=torch.ones(2), slice_thickness_mm=2.0, qc_valid=True, qc_reason="valid", timestamp_s=0.0,
    )
    support = cardiac_data_support(model, [observation], data_frame_count=1, pixels_per_frame=8)
    assert support == {"intersects": True, "inside_points": 4, "total_points": 8}
    vectors = {"encoder_all": torch.ones(1), "respiratory_mbc_all": torch.ones(1), "cardiac_mbc": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="cardiac MBC"):
        _assert_stage3b_data_connectivity("SAX/supported", vectors, support)


def test_outside_cardiac_box_allows_zero_cardiac_gradient_without_weakening_other_connectivity_checks() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text())
    model = build_source_first_model(config, _domain(), n_dynamic_frames=1, device=torch.device("cpu"))
    observation = DynamicObservation(
        image=torch.ones(1, 8, 8), view="SAX", slice_id="outside", dynamic_frame_id=0,
        center_mm=torch.tensor([0.0, 0.0, 5.0]), row_direction=torch.tensor([1.0, 0.0, 0.0]),
        column_direction=torch.tensor([0.0, 1.0, 0.0]), normal=torch.tensor([0.0, 0.0, 1.0]),
        pixel_spacing_mm=torch.ones(2), slice_thickness_mm=2.0, qc_valid=True, qc_reason="valid", timestamp_s=0.0,
    )
    support = cardiac_data_support(model, [observation], data_frame_count=1, pixels_per_frame=8)
    assert support == {"intersects": False, "inside_points": 0, "total_points": 8}
    _assert_stage3b_data_connectivity("SAX/outside", {"encoder_all": torch.ones(1), "respiratory_mbc_all": torch.ones(1), "cardiac_mbc": torch.zeros(1)}, support)
    with pytest.raises(RuntimeError, match="encoder"):
        _assert_stage3b_data_connectivity("SAX/outside", {"encoder_all": torch.zeros(1), "respiratory_mbc_all": torch.ones(1), "cardiac_mbc": torch.zeros(1)}, support)
