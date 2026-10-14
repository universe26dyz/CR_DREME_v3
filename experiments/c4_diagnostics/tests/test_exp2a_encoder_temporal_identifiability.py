from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cardioresp4d.models.film_motion_encoder import GeometryFiLMMotionEncoder
from experiments.c4_diagnostics.exp2a_encoder_temporal_identifiability import (
    counterfactual_encoder_inputs,
    extract_encoder_stages,
    feature_temporal_metrics,
    static_temporal_metrics,
    validate_constant_location_geometry,
)


def _geometry(center: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return {
        "center_mm": torch.zeros(1, 3) if center is None else center.reshape(1, 3),
        "row_direction": torch.tensor([[1.0, 0.0, 0.0]]),
        "column_direction": torch.tensor([[0.0, 1.0, 0.0]]),
        "normal": torch.tensor([[0.0, 0.0, 1.0]]),
        "pixel_spacing_mm": torch.tensor([[1.5, 1.5]]),
        "slice_thickness_mm": torch.tensor([[4.0]]),
    }


def _observation(image: torch.Tensor, center: torch.Tensor | None = None) -> SimpleNamespace:
    geometry = _geometry(center)
    return SimpleNamespace(image=image, slice_id="fixed", view="SAX", timestamp_s=0.0, **{name: value[0] for name, value in geometry.items()})


def test_extracted_encoder_path_reproduces_production_resp_scores() -> None:
    torch.manual_seed(3)
    encoder = GeometryFiLMMotionEncoder(channels=4, canonical_lower_world_mm=torch.full((3,), -12.0), canonical_upper_world_mm=torch.full((3,), 12.0))
    image, geometry = torch.randn(1, 1, 8, 8), _geometry()
    expected = encoder(image, **geometry)["resp_scores"]
    actual = extract_encoder_stages(encoder, image, **geometry)
    torch.testing.assert_close(actual["resp_scores"], expected)


def test_constant_geometry_within_location_is_validated() -> None:
    first, second = _observation(torch.zeros(1, 4, 4)), _observation(torch.ones(1, 4, 4))
    validate_constant_location_geometry([first, second])
    changed = _observation(torch.ones(1, 4, 4), torch.tensor([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="geometry changes"):
        validate_constant_location_geometry([first, changed])


def test_feature_proportional_to_gt_has_unit_correlation() -> None:
    gt = torch.tensor([-1.0, 0.0, 1.0])
    feature = torch.stack((2.0 * gt, -3.0 * gt), dim=-1)
    values = feature_temporal_metrics(feature, gt)
    assert values["max_abs_gt_correlation"] == pytest.approx(1.0)
    assert values["pc1_abs_gt_correlation"] == pytest.approx(1.0)


def test_constant_temporal_feature_is_safe() -> None:
    values = feature_temporal_metrics(torch.ones(3, 2), torch.tensor([-1.0, 0.0, 1.0]))
    assert values["temporal_rms"] == pytest.approx(0.0)
    assert torch.isnan(torch.tensor(values["max_abs_gt_correlation"]))


def test_static_to_temporal_ratio_formula() -> None:
    values = static_temporal_metrics(torch.tensor([[-2.0, 0.0], [2.0, 0.0]]), torch.tensor([1.0, 3.0]))
    assert values["between_location_mean_rms"] == pytest.approx(2.0**0.5)
    assert values["median_within_location_temporal_rms"] == pytest.approx(2.0)
    assert values["static_to_temporal_ratio"] == pytest.approx(2.0**-0.5)


def test_temporal_mean_image_counterfactual_retains_original_geometry() -> None:
    original = _observation(torch.ones(1, 4, 4))
    reference = _observation(3.0 * torch.ones(1, 4, 4), torch.tensor([2.0, 0.0, 0.0]))
    inputs = counterfactual_encoder_inputs(original, 2.0 * torch.ones(1, 4, 4), reference)
    for name in _geometry():
        torch.testing.assert_close(inputs["temporal_mean_image"][name], inputs["original"][name])
    torch.testing.assert_close(inputs["temporal_mean_image"]["image"], 2.0 * torch.ones(1, 1, 4, 4))


def test_reference_geometry_counterfactual_keeps_original_image() -> None:
    original = _observation(torch.ones(1, 4, 4))
    reference = _observation(3.0 * torch.ones(1, 4, 4), torch.tensor([2.0, 0.0, 0.0]))
    inputs = counterfactual_encoder_inputs(original, 2.0 * torch.ones(1, 4, 4), reference)
    torch.testing.assert_close(inputs["reference_geometry"]["image"], inputs["original"]["image"])
    torch.testing.assert_close(inputs["reference_geometry"]["center_mm"], reference.center_mm[None])
    assert not torch.equal(inputs["reference_geometry"]["center_mm"], inputs["original"]["center_mm"])
