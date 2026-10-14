from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.c4_diagnostics.exp1e_component_image_observability import (
    _location_variants,
    center_support_component_fields,
    component_image_metrics,
    decompose_world_dvf,
    fixed_center_support_world_points,
    reconstruct_world_dvf,
    render_component_variants,
)
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.oracle import build_oracle_motion_model, build_target_psf_offset_cache
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, render_synthetic_observation


class _XGradientPhantom(nn.Module):
    def forward(self, points_world_mm: torch.Tensor, return_features: bool = False):
        intensity = points_world_mm[..., 0]
        if return_features:
            return intensity, torch.zeros((*intensity.shape, 1), dtype=intensity.dtype)
        return intensity


def _basis() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])


def test_dvf_slice_component_decomposition_reconstructs_world_vectors() -> None:
    row = torch.tensor([0.0, 1.0, 0.0])
    column = torch.tensor([0.0, 0.0, 1.0])
    normal = torch.tensor([1.0, 0.0, 0.0])
    world = torch.tensor([[[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]]])
    components = decompose_world_dvf(world, row, column, normal)
    torch.testing.assert_close(reconstruct_world_dvf(components, row, column, normal), world)


def test_removing_zero_component_keeps_rendered_image_identical() -> None:
    row, column, normal = _basis()
    points = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    dvf = torch.tensor([[[0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0]]])
    variants = render_component_variants(_XGradientPhantom(), points, dvf, row, column, normal)
    torch.testing.assert_close(variants["full"], variants["remove_normal"])


def test_full_remove_and_flip_reuse_identical_cached_psf_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    domain = {"world_min_mm": [-12.0, -12.0, -12.0], "world_max_mm": [12.0, 12.0, 12.0], "cardiac_box": {"min_mm": [-7.0, -7.0, -7.0], "max_mm": [7.0, 7.0, 7.0]}}
    motion = GroundTruthMotion.from_domain(domain, respiratory_amplitude_mm=4.0, cardiac_amplitude_mm=0.0)
    frame = GeometryFrame("SAX", "test", torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.5, 1.5]), 4.0, 0.5, 8, 8)
    observation = render_synthetic_observation(frame, height=8, width=8, phantom=AnalyticCanonicalPhantom(domain), motion=motion, psf_samples=1, seed=31, dynamic_frame_id=0)
    cache = build_target_psf_offset_cache(observation, psf_samples=1, seed=31)
    model = build_oracle_motion_model(domain, AnalyticCanonicalPhantom(domain), device=torch.device("cpu"), psf_samples=1)
    calls: list[torch.Tensor] = []
    original = model.psf_sample_world_points

    def tracked(observation, pixel_uv, standard_normal_offsets):
        calls.append(standard_normal_offsets.detach().clone())
        return original(observation, pixel_uv, standard_normal_offsets)

    monkeypatch.setattr(model, "psf_sample_world_points", tracked)
    variants, _ = _location_variants(model, observation, cache, motion, pixel_chunk=64)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], cache.standard_normal_offsets)
    torch.testing.assert_close(variants["full"], observation.image[0], atol=1e-6, rtol=0.0)
    assert variants["full"].shape == variants["remove_row"].shape == variants["flip_row"].shape


def test_gradient_aligned_translation_has_larger_image_effect_than_insensitive_direction() -> None:
    row, column, normal = _basis()
    points = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    x_dvf = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    y_dvf = torch.tensor([[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]])
    x_variants = render_component_variants(_XGradientPhantom(), points, x_dvf, row, column, normal)
    y_variants = render_component_variants(_XGradientPhantom(), points, y_dvf, row, column, normal)
    x_effect = (x_variants["full"] - x_variants["zero"]).square().mean()
    y_effect = (y_variants["full"] - y_variants["zero"]).square().mean()
    assert x_effect > y_effect


def test_normalized_component_sensitivity_formulas_are_correct() -> None:
    full = torch.tensor([[[1.0]], [[3.0]]])
    zero = torch.tensor([[[0.0]], [[0.0]]])
    remove = torch.tensor([[[0.0]], [[2.0]]])
    flip = torch.tensor([[[-1.0]], [[1.0]]])
    component = torch.tensor([[[1.0]], [[3.0]]])
    values = component_image_metrics(full, zero, remove, flip, component)
    assert values["gt_center_component_dynamic_rms_mm"] == pytest.approx(1.0)
    assert values["remove_mse"] == pytest.approx(1.0)
    assert values["flip_mse"] == pytest.approx(4.0)
    assert values["remove_sensitivity_mse_per_mm2"] == pytest.approx(1.0)
    assert values["flip_sensitivity_mse_per_mm2"] == pytest.approx(1.0)


def _fixed_location_observations(motion: GroundTruthMotion) -> list:
    observations = []
    for index, timestamp in enumerate((0.2, 0.8)):
        frame = GeometryFrame("SAX", "fixed", torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.5, 1.5]), 4.0, timestamp, 8, 8)
        observations.append(render_synthetic_observation(frame, height=8, width=8, phantom=AnalyticCanonicalPhantom({"world_min_mm": [-12.0, -12.0, -12.0], "world_max_mm": [12.0, 12.0, 12.0], "cardiac_box": {"min_mm": [-7.0, -7.0, -7.0], "max_mm": [7.0, 7.0, 7.0]}}), motion=motion, psf_samples=1, seed=51 + index, dynamic_frame_id=index))
    return observations


def test_center_support_coordinates_are_identical_across_a_fixed_location() -> None:
    domain = {"world_min_mm": [-12.0, -12.0, -12.0], "world_max_mm": [12.0, 12.0, 12.0], "cardiac_box": {"min_mm": [-7.0, -7.0, -7.0], "max_mm": [7.0, 7.0, 7.0]}}
    points = fixed_center_support_world_points(_fixed_location_observations(GroundTruthMotion.from_domain(domain)))
    torch.testing.assert_close(points, points[0].expand_as(points))


def test_center_component_rms_uses_spatially_corresponding_pixel_centers() -> None:
    domain = {"world_min_mm": [-12.0, -12.0, -12.0], "world_max_mm": [12.0, 12.0, 12.0], "cardiac_box": {"min_mm": [-7.0, -7.0, -7.0], "max_mm": [7.0, 7.0, 7.0]}}
    motion = GroundTruthMotion.from_domain(domain, respiratory_amplitude_mm=4.0, cardiac_amplitude_mm=0.0)
    observations = _fixed_location_observations(motion)
    points, _, components = center_support_component_fields(observations, motion)
    timestamps = torch.tensor([item.timestamp_s for item in observations])
    expected = decompose_world_dvf(motion.respiratory_dvf_mm(points, timestamps), observations[0].row_direction, observations[0].column_direction, observations[0].normal)
    torch.testing.assert_close(components, expected)
    assert float((components - components.mean(0, keepdim=True)).square().mean().sqrt()) > 0.0


def test_per_frame_psf_offsets_do_not_change_center_support_rms_denominator() -> None:
    domain = {"world_min_mm": [-12.0, -12.0, -12.0], "world_max_mm": [12.0, 12.0, 12.0], "cardiac_box": {"min_mm": [-7.0, -7.0, -7.0], "max_mm": [7.0, 7.0, 7.0]}}
    motion = GroundTruthMotion.from_domain(domain, respiratory_amplitude_mm=4.0, cardiac_amplitude_mm=0.0)
    observations = _fixed_location_observations(motion)
    model = build_oracle_motion_model(domain, AnalyticCanonicalPhantom(domain), device=torch.device("cpu"), psf_samples=1)
    cache_a = build_target_psf_offset_cache(observations[0], psf_samples=1, seed=71)
    cache_b = build_target_psf_offset_cache(observations[0], psf_samples=1, seed=72)
    _, psf_dvf_a = _location_variants(model, observations[0], cache_a, motion, pixel_chunk=64)
    _, psf_dvf_b = _location_variants(model, observations[0], cache_b, motion, pixel_chunk=64)
    assert not torch.equal(psf_dvf_a, psf_dvf_b)
    _, center_dvf, center_components = center_support_component_fields(observations, motion)
    full = torch.tensor([[[1.0]], [[3.0]]])
    values = component_image_metrics(full, torch.zeros_like(full), torch.tensor([[[0.0]], [[2.0]]]), torch.tensor([[[-1.0]], [[1.0]]]), center_components[..., 0], center_dvf)
    assert values["gt_center_component_dynamic_rms_mm"] == pytest.approx(float((center_components[..., 0] - center_components[..., 0].mean(0, keepdim=True)).square().mean().sqrt()))
