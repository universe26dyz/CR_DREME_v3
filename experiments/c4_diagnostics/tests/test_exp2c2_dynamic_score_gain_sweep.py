from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from experiments.c4_diagnostics.exp2c2_dynamic_score_gain_sweep import (
    _oracle_metrics,
    alpha_dvf_least_squares,
    dynamic_cosine,
    gain_grid_best,
    gain_scores,
    render_gain_images_from_cached_points,
)
from experiments.c4_diagnostics.exp2_checkpoint_common import assert_parameters_unchanged, parameter_snapshot
from experiments.c4_diagnostics.exp2a_encoder_temporal_identifiability import validate_source_timestamps


class _ConstantMBC(torch.nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return torch.ones((points.shape[0], 1, points.shape[1], 3), dtype=points.dtype, device=points.device)


class _RecordingMBC(_ConstantMBC):
    def __init__(self) -> None:
        super().__init__(); self.calls: list[torch.Tensor] = []

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        self.calls.append(points.detach().clone())
        return super().forward(points)


class _Canonical(torch.nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points[..., 0]


def test_centered_scores_have_zero_temporal_mean() -> None:
    scores = torch.tensor([[[1.0, 2.0, 3.0]], [[3.0, 6.0, 9.0]]])
    torch.testing.assert_close(gain_scores(scores, 1.0).mean(dim=0), torch.zeros_like(scores[0]))


def test_zero_gain_produces_zero_score_and_toy_respiratory_dvf() -> None:
    scores = torch.tensor([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]]])
    zero = gain_scores(scores, 0.0)
    points = torch.zeros(1, 4, 3)
    field = ScoreWeightedMBCField(_ConstantMBC(), zero[0:1])(points)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    torch.testing.assert_close(field, torch.zeros_like(field))


def test_gain_scales_linear_score_weighted_mbc_field() -> None:
    score = torch.tensor([[[1.0, 2.0, 3.0]]])
    points = torch.zeros(1, 4, 3)
    base = ScoreWeightedMBCField(_ConstantMBC(), score)(points)
    scaled = ScoreWeightedMBCField(_ConstantMBC(), 7.0 * score)(points)
    torch.testing.assert_close(scaled, 7.0 * base)


def test_dvf_least_squares_gain_recovers_known_scale() -> None:
    base = torch.tensor([[[1.0, -2.0, 3.0]], [[-1.0, 4.0, 2.0]]])
    assert alpha_dvf_least_squares(base, 12.5 * base) == pytest.approx(12.5)


def test_dynamic_dvf_oracle_ignores_static_target_offset() -> None:
    source = torch.tensor([[[1.0, -2.0, 3.0]], [[-1.0, 4.0, 2.0]], [[5.0, 1.0, -3.0]]])
    source_dynamic = source - source.mean(dim=0, keepdim=True)
    scale = 12.5
    target = scale * source_dynamic + torch.tensor([[[7.0, -11.0, 13.0]]])

    alpha, values = _oracle_metrics(source, target)
    shifted_alpha, shifted_values = _oracle_metrics(source, target + torch.tensor([[[19.0, 23.0, -29.0]]]))

    assert alpha == pytest.approx(scale)
    assert shifted_alpha == pytest.approx(scale)
    assert values["dynamic_endpoint_error_mm"] == pytest.approx(0.0, abs=1e-5)
    assert values["dynamic_normalized_epe"] == pytest.approx(0.0, abs=1e-6)
    assert values["dynamic_rms_ratio"] == pytest.approx(1.0)
    assert values["dynamic_cosine"] == pytest.approx(1.0)
    assert shifted_values["dynamic_endpoint_error_mm"] == pytest.approx(0.0, abs=1e-5)
    assert shifted_values["dynamic_normalized_epe"] == pytest.approx(0.0, abs=1e-6)


def test_dynamic_cosine_is_invariant_for_positive_gain() -> None:
    base = torch.tensor([[[1.0, -2.0, 3.0]], [[-1.0, 4.0, 2.0]]])
    target = torch.tensor([[[3.0, 1.0, -2.0]], [[-2.0, 5.0, 1.0]]])
    assert dynamic_cosine(base, target) == pytest.approx(dynamic_cosine(30.0 * base, target))


def test_matched_query_points_are_reused_for_every_gain() -> None:
    mbc = _RecordingMBC()
    model = SimpleNamespace(respiratory_mbc=mbc)
    query = torch.randn(4, 2, 3)
    scores = {0.0: torch.zeros(1, 3), 1.0: torch.ones(1, 3), 10.0: 10.0 * torch.ones(1, 3)}
    images = render_gain_images_from_cached_points(_Canonical(), model, query, scores, (2, 2))
    assert set(images) == set(scores)
    assert all(image.shape == (2, 2) for image in images.values())
    assert len(mbc.calls) == len(scores)
    for call in mbc.calls:
        torch.testing.assert_close(call, query.reshape(1, -1, 3))


def test_gain_grid_optimum_and_boundary_flag() -> None:
    best, mse, boundary = gain_grid_best({0.0: 4.0, 1.0: 1.0, 10.0: 3.0}, [0.0, 1.0, 10.0])
    assert best == pytest.approx(1.0) and mse == pytest.approx(1.0) and not boundary
    best, _, boundary = gain_grid_best({0.0: .5, 1.0: 1.0}, [0.0, 1.0])
    assert best == pytest.approx(0.0) and boundary


def test_source_timestamps_and_parameter_snapshot_guarantees(tmp_path) -> None:
    import numpy as np

    arrays = tmp_path / "recovery_arrays.npz"
    np.savez(arrays, SAX_fixed_timestamps_s=np.array([0.0, 1.0]))
    assert validate_source_timestamps(arrays, {"SAX/fixed": [SimpleNamespace(timestamp_s=0.0), SimpleNamespace(timestamp_s=1.0)]})
    module = torch.nn.Linear(2, 1)
    assert_parameters_unchanged(module, parameter_snapshot(module))
