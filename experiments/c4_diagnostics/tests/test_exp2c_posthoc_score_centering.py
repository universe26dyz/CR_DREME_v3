from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from experiments.c4_diagnostics.exp2c_posthoc_score_centering import (
    centered_scores,
    centered_spectrum,
    temporal_rms,
    render_resp_scores_from_cached_points,
    _image_values,
)
from experiments.c4_diagnostics.synthetic.metrics import location_spectral_metrics
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField


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


def test_per_location_centering_has_zero_temporal_score_mean() -> None:
    scores = torch.tensor([[[1.0, 2.0, 3.0]], [[3.0, 6.0, 9.0]]])
    centered = centered_scores(scores)
    torch.testing.assert_close(centered.mean(dim=0), torch.zeros_like(centered[0]))


def test_centering_preserves_temporal_rms() -> None:
    scores = torch.tensor([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]], [[5.0, 5.0, 5.0]]])
    assert temporal_rms(centered_scores(scores)) == pytest.approx(temporal_rms(scores))


def test_centering_preserves_centered_frequency_spectrum() -> None:
    scores = torch.tensor([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]], [[0.0, 0.0, 0.0]], [[4.0, 4.0, 4.0]]])
    timestamps = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64)
    torch.testing.assert_close(centered_spectrum(scores, timestamps), centered_spectrum(centered_scores(scores), timestamps))


def test_zero_centered_score_removes_static_score_driven_field() -> None:
    points = torch.zeros(1, 4, 3)
    original = ScoreWeightedMBCField(_ConstantMBC(), torch.ones(1, 1, 3))(points)
    centered = ScoreWeightedMBCField(_ConstantMBC(), torch.zeros(1, 1, 3))(points)
    assert float(original.abs().max()) > 0.0
    torch.testing.assert_close(centered, torch.zeros_like(centered))


def test_original_and_centered_renders_reuse_the_same_cached_query_points() -> None:
    mbc = _RecordingMBC()
    model = SimpleNamespace(respiratory_mbc=mbc)
    query = torch.randn(4, 2, 3)
    scores = torch.ones(3, 3)
    render_resp_scores_from_cached_points(_Canonical(), query, model, scores)
    render_resp_scores_from_cached_points(_Canonical(), query, model, torch.zeros_like(scores))
    assert len(mbc.calls) == 2
    torch.testing.assert_close(mbc.calls[0], query.reshape(1, -1, 3))
    torch.testing.assert_close(mbc.calls[1], query.reshape(1, -1, 3))


def test_temporal_dynamic_image_rms_excludes_static_spatial_contrast() -> None:
    images = torch.tensor([[[0.0, 100.0]], [[0.0, 100.0]]])
    target = images.clone()
    values = _image_values(images, target, images)
    assert values["temporal_dynamic_rms"] == pytest.approx(0.0)
    assert values["temporal_image_std"] > 0.0
