from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.training.sampler import DynamicObservation

from .motion import GroundTruthMotion
from .phantom import AnalyticCanonicalPhantom


class PhantomCanonicalAdapter(nn.Module):
    """Thin C4 canonical interface around the fixed analytic ground truth."""

    def __init__(self, phantom: AnalyticCanonicalPhantom | None = None) -> None:
        super().__init__()
        self.phantom = AnalyticCanonicalPhantom() if phantom is None else phantom

    def forward(self, points_world_mm: torch.Tensor, return_features: bool = False):
        # NeSVoR's PSF adapter averages the final axis as its sample axis, so
        # expose scalar density as ``[..., samples]`` rather than ``[..., 1]``.
        intensity = self.phantom(points_world_mm).squeeze(-1)
        if not return_features:
            return intensity
        return intensity, torch.zeros((*intensity.shape, 1), dtype=intensity.dtype, device=intensity.device)

    def image_regularization(self, density: torch.Tensor, sample_points_world_mm: torch.Tensor, **_: object) -> torch.Tensor:
        return density.sum() * 0.0


@dataclass(frozen=True)
class GeometryFrame:
    view: str
    slice_id: str
    center_mm: torch.Tensor
    row_direction: torch.Tensor
    column_direction: torch.Tensor
    normal: torch.Tensor
    pixel_spacing_mm: torch.Tensor
    slice_thickness_mm: float
    timestamp_s: float


def pixel_centers(frame: GeometryFrame, height: int, width: int) -> torch.Tensor:
    rows = torch.arange(height, dtype=frame.center_mm.dtype, device=frame.center_mm.device) - (height - 1) / 2.0
    columns = torch.arange(width, dtype=frame.center_mm.dtype, device=frame.center_mm.device) - (width - 1) / 2.0
    rr, cc = torch.meshgrid(rows, columns, indexing="ij")
    return (frame.center_mm + cc.reshape(-1, 1) * frame.pixel_spacing_mm[1] * frame.row_direction + rr.reshape(-1, 1) * frame.pixel_spacing_mm[0] * frame.column_direction)


def render_synthetic_observation(frame: GeometryFrame, *, height: int, width: int, phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, psf_samples: int, seed: int, dynamic_frame_id: int) -> DynamicObservation:
    points = pixel_centers(frame, height, width)
    adapter = NeSVoRPSFAdapter(psf_samples)
    canonical = PhantomCanonicalAdapter(phantom)
    timestamp = points.new_tensor([frame.timestamp_s])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        rendered = adapter(canonical, points, torch.tensor([frame.pixel_spacing_mm[1], frame.pixel_spacing_mm[0], frame.slice_thickness_mm], dtype=points.dtype).expand(points.shape[0], 3), row_direction=frame.row_direction.expand(points.shape[0], -1), column_direction=frame.column_direction.expand(points.shape[0], -1), normal=frame.normal.expand(points.shape[0], -1), motion=lambda values: {"reference_points_mm": motion.pullback_reference_mm(values, timestamp)})
    intensity = rendered["predicted_intensity"]
    # The fixed analytic callable preserves NeSVoR's explicit sample axis;
    # average that axis here exactly as a PSF-integrated acquisition does.
    if intensity.ndim != 1:
        intensity = intensity.mean(-1)
    image = intensity.reshape(1, height, width).detach()
    return DynamicObservation(image=image, view=frame.view, slice_id=frame.slice_id, dynamic_frame_id=dynamic_frame_id, center_mm=frame.center_mm, row_direction=frame.row_direction, column_direction=frame.column_direction, normal=frame.normal, pixel_spacing_mm=frame.pixel_spacing_mm, slice_thickness_mm=frame.slice_thickness_mm, qc_valid=True, qc_reason="synthetic_gt", timestamp_s=frame.timestamp_s)
