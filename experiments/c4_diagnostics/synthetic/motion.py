from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroundTruthMotion:
    respiratory_frequency_hz: float = .25
    cardiac_frequency_hz: float = 1.10
    respiratory_amplitude_mm: float = 6.0
    cardiac_amplitude_mm: float = 3.5
    cardiac_center_mm: tuple[float, float, float] = (7.0, -5.0, 3.0)
    cardiac_radius_mm: tuple[float, float, float] = (22.0, 19.0, 24.0)

    def respiratory_score(self, timestamps_s: torch.Tensor) -> torch.Tensor:
        return torch.sin(2.0 * torch.pi * self.respiratory_frequency_hz * timestamps_s)

    def cardiac_score(self, timestamps_s: torch.Tensor) -> torch.Tensor:
        return torch.sin(2.0 * torch.pi * self.cardiac_frequency_hz * timestamps_s + .37)

    def _time(self, points_mm: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        time = torch.as_tensor(timestamps_s, dtype=points_mm.dtype, device=points_mm.device).reshape(-1)
        if time.numel() == 1:
            time = time.expand(points_mm.shape[0])
        if time.numel() != points_mm.shape[0]:
            raise ValueError("timestamps must be scalar or match point batch")
        return time[:, None]

    def respiratory_dvf_mm(self, points_mm: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        time = self._time(points_mm, timestamps_s)
        x, y, z = points_mm.unbind(-1)
        support = .70 + .30 * torch.exp(-((x / 50.0).square() + (y / 40.0).square() + (z / 55.0).square()))
        direction = torch.stack((.12 + .03 * y / 40.0, -.16 + .02 * x / 50.0, torch.ones_like(z)), dim=-1)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        return direction * support.unsqueeze(-1) * self.respiratory_amplitude_mm * self.respiratory_score(time).unsqueeze(-1)

    def cardiac_dvf_mm(self, points_mm: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        time = self._time(points_mm, timestamps_s)
        center = points_mm.new_tensor(self.cardiac_center_mm)
        radius = points_mm.new_tensor(self.cardiac_radius_mm)
        unit = (points_mm - center) / radius
        radius2 = unit.square().sum(-1)
        taper = (1.0 - radius2).clamp_min(0.0).square()
        contraction = torch.stack((-unit[..., 0], -.75 * unit[..., 1], -.30 * unit[..., 2]), dim=-1)
        translation = points_mm.new_tensor((.22, -.18, .12))
        return (contraction + translation) * taper.unsqueeze(-1) * self.cardiac_amplitude_mm * self.cardiac_score(time).unsqueeze(-1)

    def pullback_reference_mm(self, observation_points_mm: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        cardiac_points = observation_points_mm + self.cardiac_dvf_mm(observation_points_mm, timestamps_s)
        return cardiac_points + self.respiratory_dvf_mm(cardiac_points, timestamps_s)
