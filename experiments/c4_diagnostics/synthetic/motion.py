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
    domain_center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_half_extent_mm: tuple[float, float, float] = (64.0, 64.0, 64.0)

    @classmethod
    def from_domain(cls, domain: dict[str, object], **kwargs: float) -> "GroundTruthMotion":
        lower, upper = torch.tensor(domain["world_min_mm"], dtype=torch.float32), torch.tensor(domain["world_max_mm"], dtype=torch.float32)
        cardiac = domain["cardiac_box"]
        card_lower, card_upper = torch.tensor(cardiac["min_mm"], dtype=torch.float32), torch.tensor(cardiac["max_mm"], dtype=torch.float32)
        return cls(cardiac_center_mm=tuple(((card_lower + card_upper) / 2.0).tolist()), cardiac_radius_mm=tuple(((card_upper - card_lower) * .48).tolist()), domain_center_mm=tuple(((lower + upper) / 2.0).tolist()), domain_half_extent_mm=tuple(((upper - lower) / 2.0).tolist()), **kwargs)

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
        relative = (points_mm - points_mm.new_tensor(self.domain_center_mm)) / points_mm.new_tensor(self.domain_half_extent_mm)
        x, y, z = relative.unbind(-1)
        support = .70 + .30 * torch.exp(-((x / .80).square() + (y / .65).square() + (z / .85).square()))
        direction = torch.stack((.12 + .03 * y, -.16 + .02 * x, torch.ones_like(z)), dim=-1)
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

    def cardiac_support(self, points_mm: torch.Tensor) -> torch.Tensor:
        unit = (points_mm - points_mm.new_tensor(self.cardiac_center_mm)) / points_mm.new_tensor(self.cardiac_radius_mm)
        return (1.0 - unit.square().sum(-1)).clamp_min(0.0).square()

    def pullback_reference_mm(self, observation_points_mm: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        cardiac_points = observation_points_mm + self.cardiac_dvf_mm(observation_points_mm, timestamps_s)
        return cardiac_points + self.respiratory_dvf_mm(cardiac_points, timestamps_s)
