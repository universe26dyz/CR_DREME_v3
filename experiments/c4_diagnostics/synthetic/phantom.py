from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AnalyticCanonicalPhantom(nn.Module):
    """Continuous asymmetric phantom expressed relative to the supplied domain."""

    def __init__(self, domain: dict[str, object] | None = None) -> None:
        super().__init__()
        domain = domain or {"world_min_mm": [-64., -64., -64.], "world_max_mm": [64., 64., 64.], "cardiac_box": {"min_mm": [-24., -24., -28.], "max_mm": [32., 24., 30.]}}
        lower, upper = torch.tensor(domain["world_min_mm"], dtype=torch.float32), torch.tensor(domain["world_max_mm"], dtype=torch.float32)
        cardiac = domain["cardiac_box"]
        self.register_buffer("domain_center_mm", (lower + upper) / 2.0)
        self.register_buffer("domain_half_extent_mm", (upper - lower) / 2.0)
        self.register_buffer("cardiac_center_mm", (torch.tensor(cardiac["min_mm"], dtype=torch.float32) + torch.tensor(cardiac["max_mm"], dtype=torch.float32)) / 2.0)
        self.register_buffer("cardiac_half_extent_mm", (torch.tensor(cardiac["max_mm"], dtype=torch.float32) - torch.tensor(cardiac["min_mm"], dtype=torch.float32)) / 2.0)

    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        if points_mm.shape[-1] != 3:
            raise ValueError("points_mm must end in xyz")
        domain = (points_mm - self.domain_center_mm) / self.domain_half_extent_mm
        cardiac = (points_mm - self.cardiac_center_mm) / self.cardiac_half_extent_mm
        x, y, z = domain.unbind(-1)
        cx, cy, cz = cardiac.unbind(-1)
        thorax = 0.22 * torch.exp(-((x / .75).square() + ((y + .08) / .58).square() + (z / .82).square()))
        heart = 0.46 * torch.exp(-((cx / .72).square() + ((cy + .12) / .70).square() + ((cz - .08) / .75).square()))
        oblique = 0.33 * torch.exp(-(((x + .30 + .35 * z) / .16).square() + ((y - .22 - .2 * x) / .12).square() + ((z + .18) / .22).square()))
        vessel = 0.28 * torch.exp(-(((cx - .30) / .18).square() + ((cy + .20) / .22).square() + ((cz + .12) / .90).square()))
        texture = 0.12 * torch.sin(8.0 * x + 5.0 * y - 6.0 * z) * torch.exp(-((x / .70).square() + (y / .62).square() + (z / .78).square()))
        return (0.08 + thorax + heart + oblique + vessel + texture).clamp(0.0, 1.0).unsqueeze(-1)

    @torch.no_grad()
    def volume(self, lower_mm: torch.Tensor, upper_mm: torch.Tensor, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        device, dtype = self.domain_center_mm.device, self.domain_center_mm.dtype
        lower = torch.as_tensor(lower_mm, device=device, dtype=dtype)
        upper = torch.as_tensor(upper_mm, device=device, dtype=dtype)
        axes = [torch.linspace(lower[index], upper[index], shape[index], device=device, dtype=dtype) for index in range(3)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return tuple(axis.detach().cpu().numpy() for axis in axes) + (self(grid).squeeze(-1).detach().cpu().numpy(),)
