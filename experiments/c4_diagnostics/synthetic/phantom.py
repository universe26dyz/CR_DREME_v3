from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AnalyticCanonicalPhantom(nn.Module):
    """Continuous asymmetric world-mm phantom with thoracic and cardiac detail."""

    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        if points_mm.shape[-1] != 3:
            raise ValueError("points_mm must end in xyz")
        x, y, z = points_mm.unbind(-1)
        thorax = 0.22 * torch.exp(-((x / 48.0).square() + ((y + 4.0) / 35.0).square() + (z / 58.0).square()))
        heart = 0.46 * torch.exp(-(((x - 7.0) / 18.0).square() + ((y + 5.0) / 15.0).square() + ((z - 3.0) / 20.0).square()))
        oblique = 0.33 * torch.exp(-(((x + 18.0 + .35 * z) / 9.0).square() + ((y - 12.0 - .2 * x) / 7.0).square() + ((z + 10.0) / 15.0).square()))
        vessel = 0.28 * torch.exp(-(((x - 14.0) / 4.0).square() + ((y - 11.0) / 5.0).square() + ((z + 7.0) / 29.0).square()))
        texture = 0.12 * torch.sin(.23 * x + .11 * y - .17 * z) * torch.exp(-((x / 42.0).square() + (y / 34.0).square() + (z / 50.0).square()))
        return (0.08 + thorax + heart + oblique + vessel + texture).clamp(0.0, 1.0).unsqueeze(-1)

    @torch.no_grad()
    def volume(self, lower_mm: torch.Tensor, upper_mm: torch.Tensor, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        axes = [torch.linspace(float(lower_mm[index]), float(upper_mm[index]), shape[index]) for index in range(3)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return tuple(axis.numpy() for axis in axes) + (self(grid).squeeze(-1).cpu().numpy(),)
