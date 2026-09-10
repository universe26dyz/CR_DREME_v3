"""Motion-aware renderer using NeSVoR's official Gaussian sampling semantics."""
from __future__ import annotations

import torch
from torch import nn
from ._upstream import add_upstream_to_path

add_upstream_to_path("NeSVoR")
from nesvor.utils import resolution2sigma  # noqa: E402


def resolution_sigma_mm(resolution_mm: torch.Tensor) -> torch.Tensor:
    """Call the pinned upstream conversion; no local FWHM constant exists."""
    if resolution_mm.shape[-1] != 3 or torch.any(resolution_mm <= 0) or not torch.isfinite(resolution_mm).all():
        raise ValueError("resolution_mm must be finite positive [...,3]")
    return resolution2sigma(resolution_mm, isotropic=False)


class NeSVoRPSFAdapter(nn.Module):
    """Use official PSF samples, then insert the project motion pullback."""
    def __init__(self, n_samples: int = 8) -> None:
        super().__init__()
        if n_samples <= 0: raise ValueError("n_samples must be positive")
        self.n_samples = n_samples

    def sigma_mm(self, resolution_mm: torch.Tensor) -> torch.Tensor:
        return resolution_sigma_mm(resolution_mm)

    def sample(self, centers_world_mm: torch.Tensor, resolution_mm: torch.Tensor, *, inr: nn.Module | None = None) -> torch.Tensor:
        if centers_world_mm.ndim != 2 or centers_world_mm.shape[-1] != 3: raise ValueError("centers_world_mm must be [N,3]")
        sigma = self.sigma_mm(resolution_mm).to(centers_world_mm)
        if sigma.shape[0] not in (1, centers_world_mm.shape[0]): raise ValueError("resolution batch must be one or match centers")
        if inr is not None: return inr.sample_batch(centers_world_mm, None, sigma, self.n_samples)
        if self.n_samples == 1: return centers_world_mm[:, None]
        return centers_world_mm[:, None] + torch.randn(centers_world_mm.shape[0], self.n_samples, 3, dtype=centers_world_mm.dtype, device=centers_world_mm.device) * sigma.view(-1, 1, 3)

    def forward(self, canonical: nn.Module, centers_world_mm: torch.Tensor, resolution_mm: torch.Tensor, *, motion: nn.Module | None = None) -> dict[str, torch.Tensor]:
        samples = self.sample(centers_world_mm, resolution_mm, inr=getattr(canonical, "inr", None))
        reference_points = samples if motion is None else motion(samples)["reference_points_mm"]
        intensity, latent_z = canonical(reference_points, return_features=True)
        return {"predicted_intensity": intensity.mean(-1), "intensity_samples": intensity, "latent_samples": latent_z, "observation_samples_world_mm": samples, "reference_samples_world_mm": reference_points}
