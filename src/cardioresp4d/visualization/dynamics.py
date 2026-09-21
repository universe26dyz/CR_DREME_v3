"""Physical-grid pullback, direct INR query, and Jacobian helpers."""
from __future__ import annotations

import torch


def world_grid(lower_mm: torch.Tensor, upper_mm: torch.Tensor, shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(shape) != 3 or min(shape) < 2: raise ValueError("world grid shape must contain three values >=2")
    lower, upper = torch.as_tensor(lower_mm), torch.as_tensor(upper_mm)
    axes = [torch.linspace(lo, hi, count, device=lower.device, dtype=lower.dtype) for lo, hi, count in zip(lower, upper, shape)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1), (upper - lower) / torch.tensor([count - 1 for count in shape], device=lower.device, dtype=lower.dtype)


def pullback_displacement(observation_points_mm: torch.Tensor, reference_points_mm: torch.Tensor) -> torch.Tensor:
    if observation_points_mm.shape != reference_points_mm.shape or observation_points_mm.shape[-1] != 3: raise ValueError("matched [...,3] observation/reference world-mm points required")
    return reference_points_mm - observation_points_mm


def jacobian_determinant(reference_points_mm: torch.Tensor, spacing_mm: torch.Tensor) -> torch.Tensor:
    """Finite-difference det(d phi / d y) for observation-to-reference pullback."""
    if reference_points_mm.ndim != 4 or reference_points_mm.shape[-1] != 3: raise ValueError("reference points must be [D,H,W,3]")
    spacing = torch.as_tensor(spacing_mm, dtype=reference_points_mm.dtype, device=reference_points_mm.device)
    if spacing.shape != (3,) or torch.any(spacing <= 0): raise ValueError("physical spacing_mm must be positive [3]")
    derivatives = torch.gradient(reference_points_mm, spacing=tuple(float(value) for value in spacing), dim=(0, 1, 2), edge_order=1)
    matrix = torch.stack(derivatives, dim=-1)
    return torch.linalg.det(matrix)


def chunked_canonical_query(canonical, motion, observation_grid_mm: torch.Tensor, *, chunk_size: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    """Query canonical INR directly after pullback; no PSF is applied to voxels."""
    if chunk_size <= 0 or observation_grid_mm.shape[-1] != 3: raise ValueError("positive chunk_size and [...,3] world grid required")
    shape, flat = observation_grid_mm.shape[:-1], observation_grid_mm.reshape(-1, 3)
    values, reference = [], []
    for start in range(0, flat.shape[0], chunk_size):
        points = flat[start:start + chunk_size][None]
        pulled = motion(points)["reference_points_mm"] if motion is not None else points
        intensity = canonical(pulled)
        values.append(intensity.reshape(-1)); reference.append(pulled.reshape(-1, 3))
    return torch.cat(values).reshape(*shape), torch.cat(reference).reshape(*shape, 3)
