"""Physical-mm continuous-INR exports and acquisition-plane queries for Stage 1.

The reconstruction domain is DICOM patient LPS.  NIfTI arrays are written with
an explicit LPS-to-RAS affine; values are always queried from the continuous
INR, never resampled from a lower-resolution overview volume.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def lps_to_ras(points_mm: np.ndarray) -> np.ndarray:
    """Convert LPS patient-world points to NIfTI RAS world points."""
    value = np.asarray(points_mm, dtype=float).copy()
    value[..., :2] *= -1.0
    return value


def ras_to_lps(points_mm: np.ndarray) -> np.ndarray:
    """Convert NIfTI RAS world points to DICOM LPS patient-world points."""
    return lps_to_ras(points_mm)


@dataclass(frozen=True)
class PhysicalGrid:
    lower_lps_mm: np.ndarray
    upper_lps_mm: np.ndarray
    spacing_mm: np.ndarray
    shape: tuple[int, int, int]
    affine_ras: np.ndarray

    def lps_to_voxel(self, point_lps_mm: np.ndarray) -> np.ndarray:
        return (np.asarray(point_lps_mm, dtype=float) - self.lower_lps_mm) / self.spacing_mm

    def lps_points(self) -> np.ndarray:
        axes = [self.lower_lps_mm[index] + self.spacing_mm[index] * np.arange(self.shape[index]) for index in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


@dataclass(frozen=True)
class ExportResult:
    path: Path
    grid: PhysicalGrid
    data: np.ndarray


def make_export_grid(lower_lps_mm, upper_lps_mm, spacing_mm: float | np.ndarray) -> PhysicalGrid:
    """Build a constant-spacing grid that covers the requested LPS box.

    The final sampled edge can lie less than one requested spacing beyond the
    requested upper bound.  This preserves isotropic physical spacing while
    guaranteeing the requested cardiac support is not clipped.
    """
    lower = np.asarray(lower_lps_mm, dtype=float)
    upper = np.asarray(upper_lps_mm, dtype=float)
    spacing = np.broadcast_to(np.asarray(spacing_mm, dtype=float), (3,)).copy()
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(np.r_[lower, upper, spacing])):
        raise ValueError("finite 3D LPS bounds and spacing are required")
    if np.any(upper <= lower) or np.any(spacing <= 0):
        raise ValueError("upper bounds must exceed lower bounds and spacing must be positive")
    shape = tuple((np.ceil((upper - lower) / spacing).astype(int) + 1).tolist())
    affine = np.eye(4, dtype=float)
    affine[0, 0], affine[1, 1], affine[2, 2] = -spacing[0], -spacing[1], spacing[2]
    affine[:3, 3] = lps_to_ras(lower)
    sampled_upper = lower + spacing * (np.asarray(shape) - 1)
    return PhysicalGrid(lower, sampled_upper, spacing, shape, affine)


def _normalized(points_lps_mm: torch.Tensor, world_to_normalized: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*points_lps_mm.shape[:-1], 1), dtype=points_lps_mm.dtype, device=points_lps_mm.device)
    return torch.einsum("ij,...j->...i", world_to_normalized.to(points_lps_mm), torch.cat((points_lps_mm, ones), dim=-1))[..., :3]


def _domain_transform(domain: str | Path, device: torch.device) -> torch.Tensor:
    return torch.tensor(json.loads(Path(domain).read_text())["world_to_normalized"], dtype=torch.float32, device=device)


def query_points(model: torch.nn.Module, domain: str | Path, points_lps_mm: np.ndarray | torch.Tensor, *, chunk_points: int = 8192) -> np.ndarray:
    """Query continuous INR values at arbitrary DICOM LPS physical points."""
    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    transform = _domain_transform(domain, device)
    points = torch.as_tensor(points_lps_mm, dtype=torch.float32).reshape(-1, 3)
    values = []
    with torch.no_grad():
        for chunk in points.split(chunk_points):
            intensity = model(_normalized(chunk.to(device), transform)).squeeze(-1)
            values.append(intensity.detach().cpu().numpy())
    output = np.concatenate(values)
    if not np.isfinite(output).all():
        raise RuntimeError("continuous INR export produced NaN/Inf")
    return output.reshape(np.asarray(points_lps_mm).shape[:-1])


def export_volume(model: torch.nn.Module, domain: str | Path, lower_lps_mm, upper_lps_mm, spacing_mm: float, output: str | Path, *, grid: PhysicalGrid | None = None, chunk_points: int = 8192) -> ExportResult:
    """Export a physical grid directly from a continuous canonical INR."""
    result_grid = grid or make_export_grid(lower_lps_mm, upper_lps_mm, spacing_mm)
    data = query_points(model, domain, result_grid.lps_points(), chunk_points=chunk_points).astype(np.float32)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, result_grid.affine_ras), str(path))
    return ExportResult(path, result_grid, data)


def query_physical_plane(model: torch.nn.Module, domain: str | Path, geometry: dict, *, chunk_points: int = 8192) -> np.ndarray:
    """Query the exact DICOM acquisition plane from continuous INR in LPS mm."""
    rows, columns = int(geometry["rows"]), int(geometry["columns"])
    rr, cc = np.indices((rows, columns))
    origin = np.asarray(geometry["image_position_patient"], dtype=float)
    orientation = np.asarray(geometry["image_orientation_patient"], dtype=float)
    spacing = np.asarray(geometry["pixel_spacing"], dtype=float)
    points = origin + cc[..., None] * spacing[1] * orientation[:3] + rr[..., None] * spacing[0] * orientation[3:]
    return query_points(model, domain, points, chunk_points=chunk_points)
