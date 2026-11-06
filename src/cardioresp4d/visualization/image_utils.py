"""NIfTI-preserving diagnostic derivatives and scalar summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import torch


def volume_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "max": float(array.max()), **{f"p{key}": float(np.percentile(array, key)) for key in (1, 5, 50, 95, 99, 99.9)}, "zero_fraction": float(np.mean(array == 0)), "below_zero_fraction": float(np.mean(array < 0)), "above_one_fraction": float(np.mean(array > 1))}


def export_support_mask(mask: torch.Tensor | np.ndarray, affine_ras: np.ndarray, output: str | Path) -> Path:
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    array = torch.as_tensor(mask).detach().cpu().numpy().astype(np.uint8, copy=False)
    nib.save(nib.Nifti1Image(array, np.asarray(affine_ras, dtype=float)), str(path))
    return path


def save_unmasked_static_inr(values: np.ndarray, target: nib.spatialimages.SpatialImage, output: str | Path) -> Path:
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32)
    if array.shape != target.shape:
        raise ValueError("unmasked INR derivative must use target static lattice shape")
    nib.save(nib.Nifti1Image(array, target.affine, target.header.copy()), str(path))
    return path


def evaluate_inr_on_nifti_lattice(model: torch.nn.Module, target: nib.spatialimages.SpatialImage, *, device: torch.device, chunk_size: int = 65536) -> np.ndarray:
    shape = tuple(int(value) for value in target.shape[:3])
    voxel = np.indices(shape, dtype=np.float32).reshape(3, -1).T
    points = nib.affines.apply_affine(target.affine, voxel).astype(np.float32)
    outputs: list[np.ndarray] = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            result = model(torch.from_numpy(points[start:start + chunk_size]).to(device))
            outputs.append(torch.as_tensor(result).reshape(-1).detach().cpu().numpy())
    return np.concatenate(outputs).reshape(shape).astype(np.float32, copy=False)


def support_metrics(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    from scipy import ndimage
    array = np.asarray(mask).astype(bool)
    count = int(array.sum()); labels, components = ndimage.label(array)
    sizes = np.bincount(labels.ravel())[1:] if components else np.array([], dtype=int)
    points = np.argwhere(array)
    bounds = None if not len(points) else nib.affines.apply_affine(affine, np.vstack((points.min(0), points.max(0)))).tolist()
    return {"mask_voxel_count": count, "occupancy_fraction": float(array.mean()), "world_bbox_ras_mm": bounds, "connected_component_count": int(components), "largest_component_fraction": float(sizes.max() / count) if count else 0.}
