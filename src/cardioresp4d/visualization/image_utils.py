"""NIfTI-preserving diagnostic derivatives and scalar summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import nibabel as nib
import numpy as np
import torch


def volume_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "max": float(array.max()), **{f"p{key}": float(np.percentile(array, key)) for key in (1, 5, 50, 95, 99, 99.9)}, "zero_fraction": float(np.mean(array == 0)), "below_zero_fraction": float(np.mean(array < 0)), "above_one_fraction": float(np.mean(array > 1))}


def aggregate_clip_fractions(frames: Iterable[np.ndarray], *, lower: float, upper: float) -> dict[str, float | str]:
    """Compute clip fractions over every pixel in a QC-valid series."""
    if upper < lower:
        raise ValueError("upper clip threshold must be >= lower threshold")
    total = low = high = 0
    for frame in frames:
        values = np.asarray(frame)
        total += values.size
        low += int(np.count_nonzero(values <= lower))
        high += int(np.count_nonzero(values >= upper))
    if total == 0:
        raise ValueError("clip fractions require at least one pixel")
    return {"low_clip_fraction": low / total, "high_clip_fraction": high / total, "fraction_estimator": "exact_all_qc_valid_pixels"}


def export_support_mask(mask: torch.Tensor | np.ndarray, affine_ras: np.ndarray, output: str | Path) -> Path:
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    # NeSVoR Volume.image/mask is (D,H,W), while save_nii_volume writes
    # volume.transpose(2,1,0) as the NIfTI (X,Y,Z) array.
    array = torch.as_tensor(mask).detach().cpu().numpy().astype(np.uint8, copy=False).transpose(2, 1, 0)
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
    if not len(points):
        voxel_bbox = world_bbox_ras = world_bbox_lps = None
    else:
        voxel_min, voxel_max = points.min(0), points.max(0)
        voxel_bbox = [voxel_min.tolist(), voxel_max.tolist()]
        corners = np.asarray([[x, y, z] for x in (voxel_min[0], voxel_max[0]) for y in (voxel_min[1], voxel_max[1]) for z in (voxel_min[2], voxel_max[2])])
        world = nib.affines.apply_affine(affine, corners)
        world_bbox_ras = [world.min(0).tolist(), world.max(0).tolist()]
        world_lps = world * np.asarray([-1., -1., 1.])
        world_bbox_lps = [world_lps.min(0).tolist(), world_lps.max(0).tolist()]
    return {"mask_voxel_count": count, "occupancy_fraction": float(array.mean()), "voxel_index_bbox": voxel_bbox, "world_bbox_ras_mm": world_bbox_ras, "world_bbox_lps_mm": world_bbox_lps, "connected_component_count": int(components), "largest_component_fraction": float(sizes.max() / count) if count else 0.}


def evaluate_inr_psf_on_nifti_lattice(model: torch.nn.Module, target: nib.spatialimages.SpatialImage, *, device: torch.device, psf_resolution: float, n_samples: int, seed: int, chunk_size: int = 65536) -> np.ndarray:
    """Evaluate NeSVoR's isotropic Monte-Carlo PSF semantics at every output voxel."""
    if psf_resolution <= 0 or n_samples <= 0:
        raise ValueError("PSF resolution and sample count must be positive")
    from nesvor.utils import resolution2sigma
    shape = tuple(int(value) for value in target.shape[:3]); voxel = np.indices(shape, dtype=np.float32).reshape(3, -1).T
    points = torch.from_numpy(nib.affines.apply_affine(target.affine, voxel).astype(np.float32)).to(device)
    sigma = resolution2sigma(psf_resolution, isotropic=True)
    values: list[np.ndarray] = []; model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            for start in range(0, len(points), chunk_size):
                batch = points[start:start + chunk_size]
                sampled = model.sample_batch(batch, None, sigma, n_samples)
                values.append(model(sampled).mean(-1).detach().cpu().numpy())
    return np.concatenate(values).reshape(shape).astype(np.float32, copy=False)
