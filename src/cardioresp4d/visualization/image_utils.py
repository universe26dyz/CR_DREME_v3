"""NIfTI-preserving diagnostic derivatives and scalar summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

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
    shape = tuple(int(value) for value in target.shape[:3]); result = np.empty(shape, dtype=np.float32); model.eval()
    with torch.no_grad():
        for flat_start, flat_end, points in _iter_lattice_world_points(target, chunk_size):
            values = torch.as_tensor(model(torch.from_numpy(points).to(device))).reshape(-1).detach().cpu().numpy()
            result.reshape(-1)[flat_start:flat_end] = values
    return result


def representative_plane_specs(target: nib.spatialimages.SpatialImage) -> list[dict[str, Any]]:
    shape = tuple(int(value) for value in target.shape[:3]); centers = (shape[0] // 2, shape[1] // 2, shape[2] // 2); specs = []
    for axis, name in ((0, "sagittal"), (1, "coronal"), (2, "axial")):
        fixed = centers[axis]; plane_shape = list(shape); plane_shape[axis] = 1; voxel_index = [None, None, None]; voxel_index[axis] = fixed; affine = np.asarray(target.affine, dtype=float).copy(); affine[:3, 3] = nib.affines.apply_affine(target.affine, [fixed if axis == 0 else 0, fixed if axis == 1 else 0, fixed if axis == 2 else 0])
        specs.append({"axis": name, "axis_index": axis, "fixed_index": fixed, "voxel_index": tuple(voxel_index), "shape": tuple(plane_shape), "affine": affine, "world_coordinate": affine[:3, 3].tolist()})
    return specs


def _iter_lattice_world_points(target: nib.spatialimages.SpatialImage, batch_size: int, *, spec: dict[str, Any] | None = None) -> Iterator[tuple[int, int, np.ndarray]]:
    if batch_size <= 0: raise ValueError("batch_size must be positive")
    shape = tuple(int(value) for value in target.shape[:3]) if spec is None else tuple(spec["shape"])
    total = int(np.prod(shape)); axis = None if spec is None else int(spec["axis_index"]); fixed = None if spec is None else int(spec["fixed_index"])
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size); flat = np.arange(start, end, dtype=np.int64); coordinates = np.column_stack(np.unravel_index(flat, shape)).astype(np.float32)
        if spec is not None: coordinates[:, axis] = fixed
        world = nib.affines.apply_affine(target.affine, coordinates).astype(np.float32); yield start, end, world


def evaluate_inr_psf_points(model: torch.nn.Module, points_world_mm: np.ndarray | torch.Tensor, *, device: torch.device, psf_resolution: float, n_samples: int, seed: int, point_batch_size: int = 4096) -> np.ndarray:
    if psf_resolution <= 0 or n_samples <= 0 or point_batch_size <= 0: raise ValueError("positive PSF resolution, sample count, and point batch size are required")
    from nesvor.utils import resolution2sigma
    points = torch.as_tensor(points_world_mm, dtype=torch.float32)
    if points.numel() == 0: return np.empty(0, dtype=np.float32)
    sigma = resolution2sigma(psf_resolution, isotropic=True); values: list[np.ndarray] = []; model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            for start in range(0, len(points), point_batch_size):
                batch = points[start:start + point_batch_size].to(device); sampled = model.sample_batch(batch, None, sigma, n_samples); values.append(torch.as_tensor(model(sampled)).mean(-1).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


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
    """Legacy full-volume PSF helper retained for callers; V6 does not use it."""
    shape = tuple(int(value) for value in target.shape[:3]); result = np.empty(shape, dtype=np.float32)
    for start, end, points in _iter_lattice_world_points(target, chunk_size): result.reshape(-1)[start:end] = evaluate_inr_psf_points(model, points, device=device, psf_resolution=psf_resolution, n_samples=n_samples, seed=seed + start, point_batch_size=chunk_size)
    return result


def evaluate_inr_psf_plane(model: torch.nn.Module, target: nib.spatialimages.SpatialImage, spec: dict[str, Any], *, device: torch.device, psf_resolution: float, n_samples: int, seed: int, point_batch_size: int = 4096) -> np.ndarray:
    shape = tuple(spec["shape"]); point_batches = [points for _, _, points in _iter_lattice_world_points(target, point_batch_size, spec=spec)]
    points = np.concatenate(point_batches, axis=0) if point_batches else np.empty((0, 3), dtype=np.float32)
    return evaluate_inr_psf_points(model, points, device=device, psf_resolution=psf_resolution, n_samples=n_samples, seed=seed, point_batch_size=point_batch_size).reshape(shape)


def select_support_voxel_indices(mask: np.ndarray, *, max_points: int = 100000, chunk_size: int = 1000000) -> tuple[np.ndarray, int]:
    values = np.asarray(mask, dtype=bool); total = int(np.count_nonzero(values)); sample_count = min(total, int(max_points));
    if sample_count <= 0: return np.empty((0, 3), dtype=np.int64), total
    ranks = np.unique(np.rint(np.linspace(0, total - 1, sample_count)).astype(np.int64)); selected: list[int] = []; flat = values.reshape(-1); seen = 0
    for start in range(0, flat.size, chunk_size):
        local = np.flatnonzero(flat[start:start + chunk_size]); ordinals = seen + np.arange(len(local), dtype=np.int64); positions = np.searchsorted(ranks, ordinals)
        hits = np.zeros(len(local), dtype=bool)
        valid = positions < len(ranks)
        hits[valid] = ranks[positions[valid]] == ordinals[valid]
        selected.extend((start + local[hits]).tolist()); seen += len(local)
    return np.asarray(np.unravel_index(np.asarray(selected, dtype=np.int64), values.shape)).T, total


def support_point_agreement(model: torch.nn.Module, formal_values: np.ndarray, support_mask: np.ndarray, affine: np.ndarray, *, device: torch.device, psf_resolution: float, n_samples: int, seed: int, max_points: int = 100000, point_batch_size: int = 4096) -> dict[str, Any]:
    coordinates, total = select_support_voxel_indices(support_mask, max_points=max_points); world = nib.affines.apply_affine(affine, coordinates); predicted = evaluate_inr_psf_points(model, world, device=device, psf_resolution=psf_resolution, n_samples=n_samples, seed=seed, point_batch_size=point_batch_size); observed = np.asarray(formal_values)[tuple(coordinates.T)].astype(np.float32); difference = predicted - observed; correlation = float(np.corrcoef(predicted, observed)[0, 1]) if len(predicted) > 1 and np.std(predicted) > 0 and np.std(observed) > 0 else None
    return {"sample_count": int(len(coordinates)), "total_support_count": total, "sampling_policy": "uniform_index_space_ranks_over_sorted_support", "seed": seed, "psf_resolution": psf_resolution, "n_samples": n_samples, "mae": float(np.mean(np.abs(difference))) if len(difference) else None, "rmse": float(np.sqrt(np.mean(difference ** 2))) if len(difference) else None, "correlation": correlation}
