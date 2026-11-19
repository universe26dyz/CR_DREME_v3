"""Geometry-derived cardiac support primitives (patient-world millimetres)."""
from __future__ import annotations

import numpy as np


def central_plane(observations, *, stack_normal=None):
    """Choose the lower median QC-valid location after physical-coordinate sorting.

    Locations use the sampler's (view, slice_id) grouping. Normals must be
    aligned to a stack direction, and frames within a location must agree.
    Pass the acquisition stack normal when corrected planes are not parallel.
    """
    grouped = {}
    for obs in observations:
        if obs.qc_valid:
            grouped.setdefault((obs.view.upper(), obs.slice_id), []).append(obs)
    if not grouped or len({key[0] for key in grouped}) != 1:
        raise ValueError('expected QC-valid observations from exactly one view')
    reference = np.array(stack_normal if stack_normal is not None else next(iter(grouped.values()))[0].normal, dtype=float, copy=True)
    reference /= np.linalg.norm(reference)
    planes = []
    for key, frames in grouped.items():
        center = np.asarray(frames[0].center_mm, dtype=float)
        normal = np.array(frames[0].normal, dtype=float, copy=True)
        normal /= np.linalg.norm(normal)
        for frame in frames:
            n = np.array(frame.normal, dtype=float, copy=True)
            n /= np.linalg.norm(n)
            if not np.allclose(frame.center_mm, center, atol=1e-4, rtol=0) or abs(n @ normal) < 1 - 1e-4:
                raise ValueError(f'inconsistent geometry for {key}')
        if stack_normal is None and abs(normal @ reference) < 1 - 1e-4:
            raise ValueError('stack normals are not parallel; provide acquisition stack normal')
        planes.append((float(center @ reference), key[1], frames[0]))
    planes.sort(key=lambda p: (p[0], p[1]))
    return planes[(len(planes) - 1) // 2], (planes[0][0], planes[-1][0])


def intersect_planes(origins, normals, max_condition=1000.0):
    """Least-squares intersection; reject insufficiently constrained geometry."""
    origins = np.asarray(origins, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if origins.shape != (3, 3) or normals.shape != (3, 3) or not np.isfinite(origins).all() or not np.isfinite(normals).all():
        raise ValueError('expected three finite 3D planes')
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms == 0):
        raise ValueError('plane normal is zero')
    n = normals / norms[:, None]
    d = np.sum(n * origins, axis=1)
    center, _, rank, singular = np.linalg.lstsq(n, d, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] else float('inf')
    if rank < 3 or condition > max_condition:
        raise ValueError(f'ill-conditioned plane geometry: rank={rank}, condition={condition:.3g}')
    return center, n @ center - d, condition


def soft_box(points, box, taper_mm):
    """Separable cosine taper inside the box, one in its core and zero outside."""
    if not np.isfinite(taper_mm) or taper_mm <= 0 or taper_mm > min(box.size_mm) / 2:
        raise ValueError('invalid taper width')
    distance = box.size_mm / 2 - np.abs(np.asarray(points) - box.center_mm)
    t = np.clip(distance / taper_mm, 0, 1)
    return np.prod(0.5 - 0.5 * np.cos(np.pi * t), axis=-1)


def soft_box_torch(points, box, taper_mm):
    """Differentiable world-mm equivalent of soft_box for model coordinates."""
    import torch
    if not np.isfinite(taper_mm) or taper_mm <= 0 or taper_mm > min(box.size_mm) / 2:
        raise ValueError('invalid taper width')
    center = points.new_tensor(box.center_mm)
    half = points.new_tensor(box.size_mm) / 2
    t = ((half - (points - center).abs()) / taper_mm).clamp(0, 1)
    return (0.5 - 0.5 * torch.cos(torch.pi * t)).prod(dim=-1)


def observation_soft_roi(observation, box, taper_mm):
    """Sample the shared 3D ROI at each corrected observation pixel center."""
    import torch
    from cardioresp4d.training.model import SourceFirstDynamicModel
    height, width = observation.image.shape[-2:]
    row, col = torch.meshgrid(torch.arange(height, device=observation.image.device), torch.arange(width, device=observation.image.device), indexing='ij')
    pixels = torch.stack((col, row), dim=-1).reshape(-1, 2).to(observation.image)
    points = SourceFirstDynamicModel._pixel_world(observation, pixels)
    return soft_box_torch(points, box, taper_mm).reshape(height, width)
