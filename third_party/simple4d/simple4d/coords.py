"""Coordinate conventions and sampling helpers.

Normalised coordinates are cell-centred in [-1, 1] and ordered (depth, width,
height) along the last axis; volumes are stored as [B, height, width, depth].
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def index_to_norm(idx, size):
    """Continuous voxel index -> normalised cell-centre coordinate."""
    return -1.0 + (2.0 * idx + 1.0) / size


def make_coords(shape):
    """Coordinates of a full grid. shape = (B, h, w, d) -> [B, h*w*d, 3] ordered (d, w, h)."""
    batch_size, spatial = shape[0], shape[1:]
    axes = [torch.linspace(-1, 1, s + 1)[:-1] + 1 / s for s in spatial]
    grid = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack(grid[::-1], dim=-1).view(-1, len(spatial))
    return coords.unsqueeze(0).repeat(batch_size, 1, 1)


def make_slice_coords(slice_shape, slice_position, w, d):
    """Coordinates of one coronal slice per batch item.

    slice_shape = (B, H, W); slice_position: [B] integer depth indices.
    Returns [B, H*w, 3] ordered (d, w, h).
    """
    B, H, _ = slice_shape
    axes = [torch.linspace(-1, 1, s + 1)[:-1] + 1 / s for s in (H, w, d)]
    out = []
    for b in range(B):
        a = list(axes)
        p = int(slice_position[b])
        a[-1] = a[-1][p:p + 1]
        g = torch.meshgrid(*a, indexing="ij")
        out.append(torch.stack(g[::-1], dim=-1).view(-1, 3))
    return torch.stack(out, dim=0)


def _sample_3d(volume, coords):
    """Trilinear sampling of [B, h, w, d] at normalised coords [B, n, 3] -> [B, n, 1]."""
    v = volume.unsqueeze(1)
    c = coords.unsqueeze(2).unsqueeze(2)
    return F.grid_sample(v, c, mode="bilinear", padding_mode="border", align_corners=True).squeeze(1).squeeze(-1)


def sample_points(coords, num_points, image):
    """Randomly sample ``num_points`` pixels of each slice.

    coords: [B, N, 3] slice coordinates; image: [B, H, W].
    Returns sampled coordinates [B, n, 3] and intensities [B, n, 1].
    """
    B, N, _ = coords.shape
    idx = torch.randint(0, N, (B, num_points))
    sampled = torch.stack([coords[b, idx[b]] for b in range(B)])
    # the slice is repeated along depth so that the depth coordinate is irrelevant
    vol = image.unsqueeze(-1).repeat(1, 1, 1, 64)
    return sampled, _sample_3d(vol, sampled)


def chunks(coords, n_chunks) -> List[torch.Tensor]:
    """Split [B, N, 3] into ``n_chunks`` contiguous chunks along N."""
    return list(torch.chunk(coords, n_chunks, dim=1))


def weight_ramp(step, slope=0.2, max_weight=10.0):
    """Linear ramp w_s(t) = min(slope * t, max_weight) of the DVF regularisers."""
    return float(np.clip(slope * step if step > 0 else 0.0, 0.0, max_weight))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
