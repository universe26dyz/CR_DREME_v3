"""Input format and dataset.

A scan is stored as one ``.npz`` file (see ``tools/dicom_to_npz.py``) with

    frames          float32 [N, H, W]  preprocessed slices in [0, 1], in acquisition order
    slice_index     int     [N]        slice-position index in 0..D-1 (sorted along the slice normal)
    spacing         float   [3]        (slice spacing, row spacing, column spacing) in mm
    num_positions   int                D (optional; default max(slice_index) + 1)
    depth_crop      int     [2]        slice positions excluded at the start / end of the stack
                                       from the photometric loss (optional; default [0, 0])

The last ``n_holdout`` frames are held out for evaluation (within-scan temporal hold-out).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SliceSeries:
    frames: np.ndarray
    slice_index: np.ndarray
    spacing: tuple
    num_positions: int
    depth_crop: tuple

    @classmethod
    def load(cls, path, depth_crop=None):
        z = np.load(path)
        frames = z["frames"].astype(np.float32)
        slice_index = z["slice_index"].astype(np.int64)
        if frames.ndim != 3 or len(frames) != len(slice_index):
            raise ValueError("frames must be [N, H, W] and slice_index [N]")
        D = int(z["num_positions"]) if "num_positions" in z.files else int(slice_index.max()) + 1
        if depth_crop is None:
            depth_crop = tuple(int(v) for v in z["depth_crop"]) if "depth_crop" in z.files else (0, 0)
        return cls(frames, slice_index, tuple(float(v) for v in z["spacing"]), D, tuple(depth_crop))

    def __len__(self):
        return len(self.frames)

    def split(self, n_holdout):
        """Indices of the optimisation frames and of the held-out (last ``n_holdout``) frames."""
        n = len(self)
        return np.arange(0, n - n_holdout), np.arange(n - n_holdout, n)


class SliceDataset(Dataset):
    """Frames of one scan restricted to ``indices`` (kept in acquisition order)."""

    def __init__(self, series: SliceSeries, indices):
        self.series = series
        self.indices = np.asarray(indices)
        self._frames = [torch.from_numpy(series.frames[i]) for i in range(len(series))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = int(self.indices[i])
        gt = self._frames[t]
        return dict(idx=i, frame=t, slice_position=int(self.series.slice_index[t]), gt=gt)
