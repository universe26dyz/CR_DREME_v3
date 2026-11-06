"""Matplotlib-only rendering utilities; they never modify source arrays."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fixed_window(values: np.ndarray, *, robust: bool = False) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    low, high = np.percentile(array, (1., 99.)) if robust else (float(array.min()), float(array.max()))
    return (float(low), float(high if high > low else low + 1e-6))


def render_paired_images(output: str | Path, images: Sequence[np.ndarray], titles: Sequence[str], *, window: tuple[float, float]) -> dict[str, float | str]:
    if len(images) != len(titles) or not images:
        raise ValueError("paired rendering needs matching nonempty images and titles")
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
    for axis, image, title in zip(axes.ravel(), images, titles):
        axis.imshow(np.asarray(image), cmap="gray", vmin=window[0], vmax=window[1]); axis.set_title(title); axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)
    return {"kind": "fixed", "vmin": float(window[0]), "vmax": float(window[1])}


def render_independent_images(output: str | Path, images: Sequence[np.ndarray], titles: Sequence[str], windows: Sequence[tuple[float, float]]) -> dict[str, object]:
    if len(images) != len(titles) or len(images) != len(windows) or not images:
        raise ValueError("independent rendering needs matching nonempty images, titles, and windows")
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
    for axis, image, title, window in zip(axes.ravel(), images, titles, windows):
        axis.imshow(np.asarray(image), cmap="gray", vmin=window[0], vmax=window[1]); axis.set_title(title); axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)
    return {title: {"kind": "fixed", "vmin": float(window[0]), "vmax": float(window[1])} for title, window in zip(titles, windows)}


def render_montage(output: str | Path, images: Sequence[np.ndarray], labels: Sequence[str], *, columns: int = 6, window: tuple[float, float] | None = None) -> dict[str, float | str]:
    if len(images) != len(labels): raise ValueError("montage images/labels mismatch")
    rows = max(1, int(np.ceil(len(images) / columns))); figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    policy = {"kind": "fixed", "vmin": float(window[0]), "vmax": float(window[1])} if window else {"kind": "per_image_robust"}
    for axis, image, label in zip(axes.ravel(), images, labels):
        limits = window or fixed_window(image, robust=True)
        axis.imshow(np.asarray(image), cmap="gray", vmin=limits[0], vmax=limits[1]); axis.set_title(label, fontsize=8); axis.axis("off")
    for axis in axes.ravel()[len(images):]: axis.axis("off")
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)
    return policy
