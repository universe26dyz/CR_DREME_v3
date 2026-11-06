"""Pure geometry helpers shared by experiment audit adapters."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def location_token(view: str, slice_id: str) -> str:
    import hashlib
    return f"{view.lower()}_{hashlib.sha256(str(slice_id).encode()).hexdigest()[:10]}"


def ordered_representative_locations(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered = sorted(records, key=lambda row: (int(row.get("stack_index", 0)), str(row.get("slice_id", ""))))
    if not ordered:
        return []
    indices = sorted(set((0, len(ordered) // 2, len(ordered) - 1)))
    return [ordered[index] for index in indices]


def stack_tiles(records: Sequence[Mapping[str, Any]], *, lattice_size: int) -> list[dict[str, Any]]:
    by_index = {int(row["stack_index"]): row for row in records}
    return [
        ({"real": True, "label": f"{row['view']} {location_token(str(row['view']), str(row['slice_id']))} lattice {index}", **dict(row)} if (row := by_index.get(index)) else
         {"real": False, "label": "MISSING PLACEHOLDER", "stack_index": index})
        for index in range(lattice_size)
    ]


def plane_corners_lps(geometry: Mapping[str, Any]) -> np.ndarray:
    _, height, width = (int(value) for value in geometry["image_shape"])
    center = np.asarray(geometry["center_mm"], dtype=float)
    row = np.asarray(geometry["row_direction"], dtype=float)
    column = np.asarray(geometry["column_direction"], dtype=float)
    row_spacing, column_spacing = (float(value) for value in geometry["pixel_spacing_mm"])
    return np.asarray([
        center + (u - (width - 1) / 2) * column_spacing * row + (v - (height - 1) / 2) * row_spacing * column
        for u in (0., width - 1.) for v in (0., height - 1.)
    ])


def nifti_plane_corners_lps(affine_ras: np.ndarray, shape: Sequence[int], stack_index: int) -> np.ndarray:
    width, height = int(shape[0]), int(shape[1])
    voxels = np.asarray([[u, v, stack_index, 1.] for u in (0., width - 1.) for v in (0., height - 1.)])
    return (LPS_RAS @ np.asarray(affine_ras) @ voxels.T).T[:, :3]


def compare_plane_corners_lps(expected_lps: np.ndarray, affine_ras: np.ndarray, shape: Sequence[int], stack_index: int) -> dict[str, float]:
    actual = nifti_plane_corners_lps(affine_ras, shape, stack_index)
    errors = np.linalg.norm(np.asarray(expected_lps) - actual, axis=1)
    return {"max_error_mm": float(errors.max()), "mean_error_mm": float(errors.mean())}


def detect_geometry_faults(expected: np.ndarray, actual: np.ndarray, *, tolerance_mm: float = 1e-4) -> list[str]:
    """Classify common synthetic geometry failures for visible audit annotations."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    flags: list[str] = []
    if expected.shape != actual.shape:
        return ["shape_mismatch"]
    if np.max(np.linalg.norm(expected - actual, axis=1)) <= tolerance_mm:
        return flags
    if len(expected) >= 4 and np.max(np.linalg.norm(expected - actual[[1, 0, 3, 2]], axis=1)) <= tolerance_mm:
        flags.append("axis_swap")
    for axis in range(3):
        flipped = actual.copy(); flipped[:, axis] *= -1
        if np.max(np.linalg.norm(expected - flipped, axis=1)) <= tolerance_mm:
            flags.append("sign_flip")
    if len(expected) == 2 and np.max(np.linalg.norm(expected - actual[::-1], axis=1)) <= tolerance_mm:
        flags.append("reversed_stack_order")
    return flags or ["world_coordinate_mismatch"]


def pose_delta(original: Mapping[str, Any], corrected: Mapping[str, Any]) -> dict[str, float]:
    center_delta = np.asarray(corrected["center_mm"], dtype=float) - np.asarray(original["center_mm"], dtype=float)
    normal_a = np.asarray(original["normal"], dtype=float); normal_b = np.asarray(corrected["normal"], dtype=float)
    normal_angle = math.degrees(math.acos(np.clip(np.dot(normal_a, normal_b) / (np.linalg.norm(normal_a) * np.linalg.norm(normal_b)), -1., 1.)))
    row_a = np.asarray(original["row_direction"], dtype=float); row_b = np.asarray(corrected["row_direction"], dtype=float)
    column_a = np.asarray(original["column_direction"], dtype=float); column_b = np.asarray(corrected["column_direction"], dtype=float)
    def orientation(row: np.ndarray, column: np.ndarray, normal: np.ndarray) -> np.ndarray:
        row = row / np.linalg.norm(row); column = column - row * np.dot(row, column); column = column / np.linalg.norm(column)
        computed_normal = np.cross(row, column); computed_normal *= 1. if np.dot(computed_normal, normal) >= 0 else -1.
        return np.column_stack((row, column, computed_normal))
    matrix_a = orientation(row_a, column_a, normal_a); matrix_b = orientation(row_b, column_b, normal_b)
    delta = matrix_b @ matrix_a.T
    rotation = math.degrees(math.acos(np.clip((np.trace(delta) - 1.) / 2., -1., 1.)))
    return {"translation_mm": float(np.linalg.norm(center_delta)), "rotation_deg": float(rotation), "center_displacement_mm": float(np.linalg.norm(center_delta)), "normal_angle_deg": float(normal_angle)}


def numeric_corrected_slices(directory: str | Path) -> list[Path]:
    pattern = re.compile(r"^[0-9]+\.nii(?:\.gz)?$")
    return sorted((path for path in Path(directory).iterdir() if path.is_file() and pattern.fullmatch(path.name)), key=lambda path: int(path.name.split(".", 1)[0]))
