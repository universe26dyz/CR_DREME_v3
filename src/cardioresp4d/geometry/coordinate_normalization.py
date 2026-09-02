"""
功能：把所有 DICOM 平面角点的病人世界包围盒归一化到固定坐标范围。
论文来源：Necessary adaptation: subject-level world-coordinate normalization for local MRI.
输入：一个或多个 ``DicomPlane`` 及其毫米病人世界角点。
输出：可序列化的 4x4 前向/逆向矩阵和支持批量点的 ``WorldNormalizer``。
主要步骤：拟合所有平面角点的轴对齐包围盒，并线性映射到每轴 ``[-1, 1]``。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.geometry.coordinate_normalization --help
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cardioresp4d.geometry.world_geometry import DicomPlane


@dataclass(frozen=True)
class WorldNormalizer:
    """Affine subject normalization fitted from all physical DICOM plane corners.

    ``forward_matrix`` maps homogeneous patient-world millimetres to normalized
    coordinates and ``inverse_matrix`` maps them back. The target interval is
    ``[-1, 1]`` on every axis. A zero-extent physical axis is expanded by one
    millimetre around its shared coordinate so the serialized 4x4 mapping
    remains invertible for a valid single or coplanar acquisition.
    """

    forward_matrix: np.ndarray
    inverse_matrix: np.ndarray
    bounds_min_mm: np.ndarray
    bounds_max_mm: np.ndarray

    def __post_init__(self) -> None:
        """Validate supplied serializable affine matrices and bounding-box metadata."""
        forward = np.asarray(self.forward_matrix, dtype=np.float64)
        inverse = np.asarray(self.inverse_matrix, dtype=np.float64)
        lower = np.asarray(self.bounds_min_mm, dtype=np.float64)
        upper = np.asarray(self.bounds_max_mm, dtype=np.float64)
        if forward.shape != (4, 4) or inverse.shape != (4, 4):
            raise ValueError("forward_matrix and inverse_matrix must both be 4x4")
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("bounds must be finite three-dimensional vectors")
        if np.any(upper <= lower):
            raise ValueError("subject bounding box must have positive extent on every axis")
        if not np.allclose(forward @ inverse, np.eye(4), atol=1e-10):
            raise ValueError("forward_matrix and inverse_matrix must be mutual inverses")
        object.__setattr__(self, "forward_matrix", forward)
        object.__setattr__(self, "inverse_matrix", inverse)
        object.__setattr__(self, "bounds_min_mm", lower)
        object.__setattr__(self, "bounds_max_mm", upper)

    @classmethod
    def from_planes(cls, planes: Iterable[DicomPlane]) -> "WorldNormalizer":
        """Fit the normalized subject box from all corners of every supplied DICOM plane."""
        plane_list = list(planes)
        if not plane_list:
            raise ValueError("At least one DICOM plane is required for normalization")
        return cls.from_corners(np.concatenate([plane.corners() for plane in plane_list], axis=0))

    @classmethod
    def from_corners(cls, corners_mm: Any) -> "WorldNormalizer":
        """Fit an invertible ``[-1, 1]^3`` affine transform from physical corner points."""
        corners = np.asarray(corners_mm, dtype=np.float64)
        if corners.ndim != 2 or corners.shape[0] == 0 or corners.shape[1] != 3 or not np.isfinite(corners).all():
            raise ValueError("corners_mm must have shape (N, 3), N > 0, with finite values")
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        extent = upper - lower
        degenerate = extent <= np.finfo(np.float64).eps
        if np.any(degenerate):
            lower = lower.copy()
            upper = upper.copy()
            lower[degenerate] -= 0.5
            upper[degenerate] += 0.5
            extent = upper - lower
        forward = np.eye(4)
        forward[np.arange(3), np.arange(3)] = 2.0 / extent
        forward[:3, 3] = -(upper + lower) / extent
        return cls(forward, np.linalg.inv(forward), lower, upper)

    def normalize(self, world_mm: Any) -> np.ndarray:
        """Transform one or batched patient-world points from millimetres to normalized coordinates."""
        return _transform_points(self.forward_matrix, world_mm, "world_mm")

    def denormalize(self, normalized: Any) -> np.ndarray:
        """Transform one or batched normalized coordinates back to patient-world millimetres."""
        return _transform_points(self.inverse_matrix, normalized, "normalized")

    world_to_normalized = normalize
    normalized_to_world = denormalize

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe 4x4 matrices and physical bounding-box endpoints."""
        return {"forward_matrix": self.forward_matrix.tolist(), "inverse_matrix": self.inverse_matrix.tolist(), "bounds_min_mm": self.bounds_min_mm.tolist(), "bounds_max_mm": self.bounds_max_mm.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorldNormalizer":
        """Restore a normalizer from the dictionary emitted by ``to_dict``."""
        required = ("forward_matrix", "inverse_matrix", "bounds_min_mm", "bounds_max_mm")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"WorldNormalizer payload is missing field(s): {', '.join(missing)}")
        return cls(payload["forward_matrix"], payload["inverse_matrix"], payload["bounds_min_mm"], payload["bounds_max_mm"])


def _transform_points(matrix: np.ndarray, points: Any, name: str) -> np.ndarray:
    """Apply a homogeneous affine matrix while preserving input leading batch dimensions."""
    values = np.asarray(points, dtype=np.float64)
    if values.shape == (3,):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must have final dimension 3 and contain only finite values")
        return (matrix @ np.append(values, 1.0))[:3]
    if values.ndim < 1 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"{name} must have final dimension 3 and contain only finite values")
    homogeneous = np.concatenate((values, np.ones((*values.shape[:-1], 1), dtype=np.float64)), axis=-1)
    return np.einsum("ij,...j->...i", matrix, homogeneous)[..., :3]


def main() -> None:
    """Print help or fit and serialize a normalizer from an explicit JSON corner array."""
    parser = argparse.ArgumentParser(description="Fit a subject patient-world normalizer from DICOM plane corners.")
    parser.add_argument("--corners-json", help="JSON array with shape (N, 3) in patient-world millimetres")
    args = parser.parse_args()
    if args.corners_json is None:
        parser.print_help()
        return
    print(json.dumps(WorldNormalizer.from_corners(json.loads(args.corners_json)).to_dict(), indent=2))


if __name__ == "__main__":
    main()
