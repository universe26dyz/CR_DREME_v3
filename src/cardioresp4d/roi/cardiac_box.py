"""
功能：定义一个独立于分割结果的 DREME-style 局部 cardiac 病人世界坐标盒。
论文来源：DREME-MR 的局部 cardiac coordinate box 概念。
输入：明确配置的 patient-world 中心和三轴尺寸（mm），以及受试者 WorldNormalizer。
输出：同一个轴对齐 3D 盒的中心、边界、八角点及归一化坐标。
主要步骤：由 center +/- size/2 构造世界边界，验证完全落在受试者范围内，再应用统一仿射归一化。
是否属于原论文直接实现 / 必要适配 / 可选实验：DREME-style direct concept；patient-world box 是 necessary adaptation，非 segmentation。
命令行使用示例：由 cardioresp4d.roi.roi_qc CLI 创建并验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from cardioresp4d.geometry.coordinate_normalization import WorldNormalizer


def _finite_vector(value: Any, name: str) -> np.ndarray:
    """Coerce one finite three-vector without accepting scalar broadcasting."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite three-dimensional vector") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    return vector


@dataclass(frozen=True)
class CardiacBox:
    """One empirical local cardiac support box, not an anatomical segmentation."""

    center_mm: np.ndarray
    size_mm: np.ndarray

    def __post_init__(self) -> None:
        center = _finite_vector(self.center_mm, "center_mm")
        size = _finite_vector(self.size_mm, "size_mm")
        if np.any(size <= 0.0):
            raise ValueError("size_mm values must be positive")
        object.__setattr__(self, "center_mm", center)
        object.__setattr__(self, "size_mm", size)

    @property
    def min_mm(self) -> np.ndarray:
        return self.center_mm - self.size_mm / 2.0

    @property
    def max_mm(self) -> np.ndarray:
        return self.center_mm + self.size_mm / 2.0

    @property
    def corners_mm(self) -> np.ndarray:
        """Return eight corners in stable binary x/y/z endpoint order."""
        lower, upper = self.min_mm, self.max_mm
        return np.asarray(
            [[lower[0] if x == 0 else upper[0], lower[1] if y == 0 else upper[1], lower[2] if z == 0 else upper[2]] for x, y, z in product((0, 1), repeat=3)],
            dtype=np.float64,
        )

    def validate_within(self, normalizer: WorldNormalizer, tolerance_mm: float = 1e-6) -> None:
        """Reject a local box not fully supported by subject normalization bounds."""
        if tolerance_mm < 0.0 or not np.isfinite(tolerance_mm):
            raise ValueError("tolerance_mm must be a finite nonnegative scalar")
        if np.any(self.min_mm < normalizer.bounds_min_mm - tolerance_mm) or np.any(self.max_mm > normalizer.bounds_max_mm + tolerance_mm):
            raise ValueError(
                "Cardiac box lies outside subject normalization bounds: "
                f"box=[{self.min_mm.tolist()}, {self.max_mm.tolist()}], "
                f"subject=[{normalizer.bounds_min_mm.tolist()}, {normalizer.bounds_max_mm.tolist()}]"
            )

    def normalized(self, normalizer: WorldNormalizer) -> dict[str, np.ndarray]:
        self.validate_within(normalizer)
        return {
            "center": normalizer.normalize(self.center_mm),
            "min": normalizer.normalize(self.min_mm),
            "max": normalizer.normalize(self.max_mm),
            "corners": normalizer.normalize(self.corners_mm),
        }

    def to_dict(self, normalizer: WorldNormalizer | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "definition": "DREME-style independent local cardiac coordinate box; not segmentation",
            "axis_alignment": "DICOM patient-world x/y/z",
            "center_mm": self.center_mm.tolist(),
            "size_mm": self.size_mm.tolist(),
            "min_mm": self.min_mm.tolist(),
            "max_mm": self.max_mm.tolist(),
            "corners_mm": self.corners_mm.tolist(),
        }
        if normalizer is not None:
            payload["normalized"] = {name: values.tolist() for name, values in self.normalized(normalizer).items()}
        return payload


def load_cardiac_box_config(path: str | Path) -> CardiacBox:
    """Load required subject-specific values from an explicit local YAML ``roi`` section."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    roi = raw.get("roi") if isinstance(raw, Mapping) else None
    if not isinstance(roi, Mapping):
        raise ValueError("Configuration must contain an roi mapping")
    required = ("cardiac_box_center_mm", "cardiac_box_size_mm")
    missing = [name for name in required if roi.get(name) is None]
    if missing:
        raise ValueError(f"ROI configuration is missing required field(s): {', '.join(missing)}")
    return CardiacBox(roi["cardiac_box_center_mm"], roi["cardiac_box_size_mm"])
