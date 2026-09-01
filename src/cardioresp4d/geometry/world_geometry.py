"""
功能：按 DICOM PS3.3 将二维图像像素坐标映射到病人世界坐标。
论文来源：Necessary adaptation: physical DICOM geometry required for image-domain MRI.
输入：manifest 的 ImagePositionPatient、ImageOrientationPatient、PixelSpacing、Rows 和 Columns。
输出：``DicomPlane``，提供 ``(column, row)`` 像素与毫米病人世界坐标的双向转换。
主要步骤：用列索引乘 PixelSpacing[1] 和 IOP[:3]，用行索引乘 PixelSpacing[0] 和 IOP[3:]。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.geometry.world_geometry --help
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _as_vector(value: Any, length: int, name: str) -> np.ndarray:
    """Return one finite float vector of the required length."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector of length {length}") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite numeric vector of length {length}")
    return vector


def _json_if_string(value: Any, name: str) -> Any:
    """Decode a CSV JSON list while leaving already-decoded loader geometry untouched."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} must be valid JSON when provided as a string") from error
    return value


@dataclass(frozen=True)
class DicomPlane:
    """One DICOM image plane using public pixel coordinates in ``(column, row)`` order.

    ``row_direction`` is ``ImageOrientationPatient[:3]`` and advances as the
    column index advances. ``column_direction`` is ``ImageOrientationPatient[3:]``
    and advances as the row index advances. The normal is
    ``cross(row_direction, column_direction)``.
    """

    origin: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray
    pixel_spacing: np.ndarray
    rows: int
    columns: int
    slice_thickness: float | None = None
    spacing_between_slices: float | None = None

    def __post_init__(self) -> None:
        """Validate immutable DICOM geometry while preserving raw stored direction cosines."""
        origin = _as_vector(self.origin, 3, "image_position_patient")
        row_direction = _as_vector(self.row_direction, 3, "image_orientation_patient[:3]")
        column_direction = _as_vector(self.column_direction, 3, "image_orientation_patient[3:]")
        spacing = _as_vector(self.pixel_spacing, 2, "pixel_spacing")
        if np.any(spacing <= 0.0):
            raise ValueError("pixel_spacing values must be positive")
        row_norm = float(np.linalg.norm(row_direction))
        column_norm = float(np.linalg.norm(column_direction))
        if not np.isclose(row_norm, 1.0, atol=1e-5) or not np.isclose(column_norm, 1.0, atol=1e-5):
            raise ValueError("ImageOrientationPatient directions must have unit length")
        if not np.isclose(float(np.dot(row_direction, column_direction)), 0.0, atol=1e-5):
            raise ValueError("ImageOrientationPatient directions must be orthogonal")
        if int(self.rows) != self.rows or int(self.columns) != self.columns or self.rows <= 0 or self.columns <= 0:
            raise ValueError("rows and columns must be positive integers")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "row_direction", row_direction)
        object.__setattr__(self, "column_direction", column_direction)
        object.__setattr__(self, "pixel_spacing", spacing)
        object.__setattr__(self, "rows", int(self.rows))
        object.__setattr__(self, "columns", int(self.columns))

    @classmethod
    def from_geometry(cls, geometry: Mapping[str, Any]) -> "DicomPlane":
        """Construct a plane from either loader geometry or one CSV manifest row."""
        required = ("image_position_patient", "image_orientation_patient", "pixel_spacing", "rows", "columns")
        missing = [field for field in required if field not in geometry]
        if missing:
            raise ValueError(f"Geometry is missing required field(s): {', '.join(missing)}")
        orientation = _as_vector(_json_if_string(geometry["image_orientation_patient"], "image_orientation_patient"), 6, "image_orientation_patient")
        return cls(
            origin=_as_vector(_json_if_string(geometry["image_position_patient"], "image_position_patient"), 3, "image_position_patient"),
            row_direction=orientation[:3],
            column_direction=orientation[3:],
            pixel_spacing=_as_vector(_json_if_string(geometry["pixel_spacing"], "pixel_spacing"), 2, "pixel_spacing"),
            rows=int(geometry["rows"]),
            columns=int(geometry["columns"]),
            slice_thickness=_optional_float(geometry.get("slice_thickness")),
            spacing_between_slices=_optional_float(geometry.get("spacing_between_slices")),
        )

    @property
    def normal(self) -> np.ndarray:
        """Return the unit patient-world normal ``cross(row_direction, column_direction)``."""
        normal = np.cross(self.row_direction, self.column_direction)
        return normal / np.linalg.norm(normal)

    @property
    def orientation_matrix(self) -> np.ndarray:
        """Return raw DICOM in-plane vectors plus the unit normal as three matrix columns."""
        return np.column_stack((self.row_direction, self.column_direction, self.normal))

    @property
    def column_step(self) -> np.ndarray:
        """Return the millimetre vector induced by increasing the public column index."""
        return self.pixel_spacing[1] * self.row_direction

    @property
    def row_step(self) -> np.ndarray:
        """Return the millimetre vector induced by increasing the public row index."""
        return self.pixel_spacing[0] * self.column_direction

    def pixel_to_world(self, column: Any, row: Any | None = None) -> np.ndarray:
        """Map one or batched ``(column, row)`` pixels to DICOM patient-world millimetres."""
        pixels = _coerce_pixels(column, row)
        return self.origin + pixels[..., 0, None] * self.column_step + pixels[..., 1, None] * self.row_step

    def world_to_pixel(self, world: Any) -> np.ndarray:
        """Orthogonally project one or batched world points to public ``(column, row)`` pixels.

        Off-plane points are intentionally projected, so a later ROI stage can
        project a shared 3D cardiac box through every DICOM plane.
        """
        points = np.asarray(world, dtype=np.float64)
        basis = np.column_stack((self.column_step, self.row_step))
        gram_inverse = np.linalg.inv(basis.T @ basis)
        if points.shape == (3,):
            delta = points - self.origin
            return gram_inverse @ (basis.T @ delta)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("world must have final dimension 3 and contain only finite values")
        delta = points - self.origin
        return np.einsum("ij,...j->...i", gram_inverse @ basis.T, delta)

    def contains_pixel(self, column: float, row: float, tolerance: float = 0.0) -> bool:
        """Return whether a public pixel coordinate lies inside the zero-based image extent."""
        return -tolerance <= column <= self.columns - 1 + tolerance and -tolerance <= row <= self.rows - 1 + tolerance

    def corners(self) -> np.ndarray:
        """Return four patient-world corners at valid pixel index extrema, clockwise."""
        return self.pixel_to_world(np.array(((0.0, 0.0), (self.columns - 1.0, 0.0), (self.columns - 1.0, self.rows - 1.0), (0.0, self.rows - 1.0))))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable geometry representation preserving DICOM field names."""
        return {"image_position_patient": self.origin.tolist(), "image_orientation_patient": np.concatenate((self.row_direction, self.column_direction)).tolist(), "pixel_spacing": self.pixel_spacing.tolist(), "slice_thickness": self.slice_thickness, "spacing_between_slices": self.spacing_between_slices, "rows": self.rows, "columns": self.columns}


def _optional_float(value: Any) -> float | None:
    """Coerce optional scalar DICOM spacing metadata without using it in the in-plane map."""
    if value is None or value == "":
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("optional spacing metadata must be finite")
    return result


def _coerce_pixels(column: Any, row: Any | None) -> np.ndarray:
    """Validate scalar-pair or final-axis-two public pixel coordinates."""
    if row is None:
        pixels = np.asarray(column, dtype=np.float64)
        if pixels.ndim < 1 or pixels.shape[-1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("pixel coordinates must have final dimension 2 and contain only finite values")
        return pixels
    pixels = np.asarray((column, row), dtype=np.float64)
    if pixels.shape != (2,) or not np.isfinite(pixels).all():
        raise ValueError("column and row must be finite scalar coordinates")
    return pixels


def main() -> None:
    """Print help or map one pixel from a supplied DICOM geometry JSON object."""
    parser = argparse.ArgumentParser(description="Map DICOM public (column, row) pixels to patient-world coordinates.")
    parser.add_argument("--geometry-json", help="JSON object containing DICOM geometry fields")
    parser.add_argument("--column", type=float, default=0.0, help="Zero-based DICOM column coordinate")
    parser.add_argument("--row", type=float, default=0.0, help="Zero-based DICOM row coordinate")
    args = parser.parse_args()
    if args.geometry_json is None:
        parser.print_help()
        return
    plane = DicomPlane.from_geometry(json.loads(args.geometry_json))
    print(json.dumps({"world_mm": plane.pixel_to_world(args.column, args.row).tolist()}, indent=2))


if __name__ == "__main__":
    main()
