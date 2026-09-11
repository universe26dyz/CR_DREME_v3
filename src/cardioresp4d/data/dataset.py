"""
功能：以按帧访问的方式读取 CardioResp DICOM manifest 及其重建图像。
论文来源：Necessary adaptation: image-domain DICOM loader for local free-breathing MRI.
输入：Task 1 CSV manifest 和其引用的 DICOM 帧。
输出：归一化 float32 图像、AcquisitionTime 秒数、视图、固定层标识、DICOM 几何及 rescale 状态。
主要步骤：读取 manifest，按 pydicom modality LUT/rescale 规则变换像素，再按 qc-valid frames 拟合 per-series 稳定归一化。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.data.dataset --manifest results/dicom_manifest.csv --index 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from cardioresp4d.data.build_manifest import runtime_sidecar_path

try:
    from pydicom.pixels import apply_modality_lut
except ImportError:  # pydicom < 3.0 compatibility in the supported runtime.
    from pydicom.pixel_data_handlers.util import apply_modality_lut


class CardioRespDataset:
    """A lightweight manifest-backed DICOM dataset with no framework dependency."""

    def __init__(self, manifest_path: str | Path, *, valid_only: bool = True,
                 qc_table_path: str | Path | None = None, use_qc: bool = True,
                 normalization_mode: str = "per_series") -> None:
        self.manifest_path = Path(manifest_path)
        if normalization_mode not in {"per_frame_legacy", "per_series", "per_view", "subject_global"}:
            raise ValueError("normalization_mode must be per_frame_legacy, per_series, per_view, or subject_global")
        self.normalization_mode = normalization_mode
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            self._rows = list(csv.DictReader(handle))
        if not self._rows:
            raise ValueError(f"Manifest contains no frames: {self.manifest_path}")
        self._token_to_path: dict[str, str] = {}
        if "source_file_token" in self._rows[0]:
            sidecar = runtime_sidecar_path(self.manifest_path)
            if not sidecar.is_file():
                raise FileNotFoundError(f"Sensitive runtime path sidecar is missing: {sidecar}")
            with sidecar.open(encoding="utf-8") as handle:
                self._token_to_path = json.load(handle)["token_to_absolute_path"]
        qc_path = (Path(qc_table_path) if qc_table_path else self.manifest_path.parent / "acquisition_qc" / "acquisition_qc.csv") if use_qc else None
        qc_by_token: dict[str, dict[str, str]] = {}
        qc_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
        if qc_path is not None and qc_path.is_file() and "source_file_token" in self._rows[0]:
            validate_qc_table_coverage(self.manifest_path, qc_path)
            with qc_path.open(newline="", encoding="utf-8") as handle:
                qc_rows = list(csv.DictReader(handle))
                qc_by_token = {row["source_file_token"]: row for row in qc_rows if row.get("source_file_token")}
                qc_by_key = {(row["view"], row["slice_id"], row["frame_index"]): row for row in qc_rows}
        elif qc_path is not None and qc_path.is_file():
            validate_qc_table_coverage(self.manifest_path, qc_path)
            with qc_path.open(newline="", encoding="utf-8") as handle:
                qc_rows = list(csv.DictReader(handle))
                qc_by_key = {(row["view"], row["slice_id"], row["frame_index"]): row for row in qc_rows}
        for row in self._rows:
            qc = qc_by_token.get(row.get("source_file_token", ""), qc_by_key.get((row["view"], row["slice_id"], row["frame_index"]), {}))
            row.update({"qc_valid": qc.get("qc_valid", "1"), "qc_reason": qc.get("qc_reason", "not_evaluated"),
                        "qc_ncc": qc.get("qc_ncc", ""), "qc_intensity_scale": qc.get("qc_intensity_scale", ""),
                        "qc_residual": qc.get("qc_residual", "")})
        if valid_only:
            self._rows = [row for row in self._rows if row["qc_valid"] not in ("0", "false", "False")]
            if not self._rows:
                raise ValueError("Acquisition QC leaves no valid observations")
        self._normalization_parameters = self._fit_normalization_parameters()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        path = self._token_to_path.get(row.get("source_file_token", ""), row.get("dicom_path", ""))
        if not path:
            raise ValueError("Manifest row cannot be resolved to a runtime DICOM path")
        dataset = pydicom.dcmread(path)
        rescale_status = _rescale_status(dataset)
        rescaled = _rescaled_pixels(dataset)
        image = _normalise_image(rescaled, self._normalization_parameters.get(self._normalization_key(row)))
        return {
            "image": image,
            "timestamp_s": float(row["timestamp_s"]),
            "view": row["view"],
            "slice_id": row["slice_id"],
            "rescale_status": rescale_status,
            "rescaled_image": rescaled,
            "source_file_token": row.get("source_file_token"),
            "qc_valid": row["qc_valid"] not in ("0", "false", "False"),
            "qc_reason": row["qc_reason"],
            "qc_ncc": _optional_metric(row["qc_ncc"]),
            "qc_intensity_scale": _optional_metric(row["qc_intensity_scale"]),
            "qc_residual": _optional_metric(row["qc_residual"]),
            "geometry": {
                "image_position_patient": json.loads(row["image_position_patient"]),
                "image_orientation_patient": json.loads(row["image_orientation_patient"]),
                "pixel_spacing": json.loads(row["pixel_spacing"]),
                "slice_thickness": float(row["slice_thickness"]),
                "spacing_between_slices": float(row["spacing_between_slices"]),
                "rows": int(row["rows"]),
                "columns": int(row["columns"]),
            },
        }

    @property
    def normalization_parameters(self) -> dict[str, dict[str, float]]:
        """Serializable bounds estimated solely from qc-valid acquired frames."""
        return {key: dict(value) for key, value in self._normalization_parameters.items()}

    @property
    def normalization_group_report(self) -> list[dict[str, Any]]:
        """Auditable membership and percentile provenance for formal preflight."""
        groups: dict[str, list[dict[str, str]]] = {}
        for row in self._rows:
            groups.setdefault(self._normalization_key(row), []).append(row)
        report = []
        for key, rows in sorted(groups.items()):
            parameters = self._normalization_parameters.get(key, {})
            report.append({"group_key": key, "mode": self.normalization_mode, "views": sorted({row["view"] for row in rows}), "slice_ids": sorted({row["slice_id"] for row in rows}), "frame_count": len(rows), "valid_frame_count": sum(row["qc_valid"] not in ("0", "false", "False") and row["qc_reason"] not in {"slice_local_scale_absolute", "manual_exclusion"} for row in rows), "percentile_1": parameters.get("lower"), "percentile_99": parameters.get("upper")})
        return report

    def _normalization_key(self, row: dict[str, str]) -> str:
        if self.normalization_mode == "per_frame_legacy":
            return ""
        if self.normalization_mode == "subject_global":
            return "subject_global"
        if self.normalization_mode == "per_view":
            return f"view:{row['view']}"
        return f"series:{row.get('series_instance_uid') or row['view'] + ':' + row['slice_id']}"

    def _fit_normalization_parameters(self) -> dict[str, dict[str, float]]:
        if self.normalization_mode == "per_frame_legacy":
            return {}
        values: dict[str, list[np.ndarray]] = {}
        for row in self._rows:
            if row["qc_valid"] in ("0", "false", "False") or row["qc_reason"] in {"slice_local_scale_absolute", "manual_exclusion"}:
                continue
            path = self._token_to_path.get(row.get("source_file_token", ""), row.get("dicom_path", ""))
            if not path:
                continue
            pixels = _rescaled_pixels(pydicom.dcmread(path)).reshape(-1)
            # Keep deterministic, bounded memory on real 50-frame locations;
            # small fixtures use every value and therefore remain exact.
            if pixels.size > 8192:
                pixels = pixels[np.linspace(0, pixels.size - 1, 8192, dtype=np.int64)]
            values.setdefault(self._normalization_key(row), []).append(pixels)
        result: dict[str, dict[str, float]] = {}
        for key, arrays in values.items():
            lower, upper = np.percentile(np.concatenate(arrays), (1.0, 99.0))
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError("normalization percentiles must be finite")
            result[key] = {"lower": float(lower), "upper": float(upper), "mode": self.normalization_mode, "estimator": "exact_if_frame_pixels_le_8192_else_deterministic_uniform_frame_subsample", "per_frame_sample_cap": 8192.0}
        if not result:
            raise ValueError("normalization requires at least one qc-valid acquired frame")
        return result


def validate_qc_table_coverage(manifest_path: str | Path, qc_table_path: str | Path) -> None:
    """Require one, and only one, QC decision for every manifest frame."""
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with Path(qc_table_path).open(newline="", encoding="utf-8") as handle:
        qc_rows = list(csv.DictReader(handle))
    token_mode = bool(manifest_rows and "source_file_token" in manifest_rows[0])
    def key(row: dict[str, str]) -> tuple[str, ...]:
        return ((row.get("source_file_token", ""),) if token_mode else
                (row.get("view", ""), row.get("slice_id", ""), row.get("frame_index", "")))
    manifest_keys = [key(row) for row in manifest_rows]
    qc_keys = [key(row) for row in qc_rows]
    if (not manifest_keys or len(qc_keys) != len(manifest_keys) or len(set(qc_keys)) != len(qc_keys)
            or set(qc_keys) != set(manifest_keys)):
        raise ValueError("Acquisition QC table does not cover every manifest frame exactly once")
    if any("qc_valid" not in row for row in qc_rows):
        raise ValueError("Acquisition QC table lacks qc_valid decisions")


def _rescaled_pixels(dataset: pydicom.dataset.Dataset) -> np.ndarray:
    """Apply pydicom's modality LUT only when the rescale pair is complete."""
    if _rescale_status(dataset) == "identity_without_rescale_tags":
        return dataset.pixel_array.astype(np.float32)
    return np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32)


def _rescale_status(dataset: pydicom.dataset.Dataset) -> str:
    """Classify mandatory pair completeness without inventing a missing tag value."""
    has_slope = hasattr(dataset, "RescaleSlope")
    has_intercept = hasattr(dataset, "RescaleIntercept")
    if has_slope != has_intercept:
        raise ValueError("DICOM frame has partial rescale metadata: require both RescaleSlope and RescaleIntercept")
    return "modality_lut" if has_slope else "identity_without_rescale_tags"


def _normalise_image(image: np.ndarray, parameters: dict[str, float] | None = None) -> np.ndarray:
    lower, upper = (np.percentile(image, (1.0, 99.0)) if parameters is None else (parameters["lower"], parameters["upper"]))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Image percentiles must be finite")
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def _optional_metric(value: str) -> float | None:
    return None if value in (None, "") else float(value)


def main() -> None:
    """Print the metadata and image range for one manifest-backed DICOM frame."""
    parser = argparse.ArgumentParser(description="Load one normalized CardioResp DICOM frame.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV manifest path")
    parser.add_argument("--index", type=int, default=0, help="Zero-based frame index")
    args = parser.parse_args()
    sample = CardioRespDataset(args.manifest)[args.index]
    sample["image"] = {"shape": list(sample["image"].shape), "min": float(sample["image"].min()), "max": float(sample["image"].max())}
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
