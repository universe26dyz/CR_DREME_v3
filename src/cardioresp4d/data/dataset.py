"""
功能：以按帧访问的方式读取 CardioResp DICOM manifest 及其重建图像。
论文来源：Necessary adaptation: image-domain DICOM loader for local free-breathing MRI.
输入：Task 1 CSV manifest 和其引用的 DICOM 帧。
输出：归一化 float32 图像、AcquisitionTime 秒数、视图、固定层标识、DICOM 几何及 rescale 状态。
主要步骤：读取 manifest，按 pydicom modality LUT/rescale 规则变换像素，再作每帧 1/99 percentile 归一化。
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
                 qc_table_path: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
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
        qc_path = Path(qc_table_path) if qc_table_path else self.manifest_path.parent / "acquisition_qc" / "acquisition_qc.csv"
        qc_by_token: dict[str, dict[str, str]] = {}
        qc_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
        if qc_path.is_file() and "source_file_token" in self._rows[0]:
            with qc_path.open(newline="", encoding="utf-8") as handle:
                qc_rows = list(csv.DictReader(handle))
                qc_by_token = {row["source_file_token"]: row for row in qc_rows if row.get("source_file_token")}
                qc_by_key = {(row["view"], row["slice_id"], row["frame_index"]): row for row in qc_rows}
        elif qc_path.is_file():
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
        image = _normalise_image(rescaled)
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


def _normalise_image(image: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(image, (1.0, 99.0))
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
