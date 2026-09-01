"""
功能：以按帧访问的方式读取 CardioResp DICOM manifest 及其重建图像。
论文来源：Necessary adaptation: image-domain DICOM loader for local free-breathing MRI.
输入：Task 1 CSV manifest 和其引用的 DICOM 帧。
输出：归一化 float32 图像、AcquisitionTime 秒数、视图、固定层标识和 DICOM 几何。
主要步骤：读取 manifest，应用 DICOM rescale slope/intercept，再作每帧 1/99 percentile 归一化。
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


class CardioRespDataset:
    """A lightweight manifest-backed DICOM dataset with no framework dependency."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            self._rows = list(csv.DictReader(handle))
        if not self._rows:
            raise ValueError(f"Manifest contains no frames: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        dataset = pydicom.dcmread(row["dicom_path"])
        image = _normalise_image(_rescaled_pixels(dataset))
        return {
            "image": image,
            "timestamp_s": float(row["timestamp_s"]),
            "view": row["view"],
            "slice_id": row["slice_id"],
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
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return dataset.pixel_array.astype(np.float32) * slope + intercept


def _normalise_image(image: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(image, (1.0, 99.0))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Image percentiles must be finite")
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


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
