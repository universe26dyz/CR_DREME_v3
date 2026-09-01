"""
功能：从 DICOM 扫描结果生成供后续几何、频率和参考模块共用的权威 manifest。
论文来源：Necessary adaptation: image-domain manifest carrying acquisition time and plane metadata.
输入：AppConfig、只读 DICOM 目录和选定视图。
输出：无 PHI 的 CSV manifest 与等价 JSON manifest。
主要步骤：扫描帧、展平几何字段、写入稳定排序的 CSV/JSON，并记录结构摘要。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.data.build_manifest --config configs/subject_local.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cardioresp4d.config import AppConfig, load_config
from cardioresp4d.data.inspect_dataset import scan_dicom_frames


MANIFEST_COLUMNS = (
    "dicom_path", "view", "slice_id", "frame_index", "timestamp_s",
    "image_position_patient", "image_orientation_patient", "pixel_spacing",
    "slice_thickness", "spacing_between_slices", "rows", "columns",
)


def build_manifest(config: AppConfig) -> tuple[Path, Path]:
    """Write the canonical CSV and JSON manifests and return their paths."""
    frames = scan_dicom_frames(config.dicom_root, config.views)
    rows = [_flatten_frame(frame) for frame in frames]
    config.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.results_dir / f"{config.manifest_stem}.csv"
    json_path = config.results_dir / f"{config.manifest_stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    series_counts = Counter((frame["view"], frame["slice_id"]) for frame in frames)
    payload = {
        "schema_version": 1,
        "summary": {
            "frame_count": len(rows),
            "series_count": len(series_counts),
            "frames_per_series": {
                f"{view}/{slice_id}": count
                for (view, slice_id), count in sorted(series_counts.items())
            },
        },
        "frames": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return csv_path, json_path


def _flatten_frame(frame: dict[str, Any]) -> dict[str, Any]:
    geometry = frame["geometry"]
    return {
        "dicom_path": frame["dicom_path"],
        "view": frame["view"],
        "slice_id": frame["slice_id"],
        "frame_index": frame["frame_index"],
        "timestamp_s": frame["timestamp_s"],
        "image_position_patient": json.dumps(geometry["image_position_patient"]),
        "image_orientation_patient": json.dumps(geometry["image_orientation_patient"]),
        "pixel_spacing": json.dumps(geometry["pixel_spacing"]),
        "slice_thickness": geometry["slice_thickness"],
        "spacing_between_slices": geometry["spacing_between_slices"],
        "rows": geometry["rows"],
        "columns": geometry["columns"],
    }


def main() -> None:
    """Build manifests from a YAML configuration and print the output paths."""
    parser = argparse.ArgumentParser(description="Build a PHI-free CardioResp DICOM manifest.")
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration path")
    args = parser.parse_args()
    csv_path, json_path = build_manifest(load_config(args.config))
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
