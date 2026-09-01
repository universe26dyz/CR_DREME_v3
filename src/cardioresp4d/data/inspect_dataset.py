"""
功能：扫描本地 DICOM 目录，按固定层的 AcquisitionTime 生成无 PHI 的帧元数据。
论文来源：Necessary adaptation: image-domain DICOM inspection replacing unavailable k-space metadata.
输入：只读 DICOM 根目录和视图列表（SAX、2CH、4CH）。
输出：每帧的时间、视图、固定层标识和 DICOM 平面几何；可选 JSON 检查摘要。
主要步骤：识别目录视图，读取必要 DICOM tag，转换 AcquisitionTime，并逐层时间排序。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.data.inspect_dataset --root /path/to/dicom --output results/inspection.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pydicom

from cardioresp4d.config import validate_expected_frames_per_slice, validate_output_path


_VIEWS = {"SAX", "2CH", "4CH"}
_HEADER_TAGS = [
    "AcquisitionTime", "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient",
    "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "Rows", "Columns",
]


def acquisition_time_to_seconds(value: str) -> float:
    """Convert a DICOM TM ``HHMMSS.frac`` value to seconds after midnight."""
    text = str(value).strip()
    if len(text) < 6:
        raise ValueError(f"AcquisitionTime must use HHMMSS[.ffffff], got {value!r}")
    try:
        hour = int(text[0:2])
        minute = int(text[2:4])
        second = float(text[4:])
    except ValueError as error:
        raise ValueError(f"Invalid AcquisitionTime {value!r}") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second < 60:
        raise ValueError(f"Invalid AcquisitionTime {value!r}")
    return hour * 3600.0 + minute * 60.0 + second


def scan_dicom_frames(
    root: str | Path,
    views: Iterable[str],
    expected_frames_per_slice: int = 50,
) -> list[dict[str, Any]]:
    """Return selected DICOM frames ordered by view, fixed slice, and AcquisitionTime."""
    root_path = Path(root).expanduser().resolve()
    requested_views = tuple(_normalise_requested_view(view) for view in views)
    if not requested_views:
        raise ValueError("At least one view must be requested")
    validate_expected_frames_per_slice(expected_frames_per_slice)
    if not root_path.is_dir():
        raise FileNotFoundError(f"DICOM root does not exist: {root_path}")

    records: list[dict[str, Any]] = []
    dicom_paths = sorted(
        path for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".dcm"
    )
    for dicom_path in dicom_paths:
        slice_id = dicom_path.parent.name
        view = _view_from_slice_id(slice_id)
        if view not in requested_views:
            continue
        dataset = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, specific_tags=_HEADER_TAGS)
        _require_tags(dataset, dicom_path)
        timestamp_s = acquisition_time_to_seconds(str(dataset.AcquisitionTime))
        records.append({
            "dicom_path": str(dicom_path),
            "view": view,
            "slice_id": slice_id,
            "timestamp_s": timestamp_s,
            "instance_number": int(dataset.InstanceNumber),
            "geometry": {
                "image_position_patient": [float(value) for value in dataset.ImagePositionPatient],
                "image_orientation_patient": [float(value) for value in dataset.ImageOrientationPatient],
                "pixel_spacing": [float(value) for value in dataset.PixelSpacing],
                "slice_thickness": float(dataset.SliceThickness),
                "spacing_between_slices": float(dataset.SpacingBetweenSlices),
                "rows": int(dataset.Rows),
                "columns": int(dataset.Columns),
            },
        })

    rank = {view: index for index, view in enumerate(requested_views)}
    records.sort(key=lambda frame: (rank[frame["view"]], frame["slice_id"], frame["timestamp_s"], frame["instance_number"]))
    per_slice_index: defaultdict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        key = (record["view"], record["slice_id"])
        record["frame_index"] = per_slice_index[key]
        del record["instance_number"]
        per_slice_index[key] += 1
    invalid_counts = {
        f"{view}/{slice_id}": count
        for (view, slice_id), count in sorted(per_slice_index.items())
        if count != expected_frames_per_slice
    }
    if invalid_counts:
        details = ", ".join(f"{series}={count}" for series, count in invalid_counts.items())
        raise ValueError(
            f"Selected slice frame count mismatch: expected {expected_frames_per_slice} frames; {details}"
        )
    return records


def inspect_dataset(root: str | Path, views: Iterable[str]) -> dict[str, Any]:
    """Build a serializable, PHI-free structural summary of selected DICOM frames."""
    frames = scan_dicom_frames(root, views)
    counts = Counter((frame["view"], frame["slice_id"]) for frame in frames)
    series = [
        {"view": view, "slice_id": slice_id, "frame_count": frame_count}
        for (view, slice_id), frame_count in sorted(counts.items())
    ]
    return {
        "views": list(dict.fromkeys(frame["view"] for frame in frames)),
        "frame_count": len(frames),
        "series_count": len(series),
        "series": series,
    }


def write_inspection(root: str | Path, views: Iterable[str], output_path: str | Path) -> Path:
    """Write a PHI-free inspection JSON only outside the source DICOM tree."""
    output = Path(output_path)
    validate_output_path(root, output, "output")
    summary = inspect_dataset(root, views)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return output


def _normalise_requested_view(value: str) -> str:
    normalised = str(value).upper()
    if normalised not in _VIEWS:
        raise ValueError(f"Unsupported view {value!r}; expected one of {sorted(_VIEWS)}")
    return normalised


def _view_from_slice_id(slice_id: str) -> str | None:
    upper = slice_id.upper()
    for view in _VIEWS:
        if upper.startswith(f"{view}_"):
            return view
    return None


def _require_tags(dataset: pydicom.dataset.Dataset, dicom_path: Path) -> None:
    missing = [tag for tag in _HEADER_TAGS if not hasattr(dataset, tag)]
    if missing:
        raise ValueError(f"{dicom_path} is missing required DICOM tag(s): {', '.join(missing)}")


def main() -> None:
    """Write a structural DICOM inspection JSON file."""
    parser = argparse.ArgumentParser(description="Inspect selected CardioResp DICOM views without PHI.")
    parser.add_argument("--root", required=True, type=Path, help="Read-only DICOM root directory")
    parser.add_argument("--views", nargs="+", default=["SAX", "2CH", "4CH"], help="Views to scan")
    parser.add_argument("--output", required=True, type=Path, help="Inspection JSON output path")
    args = parser.parse_args()
    output_path = write_inspection(args.root, args.views, args.output)
    with output_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
