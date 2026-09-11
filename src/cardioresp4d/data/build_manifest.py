"""功能：生成不含文件路径/患者目录名的 canonical DICOM manifest 与来源证明。
论文来源：本地图像域数据边界 necessary adaptation。
输入：经验证 AppConfig 和只读 DICOM headers。
输出：path-free CSV/JSON；另写仅位于 ignored results 的 sensitive runtime path sidecar。
主要步骤：验证完整 acquisition contract、opaque token 化、hash CSV/root/config、写 schema v2。
是否属于原论文直接实现 / 必要适配 / 可选实验：necessary adaptation。
命令行：python -m cardioresp4d.data.build_manifest --config configs/subject_local.yaml
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any

from cardioresp4d.config import AppConfig, load_config, validate_manifest_config
from cardioresp4d.data.inspect_dataset import scan_dicom_frames

MANIFEST_COLUMNS = (
    "source_file_token", "view", "slice_id", "frame_index", "timestamp_s",
    "image_position_patient", "image_orientation_patient", "pixel_spacing",
    "slice_thickness", "spacing_between_slices", "rows", "columns",
)
SCHEMA_VERSION = 2


def build_manifest(config: AppConfig) -> tuple[Path, Path]:
    """Write canonical pair plus sensitive runtime map; return only canonical paths."""
    validate_manifest_config(config)
    frames = scan_dicom_frames(config.dicom_root, config.views,
                               expected_frames_per_slice=config.expected_frames_per_slice,
                               expected_series_per_view=dict(config.data.expected_series_per_view or {}))
    rows, runtime_paths = _canonical_rows_and_paths(frames, config.views)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = config.manifest_csv_path, config.manifest_json_path
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS); writer.writeheader(); writer.writerows(rows)
    series_counts = Counter((row["view"], row["slice_id"]) for row in rows)
    root_hash = dicom_root_fingerprint(config.dicom_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {"dicom_root_fingerprint": root_hash, "config_relevant_hash": config_relevant_hash(config),
                       "csv_sha256": file_sha256(csv_path)},
        "summary": {"frame_count": len(rows), "series_count": len(series_counts),
                    "series_per_view": dict(Counter(view for view, _ in series_counts)),
                    "frames_per_series": {f"{v}/{s}": n for (v, s), n in sorted(series_counts.items())}},
        "frames": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2); handle.write("\n")
    sidecar = runtime_sidecar_path(csv_path)
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump({"sensitive": True, "dicom_root_fingerprint": root_hash, "token_to_absolute_path": runtime_paths}, handle)
    return csv_path, json_path


def validate_manifest_artifacts(config: AppConfig) -> tuple[Path, Path]:
    """Reject stale, mismatched, incomplete or wrong-root canonical manifests."""
    validate_manifest_config(config)
    csv_path, json_path = config.manifest_csv_path, config.manifest_json_path
    if not csv_path.is_file() or not json_path.is_file():
        raise FileNotFoundError("Both canonical manifest CSV and JSON are required")
    with json_path.open(encoding="utf-8") as handle: payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Manifest schema mismatch")
    provenance = payload.get("provenance", {})
    if provenance.get("csv_sha256") != file_sha256(csv_path): raise ValueError("Manifest CSV/JSON hash mismatch")
    if provenance.get("dicom_root_fingerprint") != dicom_root_fingerprint(config.dicom_root): raise ValueError("Manifest DICOM root fingerprint mismatch")
    if provenance.get("config_relevant_hash") != config_relevant_hash(config): raise ValueError("Manifest config hash mismatch")
    with csv_path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    if len(rows) != payload.get("summary", {}).get("frame_count") or len(rows) != len(payload.get("frames", [])):
        raise ValueError("Manifest CSV/JSON content count mismatch")
    if _string_rows(rows) != _string_rows(payload.get("frames", [])):
        raise ValueError("Manifest CSV/JSON content mismatch")
    groups = Counter((row["view"], row["slice_id"]) for row in rows)
    expected_series = dict(config.data.expected_series_per_view or {})
    actual_series = Counter(view for view, _ in groups)
    if any(actual_series.get(view, 0) != count for view, count in expected_series.items()):
        raise ValueError("Manifest fixed-slice series contract mismatch")
    if any(count != config.expected_frames_per_slice for count in groups.values()):
        raise ValueError("Manifest complete 50-frame group contract mismatch")
    frames = scan_dicom_frames(config.dicom_root, config.views,
                               expected_frames_per_slice=config.expected_frames_per_slice,
                               expected_series_per_view=expected_series)
    expected_rows, expected_paths = _canonical_rows_and_paths(frames, config.views)
    if _string_rows(rows) != _string_rows(expected_rows):
        raise ValueError("Manifest content no longer matches the current DICOM acquisition")
    sidecar = runtime_sidecar_path(csv_path)
    if not sidecar.is_file(): raise FileNotFoundError("Sensitive runtime path sidecar is missing")
    with sidecar.open(encoding="utf-8") as handle: runtime = json.load(handle)
    if runtime.get("dicom_root_fingerprint") != provenance.get("dicom_root_fingerprint"):
        raise ValueError("Runtime path sidecar root fingerprint mismatch")
    if set(runtime.get("token_to_absolute_path", {})) != {row["source_file_token"] for row in rows}:
        raise ValueError("Runtime path sidecar token mismatch")
    if runtime.get("token_to_absolute_path") != expected_paths:
        raise ValueError("Runtime path sidecar content mismatch")
    return csv_path, json_path


def runtime_sidecar_path(manifest_csv: str | Path) -> Path:
    path = Path(manifest_csv)
    return path.with_name(f"{path.stem}.sensitive_runtime_paths.json")


def dicom_root_fingerprint(root: str | Path) -> str:
    root_path = Path(root).expanduser().resolve()
    digest = hashlib.sha256()
    for path in sorted(p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"):
        stat = path.stat(); digest.update(str(path.relative_to(root_path)).encode()); digest.update(str(stat.st_size).encode())
    return digest.hexdigest()


def config_relevant_hash(config: AppConfig) -> str:
    payload = {"project_name": config.project.name, "views": list(config.views),
               "expected_frames_per_slice": config.expected_frames_per_slice,
               "expected_series_per_view": dict(config.data.expected_series_per_view or {}),
               "manifest_stem": config.manifest_stem,
               "geometry": {"required_views": list(config.geometry.required_views),
                            "max_qc_planes_per_view": config.geometry.max_qc_planes_per_view,
                            "pixel_tolerance": config.geometry.pixel_round_trip_tolerance,
                            "world_tolerance_mm": config.geometry.world_round_trip_tolerance_mm},
               "frequency": {"respiratory_band_hz": list(config.frequency.respiratory_band_hz),
                             "cardiac_band_hz": list(config.frequency.cardiac_band_hz),
                             "dominance_threshold": config.frequency.dominance_threshold,
                             "uniform_relative_tolerance": config.frequency.uniform_relative_tolerance,
                             "uniform_absolute_tolerance_s": config.frequency.uniform_absolute_tolerance_s,
                             "pca_components": config.frequency.pca_components,
                             "consensus_min_slice_fraction": config.frequency.consensus_min_slice_fraction},
               "roi_hash": hashlib.sha256(json.dumps({"center": config.roi.cardiac_box_center_mm,
                                                       "size": config.roi.cardiac_box_size_mm}, sort_keys=True).encode()).hexdigest(),
               "reference": {"source_view": config.reference.source_view,
                             "orientation_tolerance": config.reference.orientation_tolerance,
                             "spacing_tolerance_mm": config.reference.spacing_tolerance_mm,
                             "duplicate_tolerance_mm": config.reference.duplicate_tolerance_mm,
                             "origin_affine_tolerance_mm": config.reference.origin_affine_tolerance_mm},
               "outlier_qc": {"ncc_mad_threshold": config.outlier_qc.ncc_mad_threshold,
                              "scale_mad_threshold": config.outlier_qc.scale_mad_threshold,
                              "residual_mad_threshold": config.outlier_qc.residual_mad_threshold,
                              "mad_floor": config.outlier_qc.mad_floor}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _flatten_frame(frame: dict[str, Any], token: str, slice_id: str) -> dict[str, Any]:
    geometry = frame["geometry"]
    return {"source_file_token": token, "view": frame["view"], "slice_id": slice_id,
            "frame_index": frame["frame_index"], "timestamp_s": frame["timestamp_s"],
            "image_position_patient": json.dumps(geometry["image_position_patient"]),
            "image_orientation_patient": json.dumps(geometry["image_orientation_patient"]),
            "pixel_spacing": json.dumps(geometry["pixel_spacing"]), "slice_thickness": geometry["slice_thickness"],
            "spacing_between_slices": geometry["spacing_between_slices"], "rows": geometry["rows"], "columns": geometry["columns"]}


def _canonical_rows_and_paths(
    frames: list[dict[str, Any]], views: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Deterministically derive canonical rows and the ignored runtime path map."""
    slice_keys = sorted({(frame["view"], frame["slice_id"]) for frame in frames})
    opaque: dict[tuple[str, str], str] = {}
    for view in views:
        for index, key in enumerate(key for key in slice_keys if key[0] == view):
            opaque[key] = f"{view}_s{index + 1:03d}"
    runtime_paths: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        token = f"frame_{index:06d}"
        runtime_paths[token] = frame["dicom_path"]
        rows.append(_flatten_frame(frame, token, opaque[(frame["view"], frame["slice_id"])]))
    return rows, runtime_paths


def _string_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize CSV strings and JSON scalar types for exact pair comparison."""
    return [{key: str(value) for key, value in row.items()} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a path-free CardioResp DICOM manifest.")
    parser.add_argument("--config", required=True, type=Path); args = parser.parse_args()
    csv_path, json_path = build_manifest(load_config(args.config)); print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__": main()
