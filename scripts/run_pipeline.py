#!/usr/bin/env python3
"""功能：按依赖顺序运行仅 Phase 1 的六个公开 API 阶段。
论文来源：方法设计文档的工程编排，属于 necessary adaptation。
输入：模块化 YAML，以及可选 from-stage/to-stage。
输出：inspection/manifest/geometry/frequency/ROI/reference 工件摘要。
主要步骤：验证阶段范围与依赖，再调用既有模块公开 API；不复制科学逻辑。
是否属于原论文直接实现 / 必要适配 / 可选实验：necessary adaptation。
命令行：python scripts/run_pipeline.py --config configs/subject_local.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cardioresp4d.config import AppConfig, load_config  # noqa: E402
from cardioresp4d.data.build_manifest import build_manifest  # noqa: E402
from cardioresp4d.data.inspect_dataset import write_inspection  # noqa: E402
from cardioresp4d.frequency.pca_motion import analyze_manifest  # noqa: E402
from cardioresp4d.geometry.geometry_qc import run_geometry_qc  # noqa: E402
from cardioresp4d.reference.build_initial_reference import build_initial_reference  # noqa: E402
from cardioresp4d.roi.cardiac_box import CardiacBox  # noqa: E402
from cardioresp4d.roi.roi_qc import run_roi_qc  # noqa: E402

STAGES = ("inspect", "manifest", "geometry", "frequency", "roi", "reference")


def selected_stages(from_stage: str | None = None, to_stage: str | None = None) -> tuple[str, ...]:
    """Resolve an inclusive dependency-ordered stage interval."""
    start = STAGES.index(from_stage) if from_stage is not None else 0
    end = STAGES.index(to_stage) if to_stage is not None else len(STAGES) - 1
    if start > end:
        raise ValueError(f"Invalid stage order: from-stage {STAGES[start]} follows to-stage {STAGES[end]}")
    return STAGES[start:end + 1]


def run_pipeline(config: AppConfig, from_stage: str | None = None,
                 to_stage: str | None = None) -> dict[str, Any]:
    """Execute an inclusive Phase-1 interval and return artifact summaries."""
    stages = selected_stages(from_stage, to_stage)
    manifest = config.manifest_csv_path
    if stages[0] not in ("inspect", "manifest") and not manifest.is_file():
        raise FileNotFoundError(f"Required manifest artifact is missing: {manifest}; run from stage 'manifest' or earlier")
    results: dict[str, Any] = {}
    if "inspect" in stages:
        path = write_inspection(config.dicom_root, config.views, config.inspection_path)
        results["inspect"] = {"inspection": str(path)}
    if "manifest" in stages:
        csv_path, json_path = build_manifest(config)
        manifest = csv_path
        results["manifest"] = {"csv": str(csv_path), "json": str(json_path)}
    if any(stage in stages for stage in STAGES[2:]) and not manifest.is_file():
        raise FileNotFoundError(f"Manifest stage did not produce required artifact: {manifest}")
    if "geometry" in stages:
        report, image = run_geometry_qc(manifest, config.results_dir / config.geometry.output_subdir,
                                         max_per_view=config.geometry.max_qc_planes_per_view)
        _validate_geometry_tolerances(report, config)
        results["geometry"] = {"report": str(report), "image": str(image)}
    if "frequency" in stages:
        bands = analyze_manifest(manifest, config.results_dir / config.frequency.output_subdir,
                                 relative_tolerance=config.frequency.uniform_relative_tolerance)
        results["frequency"] = {"bands": str(config.results_dir / config.frequency.output_subdir / "frequency_bands.json"),
                                "slice_count": bands.get("slice_count")}
    if "roi" in stages:
        if config.roi.cardiac_box_center_mm is None or config.roi.cardiac_box_size_mm is None:
            raise ValueError("ROI stage requires roi.cardiac_box_center_mm and roi.cardiac_box_size_mm")
        report, images = run_roi_qc(manifest, config.results_dir / config.roi.output_subdir,
                                     CardiacBox(config.roi.cardiac_box_center_mm, config.roi.cardiac_box_size_mm),
                                     config.expected_frames_per_slice)
        results["roi"] = {"report": str(report), "images": {k: str(v) for k, v in images.items()}}
    if "reference" in stages:
        nifti, metadata, qc = build_initial_reference(
            manifest, config.results_dir / config.reference.output_subdir, config.expected_frames_per_slice,
            orientation_tolerance=config.reference.orientation_tolerance,
            spacing_tolerance_mm=config.reference.spacing_tolerance_mm,
            duplicate_tolerance_mm=config.reference.duplicate_tolerance_mm,
            origin_tolerance_mm=config.reference.origin_affine_tolerance_mm)
        results["reference"] = {"nifti": str(nifti), "metadata": str(metadata), "qc": str(qc)}
    return results


def _validate_geometry_tolerances(report_path: str | Path, config: AppConfig) -> None:
    """Enforce configured acceptance thresholds on the geometry API report."""
    with Path(report_path).open(encoding="utf-8") as handle:
        errors = json.load(handle)["round_trip_errors"]
    pixel = float(errors["pixel_to_world_to_pixel_max_error_pixels"])
    world = float(errors["world_to_pixel_to_world_max_error_mm"])
    if pixel > config.geometry.pixel_round_trip_tolerance:
        raise RuntimeError(f"Geometry pixel round-trip error {pixel} exceeds configured tolerance")
    if world > config.geometry.world_round_trip_tolerance_mm:
        raise RuntimeError(f"Geometry world round-trip error {world} mm exceeds configured tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete pre-network CardioResp Phase-1 pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--from-stage", choices=STAGES)
    parser.add_argument("--to-stage", choices=STAGES)
    args = parser.parse_args()
    try:
        summary = run_pipeline(load_config(args.config), args.from_stage, args.to_stage)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
