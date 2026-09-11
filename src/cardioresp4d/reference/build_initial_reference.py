"""
功能：构建 SAX temporal-average anisotropic initial reference volume。
论文来源：S2V-DREME Stage I direct principle：每个固定 SAX location 的 50 frames 取均值，再按物理位置堆叠。
输入：Task 1 无 PHI CSV manifest，其中的 SAX DICOM 帧及 patient-world geometry。
输出：initial_reference.nii.gz、initial_reference.json 和 reference_qc.png。
主要步骤：严格验证 50 帧/层与平行等距几何，时间平均，沿统一 SAX normal 排序，以 (column,row,slice) 保存。
是否属于原论文直接实现 / 必要适配 / 可选实验：S2V-DREME direct averaging/stacking；DICOM LPS 仿射、NIfTI RAS+ 转换及 QC 为 necessary adaptation；不做 multi-view fusion。
命令行使用示例：python -m cardioresp4d.reference.build_initial_reference --manifest results/dicom_manifest.csv --output-dir results/reference
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from cardioresp4d.config import validate_expected_frames_per_slice
from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.geometry.world_geometry import DicomPlane


LPS_TO_RAS = np.diag((-1.0, -1.0, 1.0, 1.0))


def build_initial_reference(
    manifest_path: str | Path,
    output_dir: str | Path,
    expected_frames_per_slice: int = 50,
    *,
    orientation_tolerance: float = 1e-5,
    spacing_tolerance_mm: float = 1e-6,
    duplicate_tolerance_mm: float = 1e-3,
    origin_tolerance_mm: float = 0.5,
) -> tuple[Path, Path, Path]:
    """Average each fixed SAX slice and save a physically ordered NIfTI reference.

    The volume array is explicitly ``(column, row, slice)``. DICOM pixel arrays
    arrive as ``(row, column)`` and are transposed only after temporal averaging.
    Input intensities follow the existing loader boundary: modality rescale when
    present, per-frame 1st/99th-percentile normalization, and clipping to [0, 1].
    """
    validate_expected_frames_per_slice(expected_frames_per_slice)
    _validate_positive_tolerance(orientation_tolerance, "orientation_tolerance")
    _validate_positive_tolerance(spacing_tolerance_mm, "spacing_tolerance_mm")
    _validate_positive_tolerance(duplicate_tolerance_mm, "duplicate_tolerance_mm")
    _validate_positive_tolerance(origin_tolerance_mm, "origin_tolerance_mm")

    manifest = Path(manifest_path)
    # Keep every acquired SAX location in the physical stack.  QC only decides
    # whether frames provide supervision, never whether their location exists.
    # This retained legacy/ablation utility preserves its historical
    # frame-wise reference semantics; source-first mainline uses per-series
    # normalization in ``scripts/train_source_first.py`` instead.
    dataset = CardioRespDataset(manifest, valid_only=False, normalization_mode="per_frame_legacy")
    rows = dataset._rows
    with manifest.open(newline="", encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    original_sax_ids = {row["slice_id"] for row in original_rows if row.get("view", "").upper() == "SAX"}
    sax_thicknesses = {float(row["slice_thickness"]) for row in original_rows if row.get("view", "").upper() == "SAX"}
    if len(sax_thicknesses) != 1:
        raise ValueError("SAX DICOM SliceThickness must be constant for the current renderer contract")
    acquisition_slice_thickness_mm = sax_thicknesses.pop()
    sax_rows = [(index, row) for index, row in enumerate(rows) if row.get("view", "").upper() == "SAX"]
    if not sax_rows:
        raise ValueError(f"Manifest contains no SAX frames: {manifest}")

    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in sax_rows:
        grouped[row["slice_id"]].append((index, row))

    slice_records: list[dict[str, Any]] = []
    for slice_id, indexed_rows in grouped.items():
        ordered = sorted(indexed_rows, key=lambda item: int(item[1]["frame_index"]))
        frame_indices = [int(row["frame_index"]) for _, row in ordered]
        complete = len(ordered) == expected_frames_per_slice and frame_indices == list(range(expected_frames_per_slice))
        if not complete:
            raise ValueError(f"SAX/{slice_id} must contain exactly {expected_frames_per_slice} acquired frames; got {len(ordered)}")
        planes = [DicomPlane.from_geometry(row) for _, row in ordered]
        _validate_within_slice_geometry(slice_id, planes, orientation_tolerance, spacing_tolerance_mm)
        slice_records.append({
            "slice_id": slice_id,
            "indices": [index for index, _ in ordered],
            "valid_indices": [index for index, row in ordered if row["qc_valid"] not in ("0", "false", "False")],
            "frame_indices": frame_indices,
            "valid_frame_count": sum(row["qc_valid"] not in ("0", "false", "False") for _, row in ordered),
            "acquired_frame_count": len(ordered),
            "plane": planes[0],
        })

    if len(slice_records) < 2:
        raise ValueError("At least two distinct SAX slice locations are required to define a 3D reference affine")
    reference_plane = slice_records[0]["plane"]
    for record in slice_records[1:]:
        _validate_cross_slice_geometry(
            reference_plane,
            record["plane"],
            record["slice_id"],
            orientation_tolerance,
            spacing_tolerance_mm,
        )

    unit_normal = reference_plane.normal
    for record in slice_records:
        record["position_mm"] = float(np.dot(record["plane"].origin, unit_normal))
    slice_records.sort(key=lambda record: (record["position_mm"], record["slice_id"]))
    positions = np.asarray([record["position_mm"] for record in slice_records], dtype=np.float64)
    position_steps = np.diff(positions)
    if np.any(position_steps <= duplicate_tolerance_mm):
        raise ValueError("SAX manifest contains a duplicate physical slice location along the representative normal")

    median_spacing = float(np.median(position_steps))
    slice_step = unit_normal * median_spacing
    first_origin = slice_records[0]["plane"].origin
    predicted_origins = first_origin + np.arange(len(slice_records))[:, None] * slice_step
    actual_origins = np.stack([record["plane"].origin for record in slice_records])
    origin_residuals = np.linalg.norm(actual_origins - predicted_origins, axis=1)
    max_origin_residual = float(origin_residuals.max())
    if max_origin_residual > origin_tolerance_mm:
        raise ValueError(
            f"SAX origins exceed affine residual tolerance: max {max_origin_residual:.6g} mm > "
            f"{origin_tolerance_mm:.6g} mm"
        )

    dicom_lps_affine = np.eye(4, dtype=np.float64)
    dicom_lps_affine[:3, 0] = reference_plane.column_step
    dicom_lps_affine[:3, 1] = reference_plane.row_step
    dicom_lps_affine[:3, 2] = slice_step
    dicom_lps_affine[:3, 3] = first_origin
    nifti_ras_affine = LPS_TO_RAS @ dicom_lps_affine

    temporal_means: list[np.ndarray | None] = []
    valid_location_flags: list[bool] = []
    rescale_status_counts: dict[str, int] = defaultdict(int)
    for record in slice_records:
        frames = []
        for index in record["valid_indices"]:
            sample = dataset[index]
            frames.append(sample["image"])
            rescale_status_counts[sample["rescale_status"]] += 1
        valid_location_flags.append(bool(frames))
        temporal_means.append(None if not frames else np.mean(np.stack(frames), axis=0, dtype=np.float64).T.astype(np.float32))
    temporal_means, placeholder_details = _fill_missing_location_placeholders(temporal_means, positions)
    volume = np.stack(temporal_means, axis=2)
    valid_mask = np.broadcast_to(np.asarray(valid_location_flags, dtype=np.float32)[None, None, :], volume.shape).copy()
    expected_shape = (reference_plane.columns, reference_plane.rows, len(slice_records))
    if volume.shape != expected_shape or not np.isfinite(volume).all():
        raise RuntimeError(f"Invalid reference volume shape/finiteness: {volume.shape}, expected {expected_shape}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    nifti_path = output / "initial_reference.nii.gz"
    valid_mask_path = output / "initial_reference_valid_mask.nii.gz"
    metadata_path = output / "initial_reference.json"
    qc_path = output / "reference_qc.png"
    image = nib.Nifti1Image(volume, nifti_ras_affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, nifti_path)
    mask_image = nib.Nifti1Image(valid_mask, nifti_ras_affine)
    mask_image.header.set_xyzt_units("mm")
    nib.save(mask_image, valid_mask_path)
    reopened = nib.load(nifti_path)
    reopened_values = np.asanyarray(reopened.dataobj)
    if reopened.shape != volume.shape or not np.isfinite(reopened_values).all():
        raise RuntimeError("Reopened initial-reference NIfTI failed shape/finiteness validation")
    if not np.allclose(reopened.affine, nifti_ras_affine, atol=1e-5, rtol=1e-7):
        raise RuntimeError("Reopened initial-reference NIfTI affine differs from the requested RAS+ affine")
    reopened_mask = nib.load(valid_mask_path)
    if reopened_mask.shape != volume.shape or not np.allclose(reopened_mask.affine, nifti_ras_affine, atol=1e-5, rtol=1e-7):
        raise RuntimeError("Initial-reference valid mask must share the reference shape and RAS+ affine")

    metadata = {
        "schema_version": 1,
        "method": "S2V-DREME Stage I: 50-frame temporal average at each fixed SAX location, then physical stack",
        "necessary_adaptation": "DICOM patient LPS geometry is explicitly converted to the NIfTI RAS+ world convention; no multi-view fusion",
        "manifest": str(manifest),
        "source_view": "SAX",
        "array_order": "(column,row,slice)",
        "shape": list(volume.shape),
        "voxel_spacing_mm": [
            float(np.linalg.norm(dicom_lps_affine[:3, 0])),
            float(np.linalg.norm(dicom_lps_affine[:3, 1])),
            float(np.linalg.norm(dicom_lps_affine[:3, 2])),
        ],
        "acquisition_slice_thickness_mm": acquisition_slice_thickness_mm,
        "stack_slice_spacing_semantics": "IPP centre-to-centre spacing for stack/world geometry",
        "acquisition_slice_thickness_semantics": "DICOM SliceThickness retained for the SIMPLE-4D thick-slice renderer; it is not replaced by IPP stack spacing",
        "dicom_lps_affine": dicom_lps_affine.tolist(),
        "nifti_ras_affine": nifti_ras_affine.tolist(),
        "coordinate_conversion": "nifti_ras_affine = diag(-1,-1,1,1) @ dicom_lps_affine",
        "representative_sax_unit_normal_lps": unit_normal.tolist(),
        "sorted_slice_ids": [record["slice_id"] for record in slice_records],
        "sorted_slice_positions_along_normal_mm": positions.tolist(),
        "sorted_slice_origins_lps_mm": actual_origins.tolist(),
        "frames_per_slice": [record["valid_frame_count"] for record in slice_records],
        "acquired_frames_per_slice": [record["acquired_frame_count"] for record in slice_records],
        "missing_slice_ids": [record["slice_id"] for record, valid in zip(slice_records, valid_location_flags) if not valid],
        "missing_slice_positions_mm": [record["position_mm"] for record, valid in zip(slice_records, valid_location_flags) if not valid],
        "missing_slice_positions_semantics": "physical position projected onto representative_sax_unit_normal_lps",
        "missing_slice_origins_lps_mm": [record["plane"].origin.tolist() for record, valid in zip(slice_records, valid_location_flags) if not valid],
        "missing_slice_count": int(sum(not valid for valid in valid_location_flags)),
        "placeholder_method": "linear interpolation between bracketing valid SAX temporal means; nearest valid temporal mean at a stack edge",
        "placeholder_details": placeholder_details,
        "initial_reference_valid_mask": str(valid_mask_path),
        "supervision_contract": "mask_zero_means_no_stage1a_reference_supervision",
        "temporal_averaging_frames": "arithmetic mean over qc_valid frames only",
        "normalization": {
            "input": "existing CardioRespDataset per-frame normalization",
            "modality_transform": "pydicom modality LUT when both rescale tags exist; otherwise stored pixels",
            "percentiles": [1.0, 99.0],
            "clip_range": [0.0, 1.0],
            "rescale_status_frame_counts": dict(sorted(rescale_status_counts.items())),
            "output_intensity_min": float(volume.min()),
            "output_intensity_max": float(volume.max()),
            "output_intensity_mean": float(volume.mean(dtype=np.float64)),
        },
        "geometry_validation": {
            "orientation_tolerance": orientation_tolerance,
            "spacing_tolerance_mm": spacing_tolerance_mm,
            "duplicate_tolerance_mm": duplicate_tolerance_mm,
            "origin_affine_tolerance_mm": origin_tolerance_mm,
            "median_slice_spacing_mm": median_spacing,
            "slice_position_steps_mm": position_steps.tolist(),
            "origin_affine_residuals_mm": origin_residuals.tolist(),
            "max_origin_affine_residual_mm": max_origin_residual,
        },
        "saved_nifti_validation": {
            "reopened_shape": list(reopened.shape),
            "all_finite": bool(np.isfinite(reopened_values).all()),
            "affine_matches": True,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    _render_reference_qc(volume, metadata["voxel_spacing_mm"], qc_path)
    return nifti_path, metadata_path, qc_path


def _validate_within_slice_geometry(
    slice_id: str,
    planes: list[DicomPlane],
    orientation_tolerance: float,
    spacing_tolerance_mm: float,
) -> None:
    """Require every temporal frame at one location to share one plane geometry."""
    reference = planes[0]
    for plane in planes[1:]:
        if plane.rows != reference.rows or plane.columns != reference.columns:
            raise ValueError(f"SAX/{slice_id} has inconsistent in-plane matrix shape across frames")
        if not np.allclose(plane.pixel_spacing, reference.pixel_spacing, atol=spacing_tolerance_mm, rtol=0.0):
            raise ValueError(f"SAX/{slice_id} has inconsistent pixel spacing across frames")
        if not np.allclose(plane.origin, reference.origin, atol=spacing_tolerance_mm, rtol=0.0):
            raise ValueError(f"SAX/{slice_id} has inconsistent image position across frames")
        if not np.allclose(plane.row_direction, reference.row_direction, atol=orientation_tolerance, rtol=0.0) or not np.allclose(
            plane.column_direction, reference.column_direction, atol=orientation_tolerance, rtol=0.0
        ):
            raise ValueError(f"SAX/{slice_id} has inconsistent orientation across frames")


def _fill_missing_location_placeholders(
    temporal_means: list[np.ndarray | None], positions_mm: np.ndarray
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Fill storage-only holes without changing their zero-supervision status.

    Locations are already sorted by physical normal projection.  Interpolation
    is done in physical position, not slice index, so an irregular stack cannot
    silently change the placeholder weighting or the reference affine.
    """
    valid = [index for index, image in enumerate(temporal_means) if image is not None]
    if not valid:
        raise ValueError("No qc_valid SAX location remains to form a finite reference placeholder")
    output = list(temporal_means)
    details: list[dict[str, Any]] = []
    for index, image in enumerate(output):
        if image is not None:
            continue
        lower = max((candidate for candidate in valid if candidate < index), default=None)
        upper = min((candidate for candidate in valid if candidate > index), default=None)
        if lower is not None and upper is not None:
            weight = float((positions_mm[index] - positions_mm[lower]) / (positions_mm[upper] - positions_mm[lower]))
            output[index] = ((1.0 - weight) * output[lower] + weight * output[upper]).astype(np.float32)
            details.append({"slice_index": index, "method": "linear_bracketing_valid_temporal_means", "lower_valid_index": lower, "upper_valid_index": upper, "physical_weight": weight})
        else:
            neighbour = lower if lower is not None else upper
            assert neighbour is not None
            output[index] = output[neighbour].copy()
            details.append({"slice_index": index, "method": "nearest_valid_temporal_mean_edge_fallback", "nearest_valid_index": neighbour})
    if any(image is None or not np.isfinite(image).all() for image in output):
        raise RuntimeError("Reference placeholder filling failed to produce finite storage values")
    return [image for image in output if image is not None], details


def _validate_cross_slice_geometry(
    reference: DicomPlane,
    candidate: DicomPlane,
    slice_id: str,
    orientation_tolerance: float,
    spacing_tolerance_mm: float,
) -> None:
    """Require a single parallel SAX stack with a shared in-plane sampling grid."""
    if candidate.rows != reference.rows or candidate.columns != reference.columns:
        raise ValueError(f"SAX/{slice_id} has inconsistent in-plane matrix shape")
    if not np.allclose(candidate.pixel_spacing, reference.pixel_spacing, atol=spacing_tolerance_mm, rtol=0.0):
        raise ValueError(f"SAX/{slice_id} has inconsistent pixel spacing")
    if not np.allclose(candidate.row_direction, reference.row_direction, atol=orientation_tolerance, rtol=0.0) or not np.allclose(
        candidate.column_direction, reference.column_direction, atol=orientation_tolerance, rtol=0.0
    ):
        raise ValueError(f"SAX/{slice_id} orientation is not parallel and consistently directed")


def _validate_positive_tolerance(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")


def _render_reference_qc(volume: np.ndarray, spacing_mm: list[float], output_path: Path) -> None:
    """Render three central orthogonal slices for lightweight visual validation."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    column_mid, row_mid, slice_mid = (size // 2 for size in volume.shape)
    panels = (
        (volume[:, :, slice_mid].T, "SAX stack plane", spacing_mm[1] / spacing_mm[0]),
        (volume[:, row_mid, :].T, "column-slice plane", spacing_mm[2] / spacing_mm[0]),
        (volume[column_mid, :, :].T, "row-slice plane", spacing_mm[2] / spacing_mm[1]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, (panel, title, aspect) in zip(axes, panels):
        axis.imshow(panel, cmap="gray", origin="lower", interpolation="nearest", aspect=aspect)
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle("SAX 50-frame temporal-average initial reference (no multi-view fusion)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Build the Phase-1 SAX temporal-average initial reference from a manifest."""
    parser = argparse.ArgumentParser(description="Build a physically ordered SAX temporal-average initial reference.")
    parser.add_argument("--manifest", required=True, type=Path, help="Task 1 CSV manifest")
    parser.add_argument("--output-dir", required=True, type=Path, help="Ignored reference output directory")
    parser.add_argument("--expected-frames-per-slice", type=int, default=50)
    parser.add_argument("--origin-tolerance-mm", type=float, default=0.5)
    args = parser.parse_args()
    nifti, metadata, qc = build_initial_reference(
        args.manifest,
        args.output_dir,
        args.expected_frames_per_slice,
        origin_tolerance_mm=args.origin_tolerance_mm,
    )
    print(json.dumps({"nifti": str(nifti), "metadata": str(metadata), "qc": str(qc)}, indent=2))


if __name__ == "__main__":
    main()
