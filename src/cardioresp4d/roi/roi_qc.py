"""
功能：把同一个 DREME-style local cardiac 3D box 投影并截交到 SAX/2CH/4CH temporal-mean 图像。
论文来源：DREME-MR local cardiac box；NISF++ patient-world geometry 属于 necessary adaptation。
输入：Task 1 manifest、明确配置的 CardiacBox、50 frames/slice 和结果目录。
输出：roi_qc.json 与 qc_sax.png/qc_2ch.png/qc_4ch.png。
主要步骤：按 box center 到平面的绝对物理距离选层，平均该层 50 帧，以 12 条盒边求真实平面截交并审计八角投影。
是否属于原论文直接实现 / 必要适配 / 可选实验：DREME-style direct concept + necessary multi-view geometry adaptation；非 segmentation。
命令行使用示例：MPLCONFIGDIR=/tmp/cardioresp-mpl python -m cardioresp4d.roi.roi_qc --manifest results/dicom_manifest.csv --output-dir results/roi_qc --config configs/subject_local.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cardioresp4d.config import validate_expected_frames_per_slice
from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.geometry.coordinate_normalization import WorldNormalizer
from cardioresp4d.geometry.geometry_qc import load_manifest_planes
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.roi.cardiac_box import CardiacBox, load_cardiac_box_config


REQUIRED_VIEWS = ("SAX", "2CH", "4CH")
BOX_EDGES = tuple(
    (index, index ^ bit)
    for index in range(8)
    for bit in (1, 2, 4)
    if index < (index ^ bit)
)


def plane_distance_mm(point_mm: Any, plane: DicomPlane) -> float:
    """Return absolute orthogonal distance from one world point to a plane."""
    point = np.asarray(point_mm, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_mm must be a finite three-dimensional vector")
    return abs(float(np.dot(point - plane.origin, plane.normal)))


def choose_closest_planes(
    planes: Iterable[tuple[dict[str, str], DicomPlane]],
    point_mm: Any,
    required_views: Sequence[str] = REQUIRED_VIEWS,
) -> dict[str, tuple[dict[str, str], DicomPlane, float]]:
    """Choose each view's plane nearest a point by absolute physical distance."""
    grouped: dict[str, list[tuple[dict[str, str], DicomPlane]]] = defaultdict(list)
    for row, plane in planes:
        grouped[row["view"].upper()].append((row, plane))
    required = tuple(view.upper() for view in required_views)
    missing = [view for view in required if view not in grouped]
    if missing:
        raise ValueError(f"Manifest is missing {', '.join(missing)} required view(s)")
    chosen: dict[str, tuple[dict[str, str], DicomPlane, float]] = {}
    for view in required:
        candidates = [
            (plane_distance_mm(point_mm, plane), row["slice_id"], row, plane)
            for row, plane in grouped[view]
        ]
        distance, _, row, plane = min(candidates, key=lambda item: (item[0], item[1]))
        chosen[view] = (row, plane, float(distance))
    return chosen


def project_box_to_plane(box: CardiacBox, plane: DicomPlane) -> dict[str, np.ndarray]:
    """Project the same eight corners and retain signed off-plane distances."""
    corners = box.corners_mm
    return {
        "world_corners_mm": corners,
        "pixel_corners": plane.world_to_pixel(corners),
        "off_plane_distance_mm": (corners - plane.origin) @ plane.normal,
    }


def box_plane_intersection(box: CardiacBox, plane: DicomPlane, tolerance_mm: float = 1e-7) -> dict[str, np.ndarray]:
    """Intersect a plane with all 12 box edges and order the true polygon."""
    if tolerance_mm <= 0.0 or not np.isfinite(tolerance_mm):
        raise ValueError("tolerance_mm must be a finite positive scalar")
    corners = box.corners_mm
    signed = (corners - plane.origin) @ plane.normal
    points: list[np.ndarray] = []
    for first_index, second_index in BOX_EDGES:
        first, second = corners[first_index], corners[second_index]
        first_distance, second_distance = signed[first_index], signed[second_index]
        first_on = abs(first_distance) <= tolerance_mm
        second_on = abs(second_distance) <= tolerance_mm
        if first_on:
            points.append(first)
        if second_on:
            points.append(second)
        if not first_on and not second_on and first_distance * second_distance < 0.0:
            fraction = first_distance / (first_distance - second_distance)
            points.append(first + fraction * (second - first))
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance_mm for existing in unique):
            unique.append(point)
    if not unique:
        return {"world_mm": np.empty((0, 3)), "pixel": np.empty((0, 2))}
    world = np.asarray(unique, dtype=np.float64)
    pixels = plane.world_to_pixel(world)
    centre = pixels.mean(axis=0)
    order = np.argsort(np.arctan2(pixels[:, 1] - centre[1], pixels[:, 0] - centre[0]))
    return {"world_mm": world[order], "pixel": pixels[order]}


def run_roi_qc(
    manifest_path: str | Path,
    output_dir: str | Path,
    box: CardiacBox,
    expected_frames_per_slice: int = 50,
) -> tuple[Path, dict[str, Path]]:
    """Generate one shared-box audit report and three temporal-mean overlays."""
    validate_expected_frames_per_slice(expected_frames_per_slice)
    manifest = Path(manifest_path)
    dataset = CardioRespDataset(manifest)
    rows = dataset._rows
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest}")
    plane_records = load_manifest_planes(manifest)
    normalizer = WorldNormalizer.from_planes([plane for _, plane in plane_records])
    box.validate_within(normalizer)
    selected = choose_closest_planes(plane_records, box.center_mm)
    selected_keys = {(view, row["slice_id"]) for view, (row, _, _) in selected.items()}
    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["view"].upper(), row["slice_id"])
        if key in selected_keys:
            grouped_indices[key].append(index)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "DREME-style independent local cardiac coordinate box with patient-world multi-view QC; not segmentation",
        "manifest": str(manifest),
        "expected_frames_per_slice": int(expected_frames_per_slice),
        "subject_world_normalizer": normalizer.to_dict(),
        "cardiac_box": box.to_dict(normalizer),
        "views": {},
    }
    image_paths: dict[str, Path] = {}
    for view in REQUIRED_VIEWS:
        row, plane, distance = selected[view]
        key = (view, row["slice_id"])
        indices = sorted(grouped_indices[key], key=lambda index: int(rows[index]["frame_index"]))
        frame_indices = [int(rows[index]["frame_index"]) for index in indices]
        if not indices or len(indices) > expected_frames_per_slice or len(set(frame_indices)) != len(frame_indices):
            raise ValueError(
                f"{view}/{row['slice_id']} has no usable unique qc_valid frames"
            )
        temporal_mean = np.mean(np.stack([dataset[index]["image"] for index in indices]), axis=0, dtype=np.float64)
        projection = project_box_to_plane(box, plane)
        intersection = box_plane_intersection(box, plane)
        projected_inside = [plane.contains_pixel(*pixel) for pixel in projection["pixel_corners"]]
        intersection_inside = [plane.contains_pixel(*pixel) for pixel in intersection["pixel"]]
        center_pixel = plane.world_to_pixel(box.center_mm)
        report["views"][view] = {
            "slice_id": row["slice_id"],
            "selection_rule": "minimum absolute patient-world distance from cardiac-box center to plane",
            "plane_distance_to_box_center_mm": distance,
            "frame_count": len(indices),
            "frame_index_min": min(frame_indices),
            "frame_index_max": max(frame_indices),
            "plane_geometry": plane.to_dict(),
            "box_center_pixel_column_row": center_pixel.tolist(),
            "box_center_projection_in_image": bool(plane.contains_pixel(*center_pixel)),
            "corner_projection": {
                "interpretation": "orthogonal projections of the same 8 world corners; not the plane intersection polygon",
                "pixel_column_row": projection["pixel_corners"].tolist(),
                "signed_off_plane_distance_mm": projection["off_plane_distance_mm"].tolist(),
                "in_image_count": int(sum(projected_inside)),
            },
            "intersection": {
                "interpretation": "true box-plane polygon from intersections with the 12 box edges",
                "world_mm": intersection["world_mm"].tolist(),
                "pixel_column_row": intersection["pixel"].tolist(),
                "vertex_count": len(intersection["pixel"]),
                "plane_intersects_box": len(intersection["pixel"]) >= 3,
                "in_image_vertex_count": int(sum(intersection_inside)),
                "all_vertices_in_image": bool(intersection_inside) and all(intersection_inside),
            },
            "coverage_interpretation": "QC overlay only; cardiac anatomical coverage is visually assessed, not segmentation accuracy",
        }
        image_path = output / f"qc_{view.lower()}.png"
        _render_overlay(temporal_mean, view, row["slice_id"], distance, center_pixel, projection, intersection, image_path)
        image_paths[view] = image_path
    report_path = output / "roi_qc.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report_path, image_paths


def _render_overlay(
    temporal_mean: np.ndarray,
    view: str,
    slice_id: str,
    distance_mm: float,
    center_pixel: np.ndarray,
    projection: dict[str, np.ndarray],
    intersection: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Distinguish corner projections from the true intersection in the overlay."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.imshow(temporal_mean, cmap="gray", origin="upper", interpolation="nearest")
    projected = projection["pixel_corners"]
    axis.scatter(projected[:, 0], projected[:, 1], s=25, facecolors="none", edgecolors="gold", label="8 corner projections")
    polygon = intersection["pixel"]
    if len(polygon) >= 3:
        closed = np.vstack((polygon, polygon[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="red", linewidth=2.2, label="true box-plane intersection")
        axis.fill(polygon[:, 0], polygon[:, 1], color="red", alpha=0.10)
    axis.scatter([center_pixel[0]], [center_pixel[1]], marker="+", s=80, color="cyan", linewidths=1.8, label="box-center projection")
    axis.set_xlim(-0.5, temporal_mean.shape[1] - 0.5)
    axis.set_ylim(temporal_mean.shape[0] - 0.5, -0.5)
    axis.set_xlabel("DICOM column (pixel)")
    axis.set_ylabel("DICOM row (pixel)")
    axis.set_title(f"{view} {slice_id}\nshared cardiac box; plane distance {distance_mm:.3f} mm (not segmentation)")
    axis.legend(loc="upper right", fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project one patient-world cardiac box onto SAX/2CH/4CH temporal means.")
    parser.add_argument("--manifest", required=True, type=Path, help="Task 1 CSV manifest")
    parser.add_argument("--output-dir", required=True, type=Path, help="Ignored ROI QC output directory")
    parser.add_argument("--config", type=Path, help="Explicit ignored subject-local YAML containing roi box values")
    parser.add_argument("--center-mm", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Generic explicit alternative to --config")
    parser.add_argument("--size-mm", type=float, nargs=3, metavar=("SX", "SY", "SZ"), help="Generic explicit alternative to --config")
    parser.add_argument("--expected-frames-per-slice", type=int, default=50)
    args = parser.parse_args()
    if args.config is not None:
        if args.center_mm is not None or args.size_mm is not None:
            parser.error("--config cannot be combined with --center-mm/--size-mm")
        box = load_cardiac_box_config(args.config)
    else:
        if args.center_mm is None or args.size_mm is None:
            parser.error("provide --config or both --center-mm and --size-mm")
        box = CardiacBox(args.center_mm, args.size_mm)
    report, images = run_roi_qc(args.manifest, args.output_dir, box, args.expected_frames_per_slice)
    print(json.dumps({"report": str(report), "images": {view: str(path) for view, path in images.items()}}, indent=2))


if __name__ == "__main__":
    main()
