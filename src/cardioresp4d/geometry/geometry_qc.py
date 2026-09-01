"""
功能：从 DICOM manifest 生成病人世界平面、中心、法线和数值往返误差的几何质控。
论文来源：Necessary adaptation: visually and numerically audited physical geometry for local MRI.
输入：Task 1 CSV manifest，以及忽略的结果输出目录。
输出：``geometry_qc.json`` 和 ``geometry_qc.png``，含 4x4 归一化矩阵与真实往返误差。
主要步骤：读取唯一 DICOM 平面、以全部角点拟合世界包围盒、抽样 SAX/2CH/4CH 并渲染 3D 图。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：MPLCONFIGDIR=/tmp/cardioresp-mpl python -m cardioresp4d.geometry.geometry_qc --manifest results/dicom_manifest.csv --output-dir results/geometry_qc
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cardioresp4d.geometry.coordinate_normalization import WorldNormalizer
from cardioresp4d.geometry.world_geometry import DicomPlane


_REQUIRED_VIEWS = ("SAX", "2CH", "4CH")


def load_manifest_planes(manifest_path: str | Path) -> list[tuple[dict[str, str], DicomPlane]]:
    """Load one representative geometry per view/slice from a Task 1 CSV manifest."""
    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    required = {"view", "slice_id", "frame_index", "image_position_patient", "image_orientation_patient", "pixel_spacing", "rows", "columns"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing required field(s): {', '.join(sorted(missing))}")
    unique: dict[tuple[str, str], tuple[dict[str, str], DicomPlane]] = {}
    for row in rows:
        key = (row["view"].upper(), row["slice_id"])
        if key not in unique or int(row["frame_index"]) < int(unique[key][0]["frame_index"]):
            unique[key] = (row, DicomPlane.from_geometry(row))
    return [unique[key] for key in sorted(unique)]


def choose_qc_planes(planes: Iterable[tuple[dict[str, str], DicomPlane]], max_per_view: int = 3) -> list[tuple[dict[str, str], DicomPlane]]:
    """Select evenly spaced planes ordered by physical stack position within each view."""
    if max_per_view <= 0:
        raise ValueError("max_per_view must be positive")
    grouped: dict[str, list[tuple[dict[str, str], DicomPlane]]] = defaultdict(list)
    for record in planes:
        grouped[record[0]["view"].upper()].append(record)
    selected: list[tuple[dict[str, str], DicomPlane]] = []
    for view in sorted(grouped):
        items = sorted(grouped[view], key=lambda item: item[0]["slice_id"])
        stack_normal = items[0][1].normal
        items.sort(key=lambda item: (
            float(np.dot(item[1].pixel_to_world((item[1].columns - 1.0) / 2.0, (item[1].rows - 1.0) / 2.0), stack_normal)),
            item[0]["slice_id"],
        ))
        indices = np.linspace(0, len(items) - 1, min(len(items), max_per_view), dtype=int)
        selected.extend(items[index] for index in np.unique(indices))
    return selected


def run_geometry_qc(manifest_path: str | Path, output_dir: str | Path, max_per_view: int = 3) -> tuple[Path, Path]:
    """Write JSON/PNG geometry QC using real manifest planes and return both artifact paths."""
    all_records = load_manifest_planes(manifest_path)
    present_views = {row["view"].upper() for row, _ in all_records}
    missing_views = [view for view in _REQUIRED_VIEWS if view not in present_views]
    if missing_views:
        raise ValueError(
            "Manifest must contain required views: SAX, 2CH, 4CH; missing "
            + ", ".join(missing_views)
        )
    normalizer = WorldNormalizer.from_planes([plane for _, plane in all_records])
    selected = choose_qc_planes(all_records, max_per_view=max_per_view)
    if not selected:
        raise ValueError("No planes selected for geometry QC")
    errors = _round_trip_errors(plane for _, plane in selected)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "geometry_qc.json"
    image_path = output / "geometry_qc.png"
    report = {
        "manifest": str(Path(manifest_path)),
        "plane_count": len(all_records),
        "selected_plane_count": len(selected),
        "world_normalizer": normalizer.to_dict(),
        "round_trip_errors": errors,
        "planes": [_plane_report(row, plane) for row, plane in selected],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    _render_planes(selected, image_path)
    return report_path, image_path


def _round_trip_errors(planes: Iterable[DicomPlane]) -> dict[str, float]:
    """Measure requested in-plane pixel/world round trips with centres and four corners."""
    pixel_errors: list[float] = []
    world_errors: list[float] = []
    for plane in planes:
        pixels = np.array(((0.0, 0.0), (plane.columns - 1.0, 0.0), (plane.columns - 1.0, plane.rows - 1.0), (0.0, plane.rows - 1.0), ((plane.columns - 1.0) / 2.0, (plane.rows - 1.0) / 2.0)))
        world = plane.pixel_to_world(pixels)
        recovered_pixels = plane.world_to_pixel(world)
        recovered_world = plane.pixel_to_world(recovered_pixels)
        pixel_errors.append(float(np.max(np.linalg.norm(recovered_pixels - pixels, axis=-1))))
        world_errors.append(float(np.max(np.linalg.norm(recovered_world - world, axis=-1))))
    return {"pixel_to_world_to_pixel_max_error_pixels": max(pixel_errors), "world_to_pixel_to_world_max_error_mm": max(world_errors)}


def _plane_report(row: dict[str, str], plane: DicomPlane) -> dict[str, Any]:
    """Build one JSON-safe visual/QC record without writing source DICOM identifiers."""
    centre_pixel = np.array(((plane.columns - 1.0) / 2.0, (plane.rows - 1.0) / 2.0))
    return {"view": row["view"].upper(), "slice_id": row["slice_id"], "frame_index": int(row["frame_index"]), "center_mm": plane.pixel_to_world(centre_pixel).tolist(), "normal": plane.normal.tolist(), "corners_mm": plane.corners().tolist(), "geometry": plane.to_dict()}


def _render_planes(records: Iterable[tuple[dict[str, str], DicomPlane]], image_path: Path) -> None:
    """Render selected world-space plane polygons, centres, and normal arrows in a PNG."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    records_list = list(records)
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    colours = {"SAX": "tab:blue", "2CH": "tab:orange", "4CH": "tab:green"}
    all_points: list[np.ndarray] = []
    for index, (row, plane) in enumerate(records_list):
        corners = plane.corners()
        centre = plane.pixel_to_world((plane.columns - 1.0) / 2.0, (plane.rows - 1.0) / 2.0)
        colour = colours.get(row["view"].upper(), f"C{index % 10}")
        axis.add_collection3d(Poly3DCollection([corners], facecolors=colour, edgecolors=colour, alpha=0.25, linewidths=1.2))
        axis.scatter(*centre, color=colour, s=28)
        arrow_length = max(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)) * 0.15, 5.0)
        axis.quiver(*centre, *plane.normal, length=arrow_length, color=colour, arrow_length_ratio=0.12)
        axis.text(*centre, f"{row['view'].upper()} {row['slice_id']}", fontsize=7)
        all_points.extend((corners, centre[None, :], (centre + arrow_length * plane.normal)[None, :]))
    points = np.concatenate(all_points, axis=0)
    _set_equal_3d_axes(axis, points)
    axis.set_xlabel("Patient x (mm)")
    axis.set_ylabel("Patient y (mm)")
    axis.set_zlabel("Patient z (mm)")
    axis.set_title("CardioResp DICOM patient-world geometry QC")
    figure.savefig(image_path, dpi=180)
    plt.close(figure)


def _set_equal_3d_axes(axis: Any, points: np.ndarray) -> None:
    """Set equal world-space scale on Matplotlib's three dimensions for physical inspection."""
    lower, upper = points.min(axis=0), points.max(axis=0)
    centre = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1.0)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def main() -> None:
    """Generate real DICOM manifest geometry QC as JSON plus a rendered 3D PNG."""
    parser = argparse.ArgumentParser(description="Render and numerically audit DICOM patient-world geometry from a manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="Task 1 CSV manifest path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Ignored directory for geometry_qc.json and geometry_qc.png")
    parser.add_argument("--max-per-view", type=int, default=3, help="Maximum evenly sampled planes rendered per view")
    args = parser.parse_args()
    report_path, image_path = run_geometry_qc(args.manifest, args.output_dir, args.max_per_view)
    print(json.dumps({"report": str(report_path), "image": str(image_path)}, indent=2))


if __name__ == "__main__":
    main()
