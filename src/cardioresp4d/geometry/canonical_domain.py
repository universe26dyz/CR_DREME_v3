"""Build a continuous full-FOV canonical domain and separate support QC maps.
论文来源：NISF++ patient-world geometry；2026-09-08 canonical-domain necessary adaptation。
输入：authoritative manifest、可选完整 QC table、DREME-style cardiac box、可选 Stage-1 reference mask。
输出：canonical_domain.json、各 view/union coverage NIfTI、domain_coverage_qc.png。
主要步骤：从 qc_valid planes 取 acquisition support，与 cardiac box 求并，保存 world/normalized transforms。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation；不从 initial reference affine 推断 canonical domain。
命令行使用示例：python -m cardioresp4d.geometry.canonical_domain --manifest results/dicom_manifest.csv --output-dir results/geometry --cardiac-box-json box.json
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

from cardioresp4d.data.dataset import validate_qc_table_coverage
from cardioresp4d.geometry.coordinate_normalization import WorldNormalizer
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.roi.cardiac_box import CardiacBox
from cardioresp4d.adapters.nesvor_psf import resolution_sigma_mm

_VIEWS = ("SAX", "2CH", "4CH")
_LPS_TO_RAS = np.diag((-1.0, -1.0, 1.0, 1.0))


def build_canonical_domain(
    manifest_path: str | Path,
    cardiac_box: CardiacBox,
    output_dir: str | Path,
    *,
    qc_table_path: str | Path | None = None,
    initial_reference_mask_path: str | Path | None = None,
    coverage_spacing_mm: float = 4.0,
) -> tuple[Path, dict[str, Path]]:
    """Build one explicit world-space domain from valid multi-view support and cardiac target box."""
    if not np.isfinite(coverage_spacing_mm) or coverage_spacing_mm <= 0.0:
        raise ValueError("coverage_spacing_mm must be finite and positive")
    records = _load_valid_planes(manifest_path, qc_table_path)
    observations = _load_valid_observations(manifest_path, qc_table_path)
    grouped: dict[str, list[DicomPlane]] = defaultdict(list)
    for row, plane in records:
        grouped[row["view"].upper()].append(plane)
    missing = [view for view in _VIEWS if not grouped[view]]
    if missing:
        raise ValueError("Canonical domain requires qc_valid SAX, 2CH, 4CH planes; missing " + ", ".join(missing))
    acquisition_corners = np.concatenate([plane.corners() for _, plane in records], axis=0)
    normalizer = WorldNormalizer.from_corners(np.concatenate((acquisition_corners, cardiac_box.corners_mm), axis=0))
    cardiac_box.validate_within(normalizer)
    lower, upper = normalizer.bounds_min_mm, normalizer.bounds_max_mm
    shape = tuple((np.ceil((upper - lower) / coverage_spacing_mm).astype(int) + 1).tolist())
    lps_affine = np.eye(4, dtype=np.float64)
    lps_affine[np.arange(3), np.arange(3)] = coverage_spacing_mm
    lps_affine[:3, 3] = lower
    ras_affine = _LPS_TO_RAS @ lps_affine
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    coverage_stats: dict[str, dict[str, int]] = {}
    canonical_mask = np.ones(shape, dtype=np.uint8)
    canonical_path = output / "canonical_domain_mask.nii.gz"
    nib.save(nib.Nifti1Image(canonical_mask, ras_affine), canonical_path)
    outputs["canonical_domain_mask"] = canonical_path
    psf_union = np.zeros(shape, dtype=np.uint8)
    psf_by_view: dict[str, np.ndarray] = {}
    for view in _VIEWS:
        occupancy = np.zeros(shape, dtype=np.uint8)
        psf_coverage = np.zeros(shape, dtype=np.uint8)
        target_hits = 0
        for plane in grouped[view]:
            points = _sample_plane(plane, coverage_spacing_mm)
            _mark(occupancy, points, lower, coverage_spacing_mm)
            _mark(psf_coverage, _sample_psf_support(plane, coverage_spacing_mm), lower, coverage_spacing_mm)
            target_hits += int(np.count_nonzero(_inside_box(points, cardiac_box)))
        path = output / f"coverage_plane_center_{view.lower()}.nii.gz"
        nib.save(nib.Nifti1Image(occupancy, ras_affine), path)
        outputs[f"coverage_plane_center_{view.lower()}"] = path
        psf_path = output / f"coverage_psf_{view.lower()}.nii.gz"
        nib.save(nib.Nifti1Image(psf_coverage, ras_affine), psf_path)
        outputs[f"coverage_psf_{view.lower()}"] = psf_path
        psf_by_view[view] = psf_coverage
        psf_union |= psf_coverage
        coverage_stats[view] = {"plane_center_voxels": int(occupancy.sum()), "psf_support_voxels": int(psf_coverage.sum()), "target_sample_points": target_hits}
        if target_hits == 0:
            raise ValueError(f"{view} has no sampled cardiac-box target support inside canonical domain")
    union_path = output / "coverage_psf_union.nii.gz"
    nib.save(nib.Nifti1Image(psf_union, ras_affine), union_path)
    outputs["coverage_psf_union"] = union_path
    view_count = np.add.reduce([(psf_by_view[view] > 0).astype(np.uint8) for view in _VIEWS], dtype=np.uint8)
    view_count_path = output / "coverage_view_count.nii.gz"
    nib.save(nib.Nifti1Image(view_count, ras_affine), view_count_path)
    outputs["coverage_view_count"] = view_count_path
    observation_count = np.zeros(shape, dtype=np.uint16)
    for _row, plane in observations:
        _mark_count(observation_count, _sample_psf_support(plane, coverage_spacing_mm), lower, coverage_spacing_mm)
    observation_count_path = output / "coverage_observation_count.nii.gz"
    nib.save(nib.Nifti1Image(observation_count, ras_affine), observation_count_path)
    outputs["coverage_observation_count"] = observation_count_path
    reference_support = _reference_support(initial_reference_mask_path)
    payload = {
        "schema_version": 2,
        "derivation_rule": "multi_view_acquisition_supported_full_fov",
        "acquisition_support": {"world_min_mm": acquisition_corners.min(axis=0).tolist(), "world_max_mm": acquisition_corners.max(axis=0).tolist(), "valid_plane_counts": {view: len(grouped[view]) for view in _VIEWS}},
        "initial_reference_support": reference_support,
        "canonical_reconstruction_domain": "continuous full-FOV patient-world AABB; independent of coverage and legacy initial-reference artifacts",
        "world_min_mm": lower.tolist(), "world_max_mm": upper.tolist(),
        "world_to_normalized": normalizer.forward_matrix.tolist(), "normalized_to_world": normalizer.inverse_matrix.tolist(),
        "scale_mm_per_normalized_unit": ((upper - lower) / 2.0).tolist(),
        "margin_mm": [0.0, 0.0, 0.0], "coverage_spacing_mm": coverage_spacing_mm,
        "cardiac_box": cardiac_box.to_dict(normalizer), "coverage": coverage_stats,
        "coverage_interpretation": "Sparse centre-plane zero voxels are not canonical-volume holes.",
        "acceptance": {"cardiac_box_within_domain": True, "all_views_have_target_support": True,
                       "initial_reference_mask_role": "legacy_not_used_by_mainline"},
    }
    report = output / "canonical_domain.json"; report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    qc = output / "domain_coverage_qc.png"; _render_qc(records, cardiac_box, lower, upper, qc)
    outputs["domain_coverage_qc"] = qc
    return report, outputs


def _load_valid_observations(manifest_path: str | Path, qc_table_path: str | Path | None) -> list[tuple[dict[str, str], DicomPlane]]:
    manifest = Path(manifest_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest contains no rows")
    qc_path = Path(qc_table_path) if qc_table_path is not None else manifest.parent / "acquisition_qc" / "acquisition_qc.csv"
    valid: set[tuple[str, ...]] | None = None
    token_mode = "source_file_token" in rows[0]
    if qc_path.is_file():
        validate_qc_table_coverage(manifest, qc_path)
        with qc_path.open(newline="", encoding="utf-8") as handle:
            qc_rows = list(csv.DictReader(handle))
        valid = {_frame_key(row, token_mode) for row in qc_rows if str(row.get("qc_valid", "1")) not in ("0", "false", "False")}
    return [(row, DicomPlane.from_geometry(row)) for row in rows if valid is None or _frame_key(row, token_mode) in valid]


def _load_valid_planes(manifest_path: str | Path, qc_table_path: str | Path | None) -> list[tuple[dict[str, str], DicomPlane]]:
    observations = _load_valid_observations(manifest_path, qc_table_path)
    unique: dict[tuple[str, str], tuple[dict[str, str], DicomPlane]] = {}
    for row, plane in observations:
        key = (row["view"].upper(), row["slice_id"])
        if key not in unique or int(row["frame_index"]) < int(unique[key][0]["frame_index"]): unique[key] = (row, plane)
    return [unique[key] for key in sorted(unique)]


def _frame_key(row: dict[str, str], token_mode: bool) -> tuple[str, ...]:
    return ((row.get("source_file_token", ""),) if token_mode else (row["view"], row["slice_id"], row["frame_index"]))


def _sample_plane(plane: DicomPlane, spacing_mm: float) -> np.ndarray:
    count_u = max(2, int(np.ceil((plane.columns - 1) * plane.pixel_spacing[1] / spacing_mm)) + 1)
    count_v = max(2, int(np.ceil((plane.rows - 1) * plane.pixel_spacing[0] / spacing_mm)) + 1)
    u, v = np.meshgrid(np.linspace(0.0, plane.columns - 1.0, count_u), np.linspace(0.0, plane.rows - 1.0, count_v), indexing="ij")
    return plane.pixel_to_world(np.stack((u.ravel(), v.ravel()), axis=-1))


def _mark(volume: np.ndarray, points: np.ndarray, lower: np.ndarray, spacing_mm: float) -> None:
    index = np.rint((points - lower) / spacing_mm).astype(int)
    index = np.clip(index, 0, np.asarray(volume.shape) - 1)
    volume[index[:, 0], index[:, 1], index[:, 2]] = 1


def _mark_count(volume: np.ndarray, points: np.ndarray, lower: np.ndarray, spacing_mm: float) -> None:
    index = np.rint((points - lower) / spacing_mm).astype(int)
    index = np.clip(index, 0, np.asarray(volume.shape) - 1)
    unique = np.unique(index, axis=0)
    volume[unique[:, 0], unique[:, 1], unique[:, 2]] += 1


def _sample_psf_support(plane: DicomPlane, spacing_mm: float) -> np.ndarray:
    """Approximate practical (3 sigma) NeSVoR PSF support in patient world-mm."""
    resolution = np.asarray([plane.pixel_spacing[1], plane.pixel_spacing[0], plane.slice_thickness], dtype=np.float32)
    sigma = resolution_sigma_mm(__import__("torch").from_numpy(resolution[None])).detach().cpu().numpy()[0]
    base = _sample_plane(plane, spacing_mm)
    normal_radius = max(1, int(np.ceil(3.0 * sigma[2] / spacing_mm)))
    offsets = np.arange(-normal_radius, normal_radius + 1, dtype=float) * spacing_mm
    return (base[:, None, :] + offsets[None, :, None] * plane.normal[None, None, :]).reshape(-1, 3)


def _inside_box(points: np.ndarray, box: CardiacBox) -> np.ndarray:
    return np.all((points >= box.min_mm) & (points <= box.max_mm), axis=-1)


def _reference_support(mask_path: str | Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "legacy_not_used_by_mainline", "used_to_define_canonical_domain": False}
    if mask_path is None:
        payload["available"] = False
        return payload
    image = nib.load(str(mask_path)); mask = np.asanyarray(image.dataobj) > 0
    # The result is deliberately PHI-free: record the support properties, not
    # the user-specific local path of the supervision mask.
    payload.update({"available": True, "valid_voxels": int(mask.sum()), "shape": list(mask.shape)})
    return payload


def _render_qc(records: list[tuple[dict[str, str], DicomPlane]], box: CardiacBox, lower: np.ndarray, upper: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    figure = plt.figure(figsize=(9, 7), constrained_layout=True); axis = figure.add_subplot(111, projection="3d")
    colours = {"SAX": "tab:blue", "2CH": "tab:orange", "4CH": "tab:green"}
    for row, plane in records:
        if int(row["frame_index"]) != 0: continue
        axis.add_collection3d(Poly3DCollection([plane.corners()], facecolors=colours[row["view"].upper()], alpha=.06))
    for points, colour, label in ((box.corners_mm, "red", "cardiac box"), (_box_corners(lower, upper), "black", "canonical domain")):
        for edge in _box_edges(points): axis.plot(*points[list(edge)].T, color=colour, linewidth=1.5, label=label if edge == (0, 1) else None)
    axis.set_xlabel("patient x (mm)"); axis.set_ylabel("patient y (mm)"); axis.set_zlabel("patient z (mm)"); axis.legend(); figure.savefig(path, dpi=160); plt.close(figure)


def _box_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])], dtype=float)


def _box_edges(points: np.ndarray) -> list[tuple[int, int]]:
    return [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit multi-view CardioResp canonical domain coverage.")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cardiac-box-json", required=True, type=Path); parser.add_argument("--qc-table", type=Path)
    args = parser.parse_args(); box = json.loads(args.cardiac_box_json.read_text())
    report, outputs = build_canonical_domain(args.manifest, CardiacBox(np.asarray(box["center_mm"]), np.asarray(box["size_mm"])), args.output_dir, qc_table_path=args.qc_table)
    print(json.dumps({"report": str(report), **{key: str(value) for key, value in outputs.items()}}, indent=2))


if __name__ == "__main__": main()
