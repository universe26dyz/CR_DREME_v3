"""Deterministic full-FOV Stage-1 reprojection, missing-plane and variance QC."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from cardioresp4d.training.export import export_volume, query_physical_plane, query_points
from cardioresp4d.training.sampling import pixel_indices_to_world, roi_pixel_pool
from cardioresp4d.rendering.psf_renderer import AnisotropicPSFRenderer


def _safe_ncc(prediction: np.ndarray, target: np.ndarray) -> float:
    left, right = prediction.reshape(-1), target.reshape(-1)
    left, right = left - left.mean(), right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom > 1e-12 else 0.0


def metric_row(prediction: np.ndarray, target: np.ndarray) -> dict:
    residual = prediction - target
    mse = float(np.mean(np.square(residual)))
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "NRMSE": float(np.sqrt(mse) / max(float(target.max() - target.min()), 1e-6)), "NCC": _safe_ncc(prediction, target)}


def _render_pixels(model, renderer, geometry: dict, *, chunk_pixels: int) -> np.ndarray:
    rows, columns = int(geometry["rows"]), int(geometry["columns"])
    rr, cc = np.indices((rows, columns)); flat_rows, flat_cols = rr.ravel(), cc.ravel()
    device = _model_device(model)
    orientation = np.asarray(geometry["image_orientation_patient"], dtype=float)
    row_direction = torch.tensor(orientation[:3], dtype=torch.float32, device=device)
    column_direction = torch.tensor(orientation[3:], dtype=torch.float32, device=device)
    spacing = torch.tensor(geometry["pixel_spacing"], dtype=torch.float32, device=device)
    thickness = torch.tensor(float(geometry["slice_thickness"]), dtype=torch.float32, device=device)
    output = []
    with torch.no_grad():
        for start in range(0, len(flat_rows), chunk_pixels):
            part_rows, part_cols = flat_rows[start:start + chunk_pixels], flat_cols[start:start + chunk_pixels]
            points = torch.tensor(pixel_indices_to_world(part_rows, part_cols, geometry), dtype=torch.float32, device=device)
            rendered = renderer.render_pixel_centers(model, pixel_centers_mm=points, row_direction=row_direction, column_direction=column_direction, normal=torch.linalg.cross(row_direction, column_direction), pixel_spacing_mm=spacing, thickness_mm=thickness)
            output.append(rendered["predicted"].detach().cpu().numpy())
    prediction = np.concatenate(output).reshape(rows, columns)
    if not np.isfinite(prediction).all():
        raise RuntimeError("PSF reprojection produced NaN/Inf")
    return prediction


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_mean_slices(model, mean_manifest: str | Path, domain: str | Path, output: str | Path, *, chunk_pixels: int = 8192) -> list[dict]:
    """Evaluate all mean-slice FOV pixels deterministically, chunked only for memory."""
    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be positive")
    manifest = Path(mean_manifest)
    root, output = manifest.parent, Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="") as file:
        observations = list(csv.DictReader(file))
    domain_payload = json.loads(Path(domain).read_text())
    box = domain_payload["cardiac_box"]
    lower = np.asarray(box["center_mm"], dtype=float) - np.asarray(box["size_mm"], dtype=float) / 2
    upper = np.asarray(box["center_mm"], dtype=float) + np.asarray(box["size_mm"], dtype=float) / 2
    renderer = AnisotropicPSFRenderer(torch.tensor(domain_payload["world_to_normalized"], dtype=torch.float32)).to(_model_device(model))
    records = []
    for row in observations:
        target = np.load(root / row["image_file"]).astype(np.float32)
        geometry = json.loads(row["geometry_json"])
        prediction = _render_pixels(model, renderer, geometry, chunk_pixels=chunk_pixels)
        pool = roi_pixel_pool(geometry, lower, upper, margin_mm=0.0)
        roi_index = pool["roi"]
        global_values = metric_row(prediction, target)
        roi_values = metric_row(prediction.ravel()[roi_index], target.ravel()[roi_index]) if len(roi_index) else {key: float("nan") for key in global_values}
        records.append({"mean_slice_id": row["mean_slice_id"], "view": row["view"], "slice_id": row["slice_id"], "n_pixels": int(target.size), **global_values, "roi_n_pixels": int(len(roi_index)), **{f"ROI_{key}": value for key, value in roi_values.items()}, "geometry_json": row["geometry_json"]})
    fieldnames = ["mean_slice_id", "view", "slice_id", "n_pixels", "MSE", "RMSE", "NRMSE", "NCC", "roi_n_pixels", "ROI_MSE", "ROI_RMSE", "ROI_NRMSE", "ROI_NCC"]
    with (output / "stage1b_per_slice_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows([{key: row[key] for key in fieldnames} for row in records])
    return records


def aggregate_by_view(records: list[dict]) -> dict:
    result = {}
    for view in ("SAX", "2CH", "4CH"):
        rows = [row for row in records if row["view"].upper() == view]
        if not rows:
            continue
        result[view] = {}
        for key in ("MSE", "NRMSE", "NCC", "ROI_MSE", "ROI_NRMSE", "ROI_NCC"):
            values = np.asarray([row[key] for row in rows], dtype=float)
            result[view][key] = ({"mean": float("nan"), "std": float("nan")} if not np.isfinite(values).any() else {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))})
        result[view]["n_observations"] = len(rows)
    return result


def select_representative_rows(rows: list[dict]) -> list[dict]:
    """Return low/middle/high physically ordered rows, not string-ID ordering."""
    if not rows:
        return []
    def position(row):
        geometry = json.loads(row["geometry_json"])
        orientation = np.asarray(geometry["image_orientation_patient"], dtype=float)
        normal = np.cross(orientation[:3], orientation[3:]); normal /= np.linalg.norm(normal)
        return float(np.dot(np.asarray(geometry["image_position_patient"], dtype=float), normal))
    ordered = sorted(rows, key=position)
    return [ordered[index] for index in sorted({0, len(ordered) // 2, len(ordered) - 1})]


def calibration_summary(*, squared_pixel_residual: np.ndarray, pixel_variance: np.ndarray, slice_mse: np.ndarray, observation_variance: np.ndarray, total_variance: np.ndarray) -> dict:
    """Keep pixel and observation calibration samples in their separate spaces."""
    def correlation(left, right):
        if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])
    return {
        "pixel_level": {"pearson_r": correlation(squared_pixel_residual, pixel_variance), "residual_squared_mean": float(np.mean(squared_pixel_residual)), "residual_squared_std": float(np.std(squared_pixel_residual)), "pixel_variance_mean": float(np.mean(pixel_variance)), "pixel_variance_std": float(np.std(pixel_variance))},
        "observation_level": {"pearson_r": correlation(slice_mse, observation_variance), "slice_mse_mean": float(np.mean(slice_mse)), "slice_mse_std": float(np.std(slice_mse)), "observation_variance_mean": float(np.mean(observation_variance)), "observation_variance_std": float(np.std(observation_variance))},
        "total_variance_mean": float(np.mean(total_variance)), "total_variance_std": float(np.std(total_variance)),
        "normalized_residual_mean": float(np.mean(squared_pixel_residual / total_variance)),
    }


def write_representative_figures(model, mean_manifest, domain, records, output, *, chunk_pixels=8192) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    manifest, output = Path(mean_manifest), Path(output)
    root = manifest.parent
    with manifest.open(newline="") as file:
        observations = {row["mean_slice_id"]: row for row in csv.DictReader(file)}
    for view in ("SAX", "2CH", "4CH"):
        selected = select_representative_rows([row for row in records if row["view"].upper() == view])
        if not selected:
            continue
        figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
        axes = np.atleast_2d(axes)
        for axis_row, row in zip(axes, selected):
            target = np.load(root / observations[row["mean_slice_id"]]["image_file"])
            geometry = json.loads(row["geometry_json"])
            renderer = AnisotropicPSFRenderer(torch.tensor(json.loads(Path(domain).read_text())["world_to_normalized"], dtype=torch.float32)).to(_model_device(model))
            prediction = _render_pixels(model, renderer, geometry, chunk_pixels=chunk_pixels)
            window = (min(float(target.min()), float(prediction.min())), max(float(target.max()), float(prediction.max())))
            for axis, image, title in zip(axis_row, (target, prediction, np.abs(prediction - target)), ("acquired temporal mean", "predicted", "absolute residual")):
                axis.imshow(image, cmap="gray", vmin=None if title == "absolute residual" else window[0], vmax=None if title == "absolute residual" else window[1]); axis.set_title(f"{row['slice_id']} {title}"); axis.axis("off")
        figure.tight_layout(); figure.savefig(output / f"{view.lower()}_representative_reprojection.png", dpi=120); plt.close(figure)


def _geometry_from_source_row(row: dict) -> dict:
    required = ("image_position_patient", "image_orientation_patient", "pixel_spacing", "rows", "columns")
    if not all(key in row and row[key] != "" for key in required):
        raise ValueError("source manifest row lacks DICOM geometry required for missing-plane QC")
    return {
        "image_position_patient": json.loads(row["image_position_patient"]),
        "image_orientation_patient": json.loads(row["image_orientation_patient"]),
        "pixel_spacing": json.loads(row["pixel_spacing"]),
        "slice_thickness": float(row.get("slice_thickness", row.get("spacing_between_slices", 1.0))),
        "rows": int(row["rows"]), "columns": int(row["columns"]),
    }


def _missing_sax_geometry(source_manifest, mean_records: list[dict]) -> tuple[str, dict] | None:
    with Path(source_manifest).open(newline="") as file:
        source = list(csv.DictReader(file))
    source_sax = {}
    for row in source:
        if row.get("view", "").upper() == "SAX":
            source_sax.setdefault(row["slice_id"], row)
    observed = {row["slice_id"] for row in mean_records if row["view"].upper() == "SAX"}
    missing = [(slice_id, row) for slice_id, row in source_sax.items() if slice_id not in observed]
    if len(missing) != 1:
        return None
    slice_id, row = missing[0]
    return slice_id, _geometry_from_source_row(row)


def _write_missing_plane_qc(stage1a_model, stage1b_model, domain, source_manifest, mean_manifest, records, output, cardiac_lower, cardiac_upper, chunk_pixels) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    missing = _missing_sax_geometry(source_manifest, records)
    summary = {"s17_absence_confirmed": False, "query_source": "continuous_inr_not_nifti"}
    if missing is None:
        return summary
    slice_id, geometry = missing
    stage1a = query_physical_plane(stage1a_model, domain, geometry, chunk_points=chunk_pixels)
    stage1b = query_physical_plane(stage1b_model, domain, geometry, chunk_points=chunk_pixels)
    difference = stage1b - stage1a
    np.save(output / "s17_stage1a_prediction.npy", stage1a)
    np.save(output / "s17_stage1b_prediction.npy", stage1b)
    np.save(output / "s17_stage1b_minus_stage1a.npy", difference)
    for image, filename, title in ((stage1a, "s17_stage1a_prediction.png", "s17 Stage1A queried"), (stage1b, "s17_stage1b_prediction.png", "s17 Stage1B queried"), (difference, "s17_stage1b_minus_stage1a.png", "s17 Stage1B - Stage1A")):
        plt.figure(figsize=(5, 4)); plt.imshow(image, cmap="gray"); plt.title(title); plt.axis("off"); plt.tight_layout(); plt.savefig(output / filename, dpi=120); plt.close()
    orientation = np.asarray(geometry["image_orientation_patient"], dtype=float)
    normal = np.cross(orientation[:3], orientation[3:]); normal /= np.linalg.norm(normal)
    missing_position = float(np.dot(np.asarray(geometry["image_position_patient"]), normal))
    sax = [row for row in records if row["view"].upper() == "SAX"]
    def position(row):
        item = json.loads(row["geometry_json"])
        return float(np.dot(np.asarray(item["image_position_patient"]), normal))
    ordered = sorted(sax, key=position)
    lower_neighbours = [row for row in ordered if position(row) < missing_position][-2:]
    upper_neighbours = [row for row in ordered if position(row) > missing_position][:2]
    neighbours = lower_neighbours + upper_neighbours
    figure, axes = plt.subplots(5, 3, figsize=(9, 15))
    with Path(mean_manifest).open(newline="") as file:
        observations = {row["mean_slice_id"]: row for row in csv.DictReader(file)}
    for index, row in enumerate(neighbours):
        target = np.load(Path(mean_manifest).parent / observations[row["mean_slice_id"]]["image_file"])
        predicted = query_physical_plane(stage1b_model, domain, json.loads(row["geometry_json"]), chunk_points=chunk_pixels)
        for axis, image, title in zip(axes[index], (target, predicted, np.abs(predicted - target)), ("acquired temporal mean", "reconstructed", "absolute residual")):
            axis.imshow(image, cmap="gray"); axis.set_title(f"{row['slice_id']} {title}"); axis.axis("off")
    for axis, image, title in zip(axes[4], (stage1a, stage1b, difference), ("s17 Stage1A queried", "s17 Stage1B queried", "s17 B-A")):
        axis.imshow(image, cmap="gray"); axis.set_title(title); axis.axis("off")
    figure.tight_layout(); figure.savefig(output / "s17_neighboring_sax_qc.png", dpi=120); plt.close(figure)
    # Through-plane reformat uses the same physical normal and is queried directly.
    center = (np.asarray(cardiac_lower) + np.asarray(cardiac_upper)) / 2
    row_direction = orientation[:3] / np.linalg.norm(orientation[:3])
    column_direction = orientation[3:] / np.linalg.norm(orientation[3:])
    inplane_extent = max(20.0, abs(np.dot(np.asarray(cardiac_upper) - np.asarray(cardiac_lower), column_direction)))
    normal_extent = max(20.0, abs(np.dot(np.asarray(cardiac_upper) - np.asarray(cardiac_lower), normal)))
    inplane_axis = np.linspace(-inplane_extent / 2, inplane_extent / 2, 64)
    normal_axis = np.linspace(-normal_extent / 2, normal_extent / 2, 64)
    aa, bb = np.meshgrid(inplane_axis, normal_axis, indexing="ij")
    points = center + aa[..., None] * column_direction + bb[..., None] * normal
    through = query_points(stage1b_model, domain, points, chunk_points=chunk_pixels)
    if not np.isfinite(through).all():
        raise RuntimeError("s17 through-plane query produced NaN/Inf")
    plt.figure(figsize=(6, 5)); plt.imshow(through, cmap="gray"); plt.title("cardiac through-plane reformat containing queried s17"); plt.axis("off"); plt.tight_layout(); plt.savefig(output / "s17_through_plane_qc.png", dpi=120); plt.close()
    summary.update({"s17_absence_confirmed": True, "missing_slice_id": slice_id, "neighbouring_acquired_count": len(neighbours), "through_plane_finite": True, "through_plane_zero_layer": bool(np.all(np.any(np.abs(through) > 1e-8, axis=1))), "through_plane_intensity_range": [float(through.min()), float(through.max())], "through_plane_max_adjacent_jump": float(np.abs(np.diff(through, axis=1)).max())})
    return summary


def run_final_static_qc(stage1a_model, stage1b_model, mean_manifest, source_manifest, domain, output, *, overview_spacing_mm=4.0, cardiac_export_spacing_mm=1.5, cardiac_roi_margin_mm=15.0, eval_chunk_pixels=8192, stage1b_metadata: dict | None = None) -> dict:
    """Create physical exports and deterministic Stage-1 static evaluation artifacts."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(domain).read_text())
    domain_lower, domain_upper = np.asarray(payload["world_min_mm"]), np.asarray(payload["world_max_mm"])
    box = payload["cardiac_box"]
    cardiac_lower = np.asarray(box["center_mm"]) - np.asarray(box["size_mm"]) / 2 - cardiac_roi_margin_mm
    cardiac_upper = np.asarray(box["center_mm"]) + np.asarray(box["size_mm"]) / 2 + cardiac_roi_margin_mm
    overview = export_volume(stage1b_model, domain, domain_lower, domain_upper, overview_spacing_mm, output / "canonical_overview.nii.gz", chunk_points=eval_chunk_pixels)
    first = export_volume(stage1a_model, domain, cardiac_lower, cardiac_upper, cardiac_export_spacing_mm, output / "stage1a_cardiac_1p5mm.nii.gz", chunk_points=eval_chunk_pixels)
    second = export_volume(stage1b_model, domain, cardiac_lower, cardiac_upper, cardiac_export_spacing_mm, output / "stage1b_cardiac_1p5mm.nii.gz", grid=first.grid, chunk_points=eval_chunk_pixels)
    difference = second.data - first.data
    nib.save(nib.Nifti1Image(difference.astype(np.float32), first.grid.affine_ras), str(output / "stage1a_vs_stage1b_cardiac_difference.nii.gz"))
    nib.save(nib.Nifti1Image(np.abs(difference).astype(np.float32), first.grid.affine_ras), str(output / "stage1a_vs_stage1b_cardiac_absolute_difference.nii.gz"))
    anatomy = {"mean_absolute_difference": float(np.mean(np.abs(difference))), "absolute_difference_percentiles": {"p50": float(np.percentile(np.abs(difference), 50)), "p90": float(np.percentile(np.abs(difference), 90)), "p99": float(np.percentile(np.abs(difference), 99))}, "stage1a_intensity_range": [float(first.data.min()), float(first.data.max())], "stage1b_intensity_range": [float(second.data.min()), float(second.data.max())], "grid_shape": list(first.grid.shape), "grid_spacing_mm": first.grid.spacing_mm.tolist()}
    (output / "stage1_anatomy_difference_summary.json").write_text(json.dumps(anatomy, indent=2))
    records = evaluate_mean_slices(stage1b_model, mean_manifest, domain, output, chunk_pixels=eval_chunk_pixels)
    per_view = aggregate_by_view(records); (output / "stage1b_per_view_metrics.json").write_text(json.dumps(per_view, indent=2))
    write_representative_figures(stage1b_model, mean_manifest, domain, records, output, chunk_pixels=eval_chunk_pixels)
    s17 = _write_missing_plane_qc(stage1a_model, stage1b_model, domain, source_manifest, mean_manifest, records, output, cardiac_lower, cardiac_upper, eval_chunk_pixels)
    intersections = {view: 0 for view in ("SAX", "2CH", "4CH")}
    for row in records:
        geometry = json.loads(row["geometry_json"])
        intersections[row["view"].upper()] += int(roi_pixel_pool(geometry, cardiac_lower, cardiac_upper, margin_mm=0.0)["intersects_roi"])
    coverage = {"canonical_domain_extent_mm": (domain_upper - domain_lower).tolist(), "cardiac_export_requested_lps_bounds_mm": [cardiac_lower.tolist(), cardiac_upper.tolist()], "cardiac_export_sampled_lps_bounds_mm": [first.grid.lower_lps_mm.tolist(), first.grid.upper_lps_mm.tolist()], "mean_slice_observations_per_view": {view: sum(row["view"].upper() == view for row in records) for view in ("SAX", "2CH", "4CH")}, "cardiac_roi_intersection_count_per_view": intersections, "no_roi_intersection_count": (stage1b_metadata or {}).get("no_roi_intersection"), "total_sampled_target_support_metadata": {key: (stage1b_metadata or {}).get(key) for key in ("pixels_per_view", "roi_fraction", "steps_per_epoch", "visits")}, "cardiac_support_fraction": None, "support_fraction_note": "reliable observation-plane coverage summary only; no approximate PSF-footprint voxel fraction was inferred", **s17}
    (output / "stage1_coverage_summary.json").write_text(json.dumps(coverage, indent=2))
    (output / "stage1b_uncertainty_qc.json").write_text(json.dumps({"uncertainty_mode": (stage1b_metadata or {}).get("uncertainty_mode", "off"), "enabled": False, "pixel_level_calibration": None, "observation_level_calibration": None, "note": "production Stage1B is MSE-only; variance diagnostics require an explicitly enabled calibrated run"}, indent=2))
    return {"overview": str(overview.path), "cardiac_grid_shape": list(first.grid.shape), "per_view": per_view, "s17": s17, "coverage": coverage}
