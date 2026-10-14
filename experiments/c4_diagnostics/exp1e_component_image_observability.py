#!/usr/bin/env python3
"""Offline image-observability audit of respiratory DVF slice components.

This diagnostic deliberately bypasses all learned motion/encoder paths.  It
evaluates the analytic respiratory DVF on the exact, cached matched-CRN PSF
query points used by Exp1-C and asks how much each DICOM slice-basis component
changes the acquired image.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, geometry_from_manifest
from experiments.c4_diagnostics.exp1b_observability_oracle import _full_image_uv
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.oracle import TargetPSFOffsetCache, build_oracle_motion_model, build_target_psf_offset_cache
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame


COMPONENTS = ("row", "column", "normal")
EPSILON = 1e-12


def _basis(row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    basis = torch.stack((row_direction, column_direction, normal)).to(like)
    if basis.shape != (3, 3) or not torch.allclose(basis @ basis.T, torch.eye(3, dtype=basis.dtype, device=basis.device), atol=1e-4, rtol=0.0):
        raise ValueError("DICOM row/column/normal basis must be orthonormal")
    return basis


def decompose_world_dvf(dvf_world_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Return DICOM row/column/normal coefficients for a world-mm DVF."""
    if dvf_world_mm.shape[-1] != 3:
        raise ValueError("world DVF must end in xyz")
    return torch.einsum("...c,kc->...k", dvf_world_mm, _basis(row_direction, column_direction, normal, like=dvf_world_mm))


def reconstruct_world_dvf(components_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Reconstruct a world-mm DVF from DICOM row/column/normal coefficients."""
    if components_mm.shape[-1] != 3:
        raise ValueError("slice components must end in row/column/normal")
    return torch.einsum("...k,kc->...c", components_mm, _basis(row_direction, column_direction, normal, like=components_mm))


def _canonical_intensity(canonical, reference_points_mm: torch.Tensor) -> torch.Tensor:
    output = canonical(reference_points_mm, return_features=True)
    intensity = output[0] if isinstance(output, tuple) else output
    if intensity.ndim != 2:
        raise ValueError("canonical renderer must return [pixels, psf_samples] intensity")
    return intensity.mean(-1)


def render_component_variants(canonical, psf_query_points: torch.Tensor, full_resp_dvf_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor) -> dict[str, torch.Tensor]:
    """Render full/remove/flip/zero variants from one fixed PSF-query tensor."""
    if psf_query_points.shape != full_resp_dvf_mm.shape or psf_query_points.ndim != 3 or psf_query_points.shape[-1] != 3:
        raise ValueError("PSF query points and respiratory DVF must be matched [pixels, samples, xyz]")
    components = decompose_world_dvf(full_resp_dvf_mm, row_direction, column_direction, normal)
    result: dict[str, torch.Tensor] = {"psf_query_points": psf_query_points, "full": _canonical_intensity(canonical, psf_query_points + full_resp_dvf_mm), "zero": _canonical_intensity(canonical, psf_query_points)}
    for index, component in enumerate(COMPONENTS):
        world_component = reconstruct_world_dvf(torch.nn.functional.one_hot(torch.full(components.shape[:-1], index, dtype=torch.long, device=components.device), num_classes=3).to(components.dtype) * components[..., index : index + 1], row_direction, column_direction, normal)
        result[f"remove_{component}"] = _canonical_intensity(canonical, psf_query_points + full_resp_dvf_mm - world_component)
        result[f"flip_{component}"] = _canonical_intensity(canonical, psf_query_points + full_resp_dvf_mm - 2.0 * world_component)
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float("nan") if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= EPSILON else numerator / denominator


def component_image_metrics(full_gt: torch.Tensor, zero_motion: torch.Tensor, remove_component: torch.Tensor, flip_component: torch.Tensor, component_dvf_mm: torch.Tensor, total_dvf_world_mm: torch.Tensor | None = None) -> dict[str, float]:
    """Image effects and dynamic component strength for one fixed location."""
    if not (full_gt.shape == zero_motion.shape == remove_component.shape == flip_component.shape) or full_gt.ndim < 2:
        raise ValueError("image variants must be matched [time, ...]")
    if component_dvf_mm.shape[0] != full_gt.shape[0]:
        raise ValueError("component DVF and images must share time frames")
    dynamic = component_dvf_mm - component_dvf_mm.mean(dim=0, keepdim=True)
    component_rms = float(dynamic.square().mean().sqrt())
    dynamic_energy = float(dynamic.square().sum())
    if total_dvf_world_mm is None:
        total_energy = dynamic_energy
    else:
        total_dynamic = total_dvf_world_mm - total_dvf_world_mm.mean(dim=0, keepdim=True)
        total_energy = float(total_dynamic.square().sum())
    remove_mse = float((full_gt - remove_component).square().mean())
    flip_mse = float((full_gt - flip_component).square().mean())
    total_motion_mse = float((full_gt - zero_motion).square().mean())
    return {
        "gt_component_dynamic_rms_mm": component_rms,
        "gt_component_energy_fraction": _safe_ratio(dynamic_energy, total_energy),
        "remove_mse": remove_mse,
        "remove_rmse": math.sqrt(remove_mse),
        "flip_mse": flip_mse,
        "flip_rmse": math.sqrt(flip_mse),
        "remove_sensitivity_mse_per_mm2": _safe_ratio(remove_mse, component_rms**2),
        "flip_sensitivity_mse_per_mm2": _safe_ratio(flip_mse, (2.0 * component_rms) ** 2),
        "total_motion_mse": total_motion_mse,
        "remove_effect_fraction": _safe_ratio(remove_mse, total_motion_mse),
        "flip_effect_fraction": _safe_ratio(flip_mse, total_motion_mse),
    }


def _prepare_output(path: Path, *, input_dir: Path, overwrite: bool) -> None:
    if path.resolve() == input_dir.resolve():
        raise ValueError("Exp1-E output must differ from the Exp1-C input directory")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _frame_by_location(frames: list[GeometryFrame]) -> dict[str, list[GeometryFrame]]:
    grouped: dict[str, list[GeometryFrame]] = defaultdict(list)
    for frame in frames:
        grouped[f"{frame.view}/{frame.slice_id}"].append(frame)
    return {location: sorted(sequence, key=lambda frame: frame.timestamp_s) for location, sequence in grouped.items()}


def _check_location_basis(frames: list[GeometryFrame]) -> GeometryFrame:
    if not frames:
        raise ValueError("location has no geometry frames")
    reference = frames[0]
    for frame in frames[1:]:
        if not (torch.allclose(frame.row_direction, reference.row_direction, atol=1e-4, rtol=0.0) and torch.allclose(frame.column_direction, reference.column_direction, atol=1e-4, rtol=0.0) and torch.allclose(frame.normal, reference.normal, atol=1e-4, rtol=0.0)):
            raise ValueError(f"DICOM basis changes within fixed location {reference.view}/{reference.slice_id}")
    return reference


def _read_locations(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["location"] for row in csv.DictReader(handle)}


def _exp1d_recovery_rows(exp1d_dir: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    if exp1d_dir is None:
        return {}
    path = exp1d_dir / "component_per_location.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    joined: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            location = record["location"]
            joined[location] = {}
            for component in COMPONENTS:
                joined[location][component] = {
                    metric: float(record[f"{component}_{metric}"])
                    for metric in ("rms_ratio", "correlation", "normalized_epe")
                    if record.get(f"{component}_{metric}", "") != ""
                }
    return joined


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _aggregate_rows(rows: list[dict[str, object]], *, key: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for row in rows:
        scope = str(row[key])
        component = str(row["component"])
        grouped.setdefault((scope, component), {})[str(row["location"])] = {name: float(value) for name, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    output: list[dict[str, object]] = []
    for (scope, component), values in sorted(grouped.items()):
        aggregate = aggregate_location_metrics(values)
        flattened: dict[str, object] = {key: scope, "component": component, "n_locations": len(values)}
        for metric, statistics in aggregate.items():
            flattened.update({f"{metric}_{statistic}": value for statistic, value in statistics.items()})
        output.append(flattened)
    return output


def _location_variants(model, observation, cache: TargetPSFOffsetCache, motion: GroundTruthMotion, *, pixel_chunk: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if pixel_chunk < 1:
        raise ValueError("pixel chunk must be positive")
    query_points: list[torch.Tensor] = []
    for pixel_uv in _full_image_uv(observation).split(pixel_chunk):
        query_points.append(model.psf_sample_world_points(observation, pixel_uv, cache.for_pixel_uv(pixel_uv)))
    samples = torch.cat(query_points)
    timestamp = samples.new_tensor([observation.timestamp_s])
    dvf = motion.respiratory_dvf_mm(samples, timestamp)
    variants = render_component_variants(model.canonical, samples, dvf, observation.row_direction, observation.column_direction, observation.normal)
    height, width = observation.image.shape[-2:]
    return {name: value.reshape(height, width) if name != "psf_query_points" else value for name, value in variants.items()}, dvf


def _validate_exp1c(input_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    checkpoint_path, summary_path, locations_path, arrays_path = input_dir / "oracle_model_last.pt", input_dir / "summary.json", input_dir / "oracle_per_location.csv", input_dir / "recovery_arrays.npz"
    for path in (checkpoint_path, summary_path, locations_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if checkpoint.get("branch") != "resp" or checkpoint.get("mode") != "oracle_resp_only" or "model_state_dict" not in checkpoint:
        raise ValueError("Exp1-E requires an Exp1-C oracle_resp_only respiratory checkpoint")
    if checkpoint.get("psf_training_mode") != "matched_crn" or summary.get("training", {}).get("psf_training_mode") != "matched_crn":
        raise ValueError("Exp1-E requires an Exp1-C matched_crn oracle output")
    return checkpoint, summary


def _source_render_spec(input_dir: Path, locations: set[str]) -> tuple[tuple[int, int], int, dict[str, np.ndarray]]:
    """Read image/PSF dimensions and exact selected timestamps from Exp1-C arrays."""
    timestamps: dict[str, np.ndarray] = {}
    image_shapes: set[tuple[int, int]] = set()
    psf_samples: set[int] = set()
    with np.load(input_dir / "recovery_arrays.npz", allow_pickle=False) as arrays:
        for location in locations:
            stem = f"oracle_{location.replace('/', '_')}"
            image_key, psf_key, timestamp_key = f"{stem}_predicted_images", f"{stem}_psf_support_gt_dvf_mm", f"{stem}_timestamps_s"
            if image_key not in arrays or psf_key not in arrays or timestamp_key not in arrays:
                raise KeyError(f"Exp1-C arrays missing matched-CRN support data for {location}")
            images, psf_dvf, selected_timestamps = arrays[image_key], arrays[psf_key], arrays[timestamp_key]
            if images.ndim != 4 or psf_dvf.ndim != 4 or psf_dvf.shape[-1] != 3:
                raise ValueError(f"invalid Exp1-C image/PSF array shapes for {location}")
            if images.shape[0] != psf_dvf.shape[0] or images.shape[0] != len(selected_timestamps):
                raise ValueError(f"Exp1-C frame-count mismatch inside stored arrays for {location}")
            image_shapes.add(tuple(int(value) for value in images.shape[-2:]))
            psf_samples.add(int(psf_dvf.shape[-2]))
            timestamps[location] = np.asarray(selected_timestamps, dtype=np.float64)
    if len(image_shapes) != 1 or len(psf_samples) != 1:
        raise ValueError("Exp1-C locations do not share one image shape and PSF sample count")
    return image_shapes.pop(), psf_samples.pop(), timestamps


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    input_dir = args.input_dir
    checkpoint, source_summary = _validate_exp1c(input_dir)
    _prepare_output(args.output_dir, input_dir=input_dir, overwrite=args.overwrite)
    with args.experiment_config.open(encoding="utf-8") as handle:
        synthetic = yaml.safe_load(handle)["synthetic"]
    domain = _load_domain(args.canonical_domain)
    motion = GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=0.0)
    device = torch.device(args.device)
    if device.type != "cpu":
        raise ValueError("Exp1-E is an offline CPU diagnostic; use --device cpu")
    seed = int(checkpoint["seed"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device)
    by_location = _frame_by_location(frames)
    expected_locations = _read_locations(input_dir / "oracle_per_location.csv")
    if set(by_location) != expected_locations:
        raise ValueError("manifest geometry locations do not match Exp1-C oracle output")
    image_shape, psf_samples, source_timestamps = _source_render_spec(input_dir, expected_locations)
    for location, selected in by_location.items():
        actual = np.asarray([frame.timestamp_s for frame in selected], dtype=np.float64)
        if actual.shape != source_timestamps[location].shape or not np.allclose(actual, source_timestamps[location], atol=1e-8, rtol=0.0):
            raise ValueError(f"manifest timestamps for {location} do not match Exp1-C selected frames")
    phantom = AnalyticCanonicalPhantom(domain).to(device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=phantom, motion=motion, psf_samples=psf_samples, seed=seed)
    model = build_oracle_motion_model(domain, phantom, device=device, psf_samples=psf_samples)
    caches = {item.dynamic_frame_id: build_target_psf_offset_cache(item, psf_samples=psf_samples, seed=seed + item.dynamic_frame_id) for item in observations}
    recovery = _exp1d_recovery_rows(args.exp1d_dir)
    sequences: dict[str, list] = defaultdict(list)
    for observation in observations:
        sequences[f"{observation.view}/{observation.slice_id}"].append(observation)
    rows: list[dict[str, object]] = []
    for location, sequence in sorted(sequences.items()):
        basis = _check_location_basis(by_location[location])
        ordered = sorted(sequence, key=lambda item: item.timestamp_s)
        rendered = [_location_variants(model, observation, caches[observation.dynamic_frame_id], motion, pixel_chunk=args.pixel_chunk) for observation in ordered]
        for observation, (variants, _) in zip(ordered, rendered):
            if not torch.allclose(variants["full"], observation.image[0], atol=1e-6, rtol=0.0):
                raise ValueError(f"reconstructed matched-CRN full GT image differs from Exp1-C target for {location}")
        full = torch.stack([item[0]["full"] for item in rendered])
        zero = torch.stack([item[0]["zero"] for item in rendered])
        dvf = torch.stack([item[1] for item in rendered])
        components = decompose_world_dvf(dvf, basis.row_direction, basis.column_direction, basis.normal)
        for index, component in enumerate(COMPONENTS):
            remove = torch.stack([item[0][f"remove_{component}"] for item in rendered])
            flip = torch.stack([item[0][f"flip_{component}"] for item in rendered])
            values = component_image_metrics(full, zero, remove, flip, components[..., index], dvf)
            joined = recovery.get(location, {}).get(component, {})
            rows.append({"location": location, "view": basis.view, "component": component, "frame_count": len(ordered), "psf_query_count_per_frame": int(dvf.shape[1] * dvf.shape[2]), "recovery_metrics_available": float(bool(joined)), "rms_ratio": joined.get("rms_ratio", float("nan")), "correlation": joined.get("correlation", float("nan")), "normalized_epe": joined.get("normalized_epe", float("nan")), **values})
    by_view = _aggregate_rows(rows, key="view")
    overall_rows = [dict(row, view="overall") for row in rows]
    overall = _aggregate_rows(overall_rows, key="view")
    _write_rows(args.output_dir / "component_image_observability_per_location.csv", rows)
    _write_rows(args.output_dir / "component_image_observability_by_view.csv", [*by_view, *overall])
    summary = {
        "experiment": "exp1e_component_image_observability",
        "input": {"exp1c_dir": str(input_dir), "exp1d_dir": None if args.exp1d_dir is None else str(args.exp1d_dir), "checkpoint_schema": checkpoint.get("checkpoint_schema"), "matched_crn": True},
        "n_locations": len(sequences),
        "per_view": by_view,
        "overall_per_component": overall,
        "recovery_metrics_joined": bool(recovery),
        "source_exp1c_summary": {"experiment": source_summary.get("experiment"), "mode": source_summary.get("mode")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-E respiratory component image observability audit")
    parser.add_argument("--input-dir", type=Path, required=True, help="matched-CRN Exp1-C oracle output directory")
    parser.add_argument("--exp1d-dir", type=Path, help="optional Exp1-D component recovery output directory")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--device", default="cpu")
    parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--pixel-chunk", type=int, default=256)
    args = parser.parse_args()
    summary = run(args)
    print(f"Exp1-E: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
