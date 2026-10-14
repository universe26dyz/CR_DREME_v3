#!/usr/bin/env python3
"""Offline slice-coordinate respiratory DVF audit for the saved Exp1-C oracle."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import geometry_from_manifest
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.render import GeometryFrame


COMPONENTS = ("row", "column", "normal")


def project_dvf_to_slice_components(dvf_world_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Project world-mm DVFs to the existing DICOM row/column/normal basis."""
    if dvf_world_mm.shape[-1] != 3:
        raise ValueError("world DVF must end in xyz")
    basis = torch.stack((row_direction, column_direction, normal)).to(dvf_world_mm)
    if basis.shape != (3, 3) or not torch.allclose(basis @ basis.T, torch.eye(3, dtype=basis.dtype, device=basis.device), atol=1e-4, rtol=0.0):
        raise ValueError("DICOM row/column/normal basis must be orthonormal")
    return torch.einsum("...c,kc->...k", dvf_world_mm, basis)


def component_dynamic_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """One scalar component's temporally mean-centred agreement metrics."""
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("component fields must have matched [time, ...] shapes")
    pred_dynamic = predicted - predicted.mean(dim=0, keepdim=True)
    target_dynamic = target - target.mean(dim=0, keepdim=True)
    pred_rms = pred_dynamic.square().mean().sqrt()
    target_rms = target_dynamic.square().mean().sqrt()
    endpoint_error = (pred_dynamic - target_dynamic).abs().mean()
    return {
        "dynamic_rms_mm": float(pred_rms),
        "gt_dynamic_rms_mm": float(target_rms),
        "rms_ratio": float(pred_rms / target_rms.clamp_min(1e-12)),
        "correlation": float((pred_dynamic * target_dynamic).sum() / (pred_dynamic.norm() * target_dynamic.norm()).clamp_min(1e-12)),
        "endpoint_error_mm": float(endpoint_error),
        "normalized_epe": float(endpoint_error / target_rms.clamp_min(1e-12)),
    }


def _prepare_output(path: Path, *, input_dir: Path, overwrite: bool) -> None:
    if path.resolve() == input_dir.resolve():
        raise ValueError("component audit output must differ from the Exp1-C input directory")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _frame_by_location(frames: list[GeometryFrame]) -> dict[str, list[GeometryFrame]]:
    grouped: dict[str, list[GeometryFrame]] = {}
    for frame in frames:
        grouped.setdefault(f"{frame.view}/{frame.slice_id}", []).append(frame)
    return {location: sorted(sequence, key=lambda frame: frame.timestamp_s) for location, sequence in grouped.items()}


def _check_location_basis(frames: list[GeometryFrame]) -> GeometryFrame:
    if not frames:
        raise ValueError("location has no geometry frames")
    reference = frames[0]
    for frame in frames[1:]:
        if not (torch.allclose(frame.row_direction, reference.row_direction, atol=1e-4, rtol=0.0) and torch.allclose(frame.column_direction, reference.column_direction, atol=1e-4, rtol=0.0) and torch.allclose(frame.normal, reference.normal, atol=1e-4, rtol=0.0)):
            raise ValueError(f"DICOM basis changes within fixed location {reference.view}/{reference.slice_id}")
    return reference


def _array_key(location: str, suffix: str) -> str:
    return f"oracle_{location.replace('/', '_')}_{suffix}"


def _component_location_metrics(prediction: np.ndarray, target: np.ndarray, frame: GeometryFrame) -> dict[str, float]:
    predicted = torch.as_tensor(prediction, dtype=torch.float32)
    truth = torch.as_tensor(target, dtype=torch.float32)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("center-support arrays must be matched [time, pixels, xyz]")
    projected_pred = project_dvf_to_slice_components(predicted, frame.row_direction.cpu(), frame.column_direction.cpu(), frame.normal.cpu())
    projected_gt = project_dvf_to_slice_components(truth, frame.row_direction.cpu(), frame.column_direction.cpu(), frame.normal.cpu())
    result: dict[str, float] = {"point_count": float(predicted.shape[1]), "frame_count": float(predicted.shape[0])}
    for index, component in enumerate(COMPONENTS):
        result.update({f"{component}_{name}": value for name, value in component_dynamic_metrics(projected_pred[..., index], projected_gt[..., index]).items()})
    return result


def _component_label(aggregate: dict[str, dict[str, float]], component: str, thresholds: dict[str, float]) -> str:
    value = lambda metric: aggregate.get(f"{component}_{metric}", {}).get("median", float("nan"))
    passed = thresholds["min_rms_ratio"] <= value("rms_ratio") <= thresholds["max_rms_ratio"] and value("correlation") >= thresholds["min_spatial_cosine"] and value("normalized_epe") <= thresholds["max_normalized_epe"]
    return "pass" if passed else "fail"


def _view_labels(rows: dict[str, dict[str, float]], thresholds: dict[str, float]) -> dict[str, dict[str, str]]:
    by_view: dict[str, dict[str, dict[str, float]]] = {}
    for location, values in rows.items():
        by_view.setdefault(location.split("/", 1)[0], {})[location] = values
    return {view: {component: _component_label(aggregate_location_metrics(values), component, thresholds) for component in COMPONENTS} for view, values in by_view.items()}


def _interpretation(labels: dict[str, str], by_view: dict[str, dict[str, str]]) -> str:
    sax = by_view.get("SAX")
    long_axis = [by_view[view] for view in ("2CH", "4CH") if view in by_view]
    if sax is not None and long_axis and any(view != sax for view in long_axis):
        return "view-dependent observability differences"
    if labels["row"] == "pass" and labels["column"] == "pass" and labels["normal"] == "fail":
        return "Motion error is dominated by through-plane component. In-plane motion is recovered but slice-normal motion is weakly constrained."
    if all(labels[component] == "pass" for component in COMPONENTS):
        return "Motion is recovered in acquisition coordinates. Remaining full-volume error likely comes from off-support extrapolation."
    if all(labels[component] == "fail" for component in COMPONENTS):
        return "Residual error is not limited to through-plane ambiguity. Further optimization or representation analysis is required."
    return "component recovery is mixed; inspect per-location and per-view component metrics"


def _read_location_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["location"] for row in csv.DictReader(handle)}


def _write_rows(path: Path, rows: dict[str, dict[str, float]]) -> None:
    fields = sorted({name for values in rows.values() for name in values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["location", *fields]); writer.writeheader()
        writer.writerows({"location": location, **values} for location, values in rows.items())


def _write_aggregate(path: Path, aggregate: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "median", "p10", "p90"]); writer.writeheader()
        writer.writerows({"metric": metric, **values} for metric, values in aggregate.items())


def run(args: argparse.Namespace) -> dict[str, object]:
    input_dir = args.input_dir
    checkpoint_path = input_dir / "oracle_model_last.pt"
    arrays_path = input_dir / "recovery_arrays.npz"
    summary_path = input_dir / "summary.json"
    locations_path = input_dir / "oracle_per_location.csv"
    for path in (checkpoint_path, arrays_path, summary_path, locations_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    _prepare_output(args.output_dir, input_dir=input_dir, overwrite=args.overwrite)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("branch") != "resp" or checkpoint.get("mode") != "oracle_resp_only" or "model_state_dict" not in checkpoint:
        raise ValueError("Exp1-D requires an oracle_resp_only respiratory checkpoint")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source_summary.get("mode") != "oracle_resp_only":
        raise ValueError("Exp1-D requires Exp1-C oracle_resp_only summary data")
    thresholds = source_summary["recovery_thresholds"]
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=torch.device("cpu"))
    geometry = _frame_by_location(frames)
    expected_locations = _read_location_names(locations_path)
    if set(geometry) != expected_locations:
        raise ValueError("manifest geometry locations do not match the Exp1-C oracle output")
    rows: dict[str, dict[str, float]] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for location in sorted(expected_locations):
            frame = _check_location_basis(geometry[location])
            prediction_key = _array_key(location, "center_support_pred_dvf_mm")
            target_key = _array_key(location, "center_support_gt_dvf_mm")
            if prediction_key not in arrays or target_key not in arrays:
                raise KeyError(f"missing Exp1-C center-support arrays for {location}")
            prediction, target = arrays[prediction_key], arrays[target_key]
            if prediction.shape[0] != len(geometry[location]):
                raise ValueError(f"frame count mismatch for {location}: arrays={prediction.shape[0]} geometry={len(geometry[location])}")
            rows[location] = _component_location_metrics(prediction, target, frame)
    aggregate = aggregate_location_metrics(rows)
    labels = {component: _component_label(aggregate, component, thresholds) for component in COMPONENTS}
    view_labels = _view_labels(rows, thresholds)
    summary = {
        "experiment": "exp1d_motion_component_audit",
        "input": {"exp1c_dir": str(input_dir), "checkpoint": str(checkpoint_path), "checkpoint_schema": checkpoint.get("checkpoint_schema")},
        "component_recovery": labels,
        "view_component_recovery": view_labels,
        "interpretation": _interpretation(labels, view_labels),
        "thresholds": thresholds,
        "component_aggregate": aggregate,
        "n_locations": len(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_rows(args.output_dir / "component_per_location.csv", rows)
    _write_aggregate(args.output_dir / "component_aggregate.csv", aggregate)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="C4 Exp1-D offline oracle respiratory component audit")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50)
    args = parser.parse_args()
    summary = run(args)
    print(f"Exp1-D: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
