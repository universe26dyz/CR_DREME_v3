"""Static-NeSVoR + S2V-DREME-style respiratory reference experiment.

This module intentionally does not import the vNext schedule or its auxiliary
losses.  It exposes the deterministic preprocessing/provenance utilities and
the six joint-training segment definitions used by the dedicated launcher.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import nibabel as nib
import numpy as np

from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend, lps_to_ras_world_matrix
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.respiratory_staged_vnext import respiratory_dvf_rms_metrics, respiratory_score_rms_metrics
from cardioresp4d.training.sampler import DynamicObservation
from cardioresp4d.training.stage_contract import stage_contract


REFERENCE_LOSS_WEIGHTS = {"image": 2e-6, "mbc_normalization": 1e-2, "smooth_resp": 1e-3}
FORMAL_NESVOR_OPTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("--registration", ("stack",)), ("--scanner-space", ()), ("--metric", ("none",)),
    ("--weight-transformation", ("1",)), ("--weight-deform", ("0.1",)), ("--weight-image", ("1",)),
    ("--weight-pixel-var-smooth", ("0",)), ("--output-resolution", ("1.0",)), ("--output-psf-factor", ("1",)),
    ("--no-slice-scale", ()), ("--single-precision", ()), ("--log2-hashmap-size", ("19",)),
    ("--batch-size", ("2000",)), ("--n-iter", ("10000",)), ("--n-samples", ("256",)),
    ("--n-inference-samples", ("128",)), ("--inference-batch-size", ("16000",)),
    ("--pixel-variance-weight", ("1",)), ("--coarsest-resolution", ("16",)), ("--finest-resolution", ("0.5",)),
    ("--level-scale", ("1.3819",)), ("--n-features-per-level", ("2",)), ("--depth", ("1",)),
    ("--width", ("64",)), ("--n-features-z", ("15",)), ("--n-features-slice", ("16",)),
    ("--weight-bias", ("100",)), ("--image-regularization", ("edge",)), ("--delta", ("0.2",)),
    ("--learning-rate", ("0.005",)), ("--gamma", ("0.33",)), ("--milestones", ("0.5", "0.75", "0.9")), ("--seed", ("20260823",)),
)


@dataclass(frozen=True)
class TemporalMeanLocation:
    view: str
    slice_id: str
    image: torch.Tensor
    frame_count: int
    timestamps_s: tuple[float, ...]
    geometry: dict[str, Any]
    geometry_hash: str
    qc_provenance: tuple[str, ...]


@dataclass(frozen=True)
class CorrectedStaticPose:
    view: str
    slice_id: str
    source_stack_file: str
    source_stack_index: int
    original_geometry: dict[str, Any]
    corrected_geometry: dict[str, Any]
    correction_transform_lps: list[list[float]]
    corrected_slice_file: str


def _geometry(observation: DynamicObservation) -> dict[str, Any]:
    return {
        "center_mm": [float(x) for x in observation.center_mm.cpu()],
        "row_direction": [float(x) for x in observation.row_direction.cpu()],
        "column_direction": [float(x) for x in observation.column_direction.cpu()],
        "normal": [float(x) for x in observation.normal.cpu()],
        "pixel_spacing_mm": [float(x) for x in observation.pixel_spacing_mm.cpu()],
        "slice_thickness_mm": float(observation.slice_thickness_mm),
        "image_shape": list(observation.image.shape),
    }


def _geometry_hash(geometry: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_temporal_mean_locations(observations: Iterable[DynamicObservation], *, geometry_tolerance: float = 1e-5) -> tuple[TemporalMeanLocation, ...]:
    """Mean all QC-valid frames at each fixed view/location; never drops a valid group."""
    if geometry_tolerance <= 0:
        raise ValueError("geometry_tolerance must be positive")
    groups: dict[tuple[str, str], list[DynamicObservation]] = {}
    for observation in observations:
        if observation.qc_valid:
            groups.setdefault((observation.view.upper(), observation.slice_id), []).append(observation)
    result = []
    for (view, slice_id), frames in sorted(groups.items()):
        ordered = sorted(frames, key=lambda item: item.timestamp_s)
        first = ordered[0]
        if any(item.image.shape != first.image.shape for item in ordered):
            raise ValueError(f"image shape changes within location {view}/{slice_id}")
        geometry = _geometry(first)
        for item in ordered[1:]:
            other = _geometry(item)
            for key in ("center_mm", "row_direction", "column_direction", "normal", "pixel_spacing_mm", "slice_thickness_mm"):
                if not torch.allclose(torch.as_tensor(other[key]), torch.as_tensor(geometry[key]), atol=geometry_tolerance, rtol=0.0):
                    raise ValueError(f"geometry changes within location {view}/{slice_id}: {key}")
        stacked = torch.stack([item.image for item in ordered])
        if not torch.isfinite(stacked).all():
            raise ValueError(f"non-finite image in location {view}/{slice_id}")
        result.append(TemporalMeanLocation(view, slice_id, stacked.mean(dim=0), len(ordered), tuple(float(item.timestamp_s) for item in ordered), geometry, _geometry_hash(geometry), tuple(item.qc_reason for item in ordered)))
    if not result:
        raise ValueError("no QC-valid observations available for temporal means")
    return tuple(result)


def export_temporal_mean_stacks(locations: Sequence[TemporalMeanLocation], output_dir: str | Path) -> Path:
    """Export view-wise NIfTI mean stacks plus geometry required by NeSVoR."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for view in ("SAX", "2CH", "4CH"):
        selected = [item for item in locations if item.view == view]
        if not selected:
            continue
        shapes = {tuple(item.image.shape) for item in selected}
        if len(shapes) != 1:
            raise ValueError(f"cannot safely stack unequal mean image shapes for {view}")
        first_geometry = selected[0].geometry
        thicknesses = {float(item.geometry["slice_thickness_mm"]) for item in selected}
        if len(thicknesses) != 1:
            raise ValueError(f"cannot safely represent inconsistent slice thickness for {view}")
        normal = torch.as_tensor(first_geometry["normal"], dtype=torch.float64)
        normal = normal / torch.linalg.vector_norm(normal)
        row = torch.as_tensor(first_geometry["row_direction"], dtype=torch.float64)
        column = torch.as_tensor(first_geometry["column_direction"], dtype=torch.float64)
        for item in selected:
            item_row = torch.as_tensor(item.geometry["row_direction"], dtype=torch.float64)
            item_column = torch.as_tensor(item.geometry["column_direction"], dtype=torch.float64)
            item_normal = torch.as_tensor(item.geometry["normal"], dtype=torch.float64)
            if abs(float(torch.dot(item_normal, normal))) < 1.0 - 1e-5:
                raise ValueError(f"cannot safely stack nonparallel locations for {view}")
            if not torch.allclose(item_row, row, atol=1e-5, rtol=0.0) or not torch.allclose(item_column, column, atol=1e-5, rtol=0.0) or not torch.allclose(torch.as_tensor(item.geometry["pixel_spacing_mm"]), torch.as_tensor(first_geometry["pixel_spacing_mm"]), atol=1e-5, rtol=0.0):
                raise ValueError(f"cannot safely stack incompatible in-plane geometry for {view}")
        selected = sorted(selected, key=lambda item: float(torch.dot(torch.as_tensor(item.geometry["center_mm"], dtype=torch.float64), normal)))
        centers = [torch.as_tensor(item.geometry["center_mm"], dtype=torch.float64) for item in selected]
        projections = [float(torch.dot(center, normal)) for center in centers]
        step = float(np.median(np.diff(projections))) if len(projections) > 1 else float(next(iter(thicknesses)))
        if step <= 0 or (len(projections) > 2 and not np.allclose(np.diff(projections), step, atol=1e-4, rtol=0.0)):
            raise ValueError(f"cannot safely represent irregular slice positions for {view}")
        _, height, width = selected[0].image.shape
        spacing_row, spacing_column = first_geometry["pixel_spacing_mm"]
        origin = centers[0] - (width - 1) / 2 * spacing_column * row - (height - 1) / 2 * spacing_row * column
        lps = np.eye(4); lps[:3, 0] = (row * spacing_column).numpy(); lps[:3, 1] = (column * spacing_row).numpy(); lps[:3, 2] = (normal * step).numpy(); lps[:3, 3] = origin.numpy()
        ras = np.diag([-1., -1., 1., 1.]) @ lps
        path = output / f"mean_stack_{view}.nii.gz"
        volume = torch.stack([item.image[0].cpu() for item in selected]).permute(2, 1, 0).numpy().astype(np.float32, copy=False)
        nib.save(nib.Nifti1Image(volume, ras), path)
        for index, item in enumerate(selected):
            records.append({"view": item.view, "slice_id": item.slice_id, "stack_file": path.name, "stack_index": index, "frame_count": item.frame_count, "timestamp_min_s": min(item.timestamps_s), "timestamp_max_s": max(item.timestamps_s), "image_shape": item.geometry["image_shape"], "geometry": item.geometry, "geometry_hash": item.geometry_hash, "qc_provenance": item.qc_provenance, "stack_thickness_mm": next(iter(thicknesses))})
    manifest = output / "mean_slice_manifest.json"
    manifest.write_text(json.dumps({"policy": "qc_valid_per_location_exact_temporal_mean", "locations": records}, indent=2, sort_keys=True))
    return manifest


def nesvor_reconstruct_command(mean_stacks: Sequence[str | Path], *, output_dir: str | Path, thicknesses_mm: Sequence[float], smoke: bool = False) -> list[str]:
    """Create the pinned static reconstruction command without running it."""
    if not mean_stacks:
        raise ValueError("NeSVoR reconstruction requires at least one mean stack")
    if len(thicknesses_mm) != len(mean_stacks) or any(not math.isfinite(value) or value <= 0 for value in thicknesses_mm):
        raise ValueError("each mean stack requires one finite acquisition thickness")
    output = Path(output_dir)
    command = ["nesvor", "reconstruct", "--input-stacks", *[str(path) for path in mean_stacks], "--thicknesses", *[str(value) for value in thicknesses_mm], "--output-volume", str(output / "static_reference.nii.gz"), "--output-model", str(output / "model.pt"), "--output-json", str(output / "nesvor_effective_args.json"), "--output-slices", str(output / "slices"), "--output-log", str(output / "nesvor.log"), "--output-intensity-mean", "0"]
    for key, values in FORMAL_NESVOR_OPTIONS:
        command.extend((key, *values))
    if smoke:
        command[command.index("--n-iter") + 1] = "2"
        command[command.index("--n-samples") + 1] = "8"
        command[command.index("--n-inference-samples") + 1] = "8"
        command[command.index("--inference-batch-size") + 1] = "64"
    return command


@dataclass(frozen=True)
class ReferenceSegment:
    name: str
    stage: str
    steps: int
    active_respiratory_levels: int
    owners: frozenset[str]


def reference_schedule() -> tuple[ReferenceSegment, ...]:
    return tuple(
        ReferenceSegment(f"L{level}_{kind}", f"stage2{letter}", steps, level, frozenset({"respiratory_encoder", "respiratory_mbc"}) | (frozenset({"canonical"}) if kind == "joint" else frozenset()))
        for level, letter in ((1, "a"), (2, "b"), (3, "c"))
        for kind, steps in (("warm", 200), ("joint", 600))
    )


REFERENCE_CHECKPOINT_BOUNDARIES = tuple(f"checkpoint_{segment.name}_end.pt" for segment in reference_schedule())


def configure_reference_segment(model: torch.nn.Module, segment: ReferenceSegment, *, optimizer_config: Mapping[str, float] | None = None) -> torch.optim.Adam:
    """Fresh joint optimizer with no cardiac/uncertainty ownership."""
    settings = {"canonical_lr": 1e-4, "respiratory_encoder_lr": 5e-4, "respiratory_mbc_lr": 5e-4} if optimizer_config is None else dict(optimizer_config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.respiratory_mbc.set_active_levels(segment.active_respiratory_levels)
    groups: list[dict[str, Any]] = []
    if "canonical" in segment.owners:
        parameters = list(model.canonical.parameters())
        for parameter in parameters: parameter.requires_grad_(True)
        groups.append({"params": parameters, "lr": float(settings["canonical_lr"]), "name": "canonical"})
    if "respiratory_encoder" in segment.owners:
        parameters = list(model.film_encoder.parameters())
        for parameter in parameters: parameter.requires_grad_(True)
        groups.append({"params": parameters, "lr": float(settings["respiratory_encoder_lr"]), "name": "respiratory_encoder"})
    if "respiratory_mbc" in segment.owners:
        parameters = [parameter for level in model.respiratory_mbc.levels[:segment.active_respiratory_levels] for parameter in level.parameters()]
        for parameter in parameters: parameter.requires_grad_(True)
        groups.append({"params": parameters, "lr": float(settings["respiratory_mbc_lr"]), "name": "respiratory_mbc"})
    return torch.optim.Adam(groups)


def reference_loss(components: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """The complete reference objective; no vNext temporal auxiliary is inherited."""
    return components["data"] + REFERENCE_LOSS_WEIGHTS["image"] * components["image"] + REFERENCE_LOSS_WEIGHTS["mbc_normalization"] * components["mbc_normalization"] + REFERENCE_LOSS_WEIGHTS["smooth_resp"] * components["smooth_resp"]


def reference_score_diagnostics(scores_by_location: tuple[torch.Tensor, ...]) -> dict[str, float]:
    metrics = respiratory_score_rms_metrics(scores_by_location)
    metrics["respiratory_score_static_dynamic_ratio"] = math.inf if metrics["respiratory_score_dynamic_rms"] == 0 else metrics["respiratory_score_static_rms"] / metrics["respiratory_score_dynamic_rms"]
    return metrics


def reference_dvf_diagnostics(dvfs_by_location: tuple[torch.Tensor, ...]) -> dict[str, float]:
    metrics = respiratory_dvf_rms_metrics(dvfs_by_location)
    metrics["respiratory_dvf_static_dynamic_ratio"] = math.inf if metrics["respiratory_dvf_dynamic_rms_mm"] == 0 else metrics["respiratory_dvf_static_rms_mm"] / metrics["respiratory_dvf_dynamic_rms_mm"]
    return metrics


def strict_canonical_state_from_static_checkpoint(payload: Mapping[str, Any], expected: Mapping[str, torch.Tensor], *, internal_bounds_world_mm: torch.Tensor | None = None, expected_architecture: Any | None = None, expected_encoding_backend: str | None = None, world_to_internal_transform: torch.Tensor | None = None, tolerance_mm: float = 1e-5) -> dict[str, torch.Tensor]:
    """Admit only a schema explicitly containing an exact INR state dictionary."""
    # Pinned NeSVoR 0.6.0rc1 writes ``{"model": inr.state_dict(), ...}``
    # in ``nesvor.cli.io.save_volume``.  ``inr_state_dict`` is accepted only
    # for a deliberately exported equivalent artifact.
    candidate = payload.get("model", payload.get("inr_state_dict"))
    if not isinstance(candidate, Mapping):
        raise ValueError("static checkpoint has no auditable inr_state_dict; use the volume-MSE warm start")
    if set(candidate) != set(expected) or any(not isinstance(candidate[key], torch.Tensor) or candidate[key].shape != expected[key].shape for key in expected):
        raise ValueError("static checkpoint INR state is not an exact canonical key/shape match")
    if internal_bounds_world_mm is not None:
        bbox = candidate.get("bounding_box")
        if not isinstance(bbox, torch.Tensor) or bbox.shape != (2, 3) or not torch.allclose(bbox.detach().cpu(), internal_bounds_world_mm.detach().cpu(), atol=tolerance_mm, rtol=0.0):
            raise ValueError("static checkpoint bounding_box does not match the internal INR domain")
    if expected_architecture is not None:
        static_args = payload.get("args")
        for name in ("coarsest_resolution", "finest_resolution", "level_scale", "n_features_per_level", "log2_hashmap_size", "n_features_z", "width", "depth"):
            if static_args is None or not hasattr(static_args, name) or getattr(static_args, name) != getattr(expected_architecture, name):
                raise ValueError(f"static checkpoint architecture does not match dynamic canonical: {name}")
    if expected_encoding_backend is not None and detect_checkpoint_encoding_backend(dict(candidate)) != expected_encoding_backend:
        raise ValueError("static checkpoint encoding backend does not match the constructed canonical INR")
    if world_to_internal_transform is not None:
        expected_bridge = lps_to_ras_world_matrix(dtype=world_to_internal_transform.dtype, device=world_to_internal_transform.device)
        if world_to_internal_transform.shape != (3, 3) or not torch.equal(world_to_internal_transform, expected_bridge):
            raise ValueError("static direct admission requires the fixed LPS-to-RAS coordinate bridge")
    return {key: candidate[key] for key in expected}


def _lps_affine_from_geometry(geometry: Mapping[str, Any]) -> np.ndarray:
    _, height, width = geometry["image_shape"]
    row, column, center = (np.asarray(geometry[key], dtype=float) for key in ("row_direction", "column_direction", "center_mm"))
    spacing_row, spacing_column = geometry["pixel_spacing_mm"]
    normal = np.asarray(geometry["normal"], dtype=float)
    affine = np.eye(4)
    affine[:3, 0] = row * spacing_column; affine[:3, 1] = column * spacing_row; affine[:3, 2] = normal * float(geometry["slice_thickness_mm"])
    affine[:3, 3] = center - (width - 1) / 2 * affine[:3, 0] - (height - 1) / 2 * affine[:3, 1]
    return affine


def _corrected_geometry_from_slice(path: Path, original: Mapping[str, Any]) -> dict[str, Any]:
    image = nib.load(str(path)); affine_lps = np.diag([-1., -1., 1., 1.]) @ image.affine
    width, height, depth = image.shape[:3]
    axes = affine_lps[:3, :3]
    row = axes[:, 0] / np.linalg.norm(axes[:, 0]); column = axes[:, 1] / np.linalg.norm(axes[:, 1]); normal = axes[:, 2] / np.linalg.norm(axes[:, 2])
    center = nib.affines.apply_affine(affine_lps, [(width - 1) / 2, (height - 1) / 2, (depth - 1) / 2])
    return {"center_mm": center.tolist(), "row_direction": row.tolist(), "column_direction": column.tolist(), "normal": normal.tolist(), "pixel_spacing_mm": list(original["pixel_spacing_mm"]), "slice_thickness_mm": float(original["slice_thickness_mm"]), "image_shape": list(original["image_shape"])}


def build_corrected_pose_map(mean_slice_manifest: str | Path, corrected_slices_dir: str | Path) -> dict[tuple[str, str], CorrectedStaticPose]:
    """Map NeSVoR's ordered ``output_slices/<index>.nii.gz`` back to mean locations.

    Pinned NeSVoR saves its post-registration ``output_slices`` in input order;
    the mean manifest records the command's stack order and intra-stack index.
    The NIfTI affine is the corrected pose and is converted RAS→LPS here.
    """
    manifest_path = Path(mean_slice_manifest); records = json.loads(manifest_path.read_text())["locations"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records: groups.setdefault(record["stack_file"], []).append(record)
    ordered = [record for stack in sorted(groups) for record in sorted(groups[stack], key=lambda item: int(item["stack_index"]))]
    corrected_dir = Path(corrected_slices_dir); files = [corrected_dir / f"{index}.nii.gz" for index in range(len(ordered))]
    if len(list(corrected_dir.glob("[0-9]*.nii*"))) != len(ordered) or any(not path.is_file() for path in files):
        raise ValueError("corrected slice count does not exactly match the mean-slice locations")
    result: dict[tuple[str, str], CorrectedStaticPose] = {}
    for record, path in zip(ordered, files):
        key = (str(record["view"]).upper(), str(record["slice_id"]))
        if key in result:
            raise ValueError(f"ambiguous corrected pose mapping for {key[0]}/{key[1]}")
        corrected = _corrected_geometry_from_slice(path, record["geometry"])
        transform = _lps_affine_from_geometry(corrected) @ np.linalg.inv(_lps_affine_from_geometry(record["geometry"]))
        result[key] = CorrectedStaticPose(key[0], key[1], str(record["stack_file"]), int(record["stack_index"]), dict(record["geometry"]), corrected, transform.tolist(), str(path))
    return result


def apply_corrected_static_poses(observations: Iterable[DynamicObservation], pose_map: Mapping[tuple[str, str], CorrectedStaticPose]) -> list[DynamicObservation]:
    """Apply one immutable static corrected pose to every valid temporal frame."""
    result = []
    for observation in sorted(observations, key=lambda item: (item.view.upper(), item.slice_id, item.timestamp_s, item.dynamic_frame_id)):
        if not observation.qc_valid:
            result.append(observation); continue
        pose = pose_map.get((observation.view.upper(), observation.slice_id))
        if pose is None:
            raise ValueError(f"missing corrected static pose for {observation.view}/{observation.slice_id}")
        geometry = pose.corrected_geometry
        result.append(replace(observation, center_mm=torch.tensor(geometry["center_mm"], dtype=observation.center_mm.dtype, device=observation.center_mm.device), row_direction=torch.tensor(geometry["row_direction"], dtype=observation.row_direction.dtype, device=observation.row_direction.device), column_direction=torch.tensor(geometry["column_direction"], dtype=observation.column_direction.dtype, device=observation.column_direction.device), normal=torch.tensor(geometry["normal"], dtype=observation.normal.dtype, device=observation.normal.device)))
    return result


def corrected_pose_containment_audit(poses: Mapping[tuple[str, str], CorrectedStaticPose | Mapping[str, Any]], *, lower_world_mm: Sequence[float], upper_world_mm: Sequence[float], tolerance_mm: float = 1e-5) -> dict[str, Any]:
    """Fail-closed corrected in-plane-corner admission using DynamicObservation geometry."""
    lower, upper = np.asarray(lower_world_mm, dtype=float), np.asarray(upper_world_mm, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower) or tolerance_mm < 0:
        raise ValueError("canonical containment bounds/tolerance are invalid")
    min_margin = math.inf; max_margin = -math.inf
    for (view, slice_id), value in sorted(poses.items()):
        geometry = value.corrected_geometry if isinstance(value, CorrectedStaticPose) else value
        _, height, width = geometry["image_shape"]
        center, row, column = (np.asarray(geometry[key], dtype=float) for key in ("center_mm", "row_direction", "column_direction"))
        spacing_row, spacing_column = geometry["pixel_spacing_mm"]
        corners = np.asarray([center + (u - (width - 1) / 2.) * spacing_column * row + (v - (height - 1) / 2.) * spacing_row * column for u in (0., width - 1.) for v in (0., height - 1.)])
        below, above = lower - corners.min(axis=0), corners.max(axis=0) - upper
        violation = np.maximum(np.maximum(below, above), 0.)
        if np.any(violation > tolerance_mm):
            raise ValueError(f"corrected pose outside canonical domain for {view}/{slice_id}: corrected_min={corners.min(axis=0).tolist()} corrected_max={corners.max(axis=0).tolist()} canonical_min={lower.tolist()} canonical_max={upper.tolist()} violation_mm={violation.tolist()}")
        margins = np.minimum(corners - lower, upper - corners)
        min_margin = min(min_margin, float(margins.min())); max_margin = max(max_margin, float(margins.max()))
    if not poses:
        raise ValueError("corrected pose containment requires at least one location")
    return {"policy": "corrected_slice_corners", "tolerance_mm": tolerance_mm, "location_count": len(poses), "minimum_margin_to_boundary_mm": min_margin, "maximum_margin_to_boundary_mm": max_margin, "passed": True}


def corrected_pose_internal_inr_containment_audit(poses: Mapping[tuple[str, str], CorrectedStaticPose | Mapping[str, Any]], *, internal_lower_world_mm: Sequence[float], internal_upper_world_mm: Sequence[float], world_to_internal_transform: torch.Tensor, tolerance_mm: float = 1e-5) -> dict[str, Any]:
    """Fail closed when corrected LPS slice support exceeds inherited INR RAS support.

    The tolerance is explicit because the NeSVoR corrected affine and saved INR
    box are float32 artifacts; it is not a domain expansion.
    """
    lower, upper = np.asarray(internal_lower_world_mm, dtype=float), np.asarray(internal_upper_world_mm, dtype=float)
    transform = np.asarray(torch.as_tensor(world_to_internal_transform, dtype=torch.float64).cpu(), dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or transform.shape != (3, 3) or np.any(upper <= lower) or tolerance_mm < 0:
        raise ValueError("internal INR containment bounds/transform/tolerance are invalid")
    minimum_margin = math.inf
    for (view, slice_id), value in sorted(poses.items()):
        geometry = value.corrected_geometry if isinstance(value, CorrectedStaticPose) else value
        _, height, width = geometry["image_shape"]
        center, row, column = (np.asarray(geometry[key], dtype=float) for key in ("center_mm", "row_direction", "column_direction"))
        spacing_row, spacing_column = geometry["pixel_spacing_mm"]
        lps_corners = np.asarray([center + (u - (width - 1) / 2.) * spacing_column * row + (v - (height - 1) / 2.) * spacing_row * column for u in (0., width - 1.) for v in (0., height - 1.)])
        ras_corners = lps_corners @ transform.T
        below, above = lower - ras_corners.min(axis=0), ras_corners.max(axis=0) - upper
        violation = np.maximum(np.maximum(below, above), 0.)
        if np.any(violation > tolerance_mm):
            raise ValueError(f"corrected pose outside static INR support for {view}/{slice_id}: corrected_lps_min={lps_corners.min(axis=0).tolist()} corrected_lps_max={lps_corners.max(axis=0).tolist()} converted_ras_min={ras_corners.min(axis=0).tolist()} converted_ras_max={ras_corners.max(axis=0).tolist()} static_inr_ras_min={lower.tolist()} static_inr_ras_max={upper.tolist()} violation_mm={violation.tolist()}")
        minimum_margin = min(minimum_margin, float(np.minimum(ras_corners - lower, upper - ras_corners).min()))
    if not poses:
        raise ValueError("internal INR containment requires at least one location")
    return {"policy": "corrected_slice_corners_lps_to_ras", "tolerance_mm": tolerance_mm, "location_count": len(poses), "minimum_margin_to_boundary_mm": minimum_margin, "passed": True}


def write_static_reference_manifest(*, output_dir: str | Path, source_manifest: str | Path, qc_table: str | Path, mean_slice_manifest: str | Path, command: Sequence[str], executable: str, version: str, mode: str) -> Path:
    """Close the static artifact/provenance chain after NeSVoR has completed."""
    output = Path(output_dir); mean_path = Path(mean_slice_manifest)
    if mode not in {"formal", "smoke"} or "--output-intensity-mean" not in command or command[command.index("--output-intensity-mean") + 1] != "0":
        raise ValueError("static manifest requires the provenance-proven output_intensity_mean=0 contract")
    pose_map = build_corrected_pose_map(mean_path, output / "slices")
    pose_path = output / "corrected_pose_manifest.json"
    pose_path.write_text(json.dumps({"mapping_policy": "NeSVoR output_slices follows deterministic input-stack then intra-stack order", "poses": {f"{view}/{slice_id}": asdict(pose) for (view, slice_id), pose in sorted(pose_map.items())}}, indent=2) + "\n")
    mean = json.loads(mean_path.read_text()); stack_paths = sorted({item["stack_file"] for item in mean["locations"]})
    def artifact(path: Path) -> dict[str, str]:
        if not path.is_file(): raise ValueError(f"required static output is missing: {path}")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    outputs = {name: artifact(output / filename) for name, filename in {"model": "model.pt", "volume": "static_reference.nii.gz", "effective_args": "nesvor_effective_args.json", "log": "nesvor.log"}.items()}
    stacks = []
    for stack in stack_paths:
        locations = sorted((item for item in mean["locations"] if item["stack_file"] == stack), key=lambda item: int(item["stack_index"]))
        stacks.append({"path": str(mean_path.parent / stack), "sha256": hashlib.sha256((mean_path.parent / stack).read_bytes()).hexdigest(), "view": locations[0]["view"], "location_count": len(locations), "slice_ids_in_stack_order": [item["slice_id"] for item in locations], "thickness_mm": locations[0]["stack_thickness_mm"], "geometry_hashes": [item["geometry_hash"] for item in locations]})
    payload = {"source_manifest_sha256": hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest(), "qc_table_sha256": hashlib.sha256(Path(qc_table).read_bytes()).hexdigest(), "mean_slice_manifest": artifact(mean_path), "mean_stacks": stacks, "nesvor": {"executable": executable, "version": version, "command": list(command), "mode": mode, "pinned_commit": "2e96a91bdd30174210caea911e03a2778c65adbe"}, "outputs": outputs, "corrected_pose_manifest": artifact(pose_path), "pose_location_count": len(pose_map), "scientific_contract": {"registration": "stack", "transformation_optimization": "enabled", "output_intensity_mean": 0, "formal_n_iter": 10000, "formal_n_samples": 256}}
    manifest = output / "static_reference_manifest.json"; manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def load_corrected_pose_map(path: str | Path) -> dict[tuple[str, str], CorrectedStaticPose]:
    payload = json.loads(Path(path).read_text())
    result = {}
    for key, value in payload["poses"].items():
        view, slice_id = key.split("/", 1); pose = CorrectedStaticPose(**value)
        if (view, slice_id) in result or pose.view != view or pose.slice_id != slice_id:
            raise ValueError("ambiguous corrected pose manifest")
        result[(view, slice_id)] = pose
    return result


def canonical_init_provenance(static_model_path: str | Path, *, mode: str) -> dict[str, str]:
    if mode not in {"direct_exact_state", "direct_exact_state_with_coordinate_bridge", "static_volume_mse_warm_start"}:
        raise ValueError("unknown canonical initialization mode")
    path = Path(static_model_path)
    return {"mode": mode, "static_model_path": str(path), "static_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def warm_start_canonical_from_static_volume(model, sample_points_world_mm: torch.Tensor, target_intensity: torch.Tensor, *, steps: int, learning_rate: float = 1e-4) -> dict[str, float]:
    """Deterministically fit a compatible canonical INR to an exported static volume.

    This fallback is intentionally explicit: callers must supply physical sample
    points and intensity targets from a verified NeSVoR volume export.
    """
    if steps <= 0 or learning_rate <= 0 or sample_points_world_mm.ndim != 2 or sample_points_world_mm.shape[-1] != 3 or target_intensity.shape != sample_points_world_mm.shape[:1]:
        raise ValueError("static-volume warm start requires finite [N,3] points, [N] targets, and positive steps")
    if not torch.isfinite(sample_points_world_mm).all() or not torch.isfinite(target_intensity).all():
        raise ValueError("static-volume warm start inputs must be finite")
    optimizer = torch.optim.Adam(model.canonical.parameters(), lr=learning_rate)
    model.zero_grad(set_to_none=True); losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.canonical(sample_points_world_mm).reshape_as(target_intensity) - target_intensity).square().mean()
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    return {"steps_run": steps, "learning_rate": learning_rate, "loss_first": losses[0], "loss_last": losses[-1]}


def static_volume_samples(volume_path: str | Path, *, max_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a static NIfTI in RAS and return deterministic physical LPS samples."""
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    image = nib.load(str(volume_path)); volume = np.asanyarray(image.dataobj, dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("static warm start requires one finite 3-D NIfTI volume")
    indices = np.stack(np.unravel_index(np.arange(volume.size), volume.shape), axis=-1)
    if indices.shape[0] > max_points:
        indices = indices[np.linspace(0, indices.shape[0] - 1, max_points).round().astype(np.int64)]
    ras = nib.affines.apply_affine(image.affine, indices)
    ras[:, :2] *= -1.0  # NIfTI RAS back to CR_DREME/NeSVoR LPS world-mm.
    targets = volume[tuple(indices.T)]
    return torch.as_tensor(ras, device=device, dtype=torch.float32), torch.as_tensor(targets, device=device, dtype=torch.float32)


def validate_reference_config(config: Mapping[str, Any]) -> None:
    """Reject accidental inheritance of vNext scientific settings."""
    training = config.get("training", {})
    if training.get("observations_per_update") != 32 or training.get("loss_weights") != {"image": 2e-6, "mbc_normalization": 1e-2, "smooth_resp": 1e-3} or training.get("optimizer") != {"canonical_lr": 1e-4, "respiratory_encoder_lr": 5e-4, "respiratory_mbc_lr": 5e-4}:
        raise ValueError("reference experiment requires fixed batch size and explicit S2V-DREME loss weights")
    schedule = training.get("schedule")
    expected = [{"level": 1, "warm_steps": 200, "joint_steps": 600}, {"level": 2, "warm_steps": 200, "joint_steps": 600}, {"level": 3, "warm_steps": 200, "joint_steps": 600}]
    if schedule != expected:
        raise ValueError("reference experiment requires the fixed L1/L2/L3 warm-to-joint schedule")
    disabled = set(training.get("disabled", ()))
    required_disabled = {"pca_waveform", "pca_warmup", "hard_score_centering", "per_location_dc_subtraction", "zero_mean_score", "groupwise_encoder", "cardiac", "uncertainty"}
    if not required_disabled <= disabled:
        raise ValueError("reference experiment must explicitly disable PCA/DC/groupwise/cardiac/uncertainty auxiliaries")
    fallback = config.get("static_reference", {}).get("canonical_initialization", {}).get("fallback", {})
    if fallback.get("enabled") is not True or fallback.get("mode") not in {"calibration_required", "fixed_steps"} or float(fallback.get("learning_rate", 0.)) != 1e-4 or int(fallback.get("max_points", 0)) <= 0:
        raise ValueError("reference experiment requires an explicit static-volume fallback policy")
    if fallback["mode"] == "fixed_steps" and int(fallback.get("steps", 0)) <= 0:
        raise ValueError("fixed-step static-volume fallback requires pinned positive steps")


def deterministic_diagnostic_cohort(sampler) -> tuple[tuple[str, str, tuple[DynamicObservation, ...]], ...]:
    return tuple((view, slice_id, tuple(sorted(frames, key=lambda item: item.timestamp_s))) for view in sorted(sampler._grouped) for slice_id, frames in sorted(sampler._grouped[view].items()) if frames)


@contextmanager
def diagnostic_rng_context(seed: int = 20260823):
    """Isolate repeatable diagnostic PSF draws from the training RNG stream."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def static_reference_diagnostics(model, sampler) -> dict[str, float]:
    """Deterministic, read-only static-canonical reconstruction audit before motion."""
    cohort = deterministic_diagnostic_cohort(sampler); sums: dict[str, float] = {}; counts: dict[str, int] = {}
    with diagnostic_rng_context(), torch.no_grad():
        for view, _, frames in cohort:
            for observation in frames:
                height, width = observation.image.shape[-2:]; linear = torch.arange(height * width, device=observation.image.device)
                uv = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)
                error = (model.predict(observation, uv, "stage1")["predicted_intensity"] - observation.image[0].flatten()).square()
                sums[view] = sums.get(view, 0.) + float(error.sum()); counts[view] = counts.get(view, 0) + error.numel()
    result = {f"reconstruction_mse_{view}": sums[view] / counts[view] for view in sorted(sums)}
    result["reconstruction_mse_full"] = sum(sums.values()) / sum(counts.values())
    result["diagnostic_location_count"] = len(cohort); result["diagnostic_frame_count"] = sum(len(frames) for _, _, frames in cohort)
    return result


def boundary_motion_diagnostics(model, sampler, *, stage: str) -> dict[str, float]:
    """Read-only full-span, per-location diagnostics shared across all boundaries."""
    cohort = deterministic_diagnostic_cohort(sampler)
    levels = stage_contract(stage).active_respiratory_levels
    with diagnostic_rng_context(), torch.no_grad():
        score_values = tuple(torch.cat([model.encode_motion_scores(item)["resp_scores"][:, :levels] for item in frames]) for _, _, frames in cohort)
        grid, _ = model._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (4, 4, 4))
        dvf_values = tuple(torch.stack([ScoreWeightedMBCField(model.respiratory_mbc, model.encode_motion_scores(item)["resp_scores"][:, :levels])(grid)[0] for item in frames]) for _, _, frames in cohort)
        reconstruction_sum: dict[str, float] = {}
        reconstruction_count: dict[str, int] = {}
        for view, _, frames in cohort:
            for observation in frames:
                height, width = observation.image.shape[-2:]
                linear = torch.arange(height * width, device=observation.image.device)
                uv = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)
                prediction = model.predict(observation, uv, stage)["predicted_intensity"]
                squared_error = (prediction - observation.image[0].flatten()).square()
                reconstruction_sum[view] = reconstruction_sum.get(view, 0.) + float(squared_error.sum())
                reconstruction_count[view] = reconstruction_count.get(view, 0) + squared_error.numel()
    reconstruction = {f"reconstruction_mse_{view}": reconstruction_sum[view] / reconstruction_count[view] for view in sorted(reconstruction_sum)}
    reconstruction["reconstruction_mse_full"] = sum(reconstruction_sum.values()) / sum(reconstruction_count.values())
    return {**reference_score_diagnostics(score_values), **reference_dvf_diagnostics(dvf_values), **reconstruction, "diagnostic_location_count": len(cohort), "diagnostic_frame_count": sum(len(frames) for _, _, frames in cohort)}


class S2VDREMEReferenceRunner:
    """Dedicated mini-batch runner for the paper-style warm-to-joint schedule.

    It owns neither cardiac nor uncertainty parameters and deliberately avoids
    ``UnifiedProgressiveTrainer`` because that trainer's temporal auxiliaries
    are not part of this reference protocol.
    """

    def __init__(self, model, sampler, *, pixel_samples: int = 256, observations_per_update: int = 32, optimizer_config: Mapping[str, float] | None = None) -> None:
        if observations_per_update != 32:
            raise ValueError("S2V-DREME reference protocol requires 32 observations/update")
        if pixel_samples <= 0:
            raise ValueError("pixel_samples must be positive")
        self.model, self.sampler = model, sampler
        self.pixel_samples, self.observations_per_update, self.optimizer_config = pixel_samples, observations_per_update, dict(optimizer_config or {})
        self.global_step = 0
        self.completed: list[dict[str, Any]] = []
        self.last_optimizer_state: dict[str, Any] | None = None

    def _pixels(self, observation: DynamicObservation) -> torch.Tensor:
        height, width = observation.image.shape[-2:]
        count = min(self.pixel_samples, height * width)
        linear = torch.randint(height * width, (count,), device=observation.image.device)
        return torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)

    def _components(self, observation: DynamicObservation, stage: str) -> dict[str, torch.Tensor]:
        pixel_uv = self._pixels(observation)
        target = observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]
        render = self.model.predict(observation, pixel_uv, stage)
        image = self.model.canonical.image_regularization(render["intensity_samples"], render["reference_samples_world_mm"], mode=self.model.image_regularization_mode, delta=self.model.image_regularization_delta)
        motion = self.model.motion_regularizers(render["scores"], stage)
        return {"data": (render["predicted_intensity"] - target).square().mean(), "image": image, "mbc_normalization": motion["mbc_normalization"], "smooth_resp": motion["smooth_resp"]}

    def _parameter_norms(self) -> dict[str, float]:
        modules = {"canonical": self.model.canonical, "respiratory_encoder": self.model.film_encoder, "respiratory_mbc": self.model.respiratory_mbc}
        return {f"{name}_parameter_norm": float(torch.sqrt(sum((parameter.detach().square().sum() for parameter in module.parameters()), start=torch.zeros((), device=next(module.parameters()).device)))) for name, module in modules.items()}

    def run_segment(self, segment: ReferenceSegment) -> dict[str, Any]:
        optimizer = configure_reference_segment(self.model, segment, optimizer_config=self.optimizer_config)
        self.model.zero_grad(set_to_none=True)
        before_loss = self.completed[-1].get("loss_last") if self.completed else None
        losses: list[float] = []
        self.model.train()
        for _ in range(segment.steps):
            self.model.zero_grad(set_to_none=True)
            selected = self.sampler.sample_step(observations_per_update=self.observations_per_update)
            total = next(self.model.parameters()).sum() * 0.
            for observation in selected:
                total = total + reference_loss(self._components(observation, segment.stage)) / len(selected)
            total.backward()
            frozen = [name for name, parameter in self.model.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
            if frozen:
                raise RuntimeError("reference frozen-gradient audit failed: " + ", ".join(frozen))
            if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0 for parameter in self.model.parameters() if parameter.requires_grad):
                raise RuntimeError("reference gradient audit found no finite nonzero owner gradient")
            optimizer.step(); self.global_step += 1; losses.append(float(total.detach()))
        diagnostics = boundary_motion_diagnostics(self.model, self.sampler, stage=segment.stage)
        report = {"segment": segment.name, "stage": segment.stage, "segment_step": segment.steps, "global_step": self.global_step, "owners": sorted(segment.owners), "loss_last": losses[-1], "loss_mean": sum(losses) / len(losses), "prior_boundary_loss_last": before_loss, "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}, **diagnostics, **self._parameter_norms()}
        self.completed.append(report); self.last_optimizer_state = optimizer.state_dict()
        return report

    def state_dict(self) -> dict[str, Any]:
        return {"global_step": self.global_step, "completed": list(self.completed), "sampler_state": self.sampler.state_dict(), "schedule": [segment.name for segment in reference_schedule()]}

    def load_boundary_state(self, state: Mapping[str, Any]) -> None:
        self.global_step = int(state["global_step"])
        self.completed = list(state["completed"])
        self.sampler.load_state_dict(state["sampler_state"])


def joint_stage_audits(completed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report warm-to-joint changes without assigning a one-dimensional verdict."""
    by_name = {str(item["segment"]): item for item in completed}
    fields = ("reconstruction_mse_full", "reconstruction_mse_SAX", "reconstruction_mse_2CH", "reconstruction_mse_4CH", "respiratory_score_static_rms", "respiratory_score_dynamic_rms", "respiratory_dvf_static_rms_mm", "respiratory_dvf_dynamic_rms_mm", "respiratory_encoder_parameter_norm", "respiratory_mbc_parameter_norm", "canonical_parameter_norm")
    audits = []
    for level in range(1, 4):
        warm, joint = by_name.get(f"L{level}_warm"), by_name.get(f"L{level}_joint")
        if warm is None or joint is None:
            continue
        audits.append({"level": level, "warm_segment": warm["segment"], "joint_segment": joint["segment"], **{f"delta_{field}": float(joint[field]) - float(warm[field]) for field in fields}})
    return audits
