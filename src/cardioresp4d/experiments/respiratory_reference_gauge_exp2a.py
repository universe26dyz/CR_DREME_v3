"""Isolated Exp2-A hard reference-score gauge protocol contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import (ReferenceSegment, S2VDREMEReferenceRunner, boundary_motion_diagnostics,
                                                                        configure_reference_segment, reference_loss)
from cardioresp4d.visualization.dynamic_audit import fixed_physical_grid, location_local_rms_metrics, weighted_dvf_from_precomputed_basis


EXP2A_PROTOCOL = {
    "experiment": "respiratory_reference_gauge_exp2a",
    "seed": 20260823,
    "observations_per_update": 32,
    "pixel_samples": 256,
    "steps": 200,
    "active_respiratory_levels": 1,
    "canonical_frozen": True,
    "encoder_lr": 5e-4,
    "mbc_lr": 5e-4,
    "canonical_lr": None,
    "normalization": "per_series",
    "score_policy": "raw_minus_same_encoder_reference_mean",
}


class Exp2AReferenceRunner(S2VDREMEReferenceRunner):
    """Exp2-A-only runner with compact gradient and ownership diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_records: list[dict[str, float]] = []
        self.frozen_gradient_violations: list[str] = []

    @staticmethod
    def _gradient_summary(parameters) -> tuple[float, float]:
        values = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
        if not values:
            return 0., 0.
        flat = torch.cat(values)
        return float(torch.linalg.vector_norm(flat)), float(flat.square().mean().sqrt())

    def run_segment(self, segment: ReferenceSegment) -> dict[str, Any]:
        expected = ReferenceSegment("exp2a_step_200", "stage2a", 200, 1, frozenset({"respiratory_encoder", "respiratory_mbc"}))
        if segment != expected:
            raise ValueError("Exp2-A accepts only the fixed exp2_step_200 segment")
        optimizer = configure_reference_segment(self.model, segment, optimizer_config=self.optimizer_config)
        encoder_parameters = list(self.model.film_encoder.parameters())
        l1_parameters = list(self.model.respiratory_mbc.levels[0].parameters())
        canonical_parameters = list(self.model.canonical.parameters())
        inactive_parameters = [parameter for level in self.model.respiratory_mbc.levels[1:] for parameter in level.parameters()]
        losses = []
        self.model.train()
        for _ in range(segment.steps):
            self.model.zero_grad(set_to_none=True)
            selected = self.sampler.sample_step(observations_per_update=self.observations_per_update)
            total = next(self.model.parameters()).sum() * 0.
            for observation in selected:
                total = total + reference_loss(self._components(observation, segment.stage)) / len(selected)
            total.backward()
            for name, parameters in (("canonical", canonical_parameters), ("L2_MBC", inactive_parameters)):
                if any(parameter.grad is not None for parameter in parameters):
                    self.frozen_gradient_violations.append(name)
            encoder_norm, encoder_rms = self._gradient_summary(encoder_parameters)
            mbc_norm, mbc_rms = self._gradient_summary(l1_parameters)
            if not all(torch.isfinite(torch.tensor(value)) for value in (encoder_norm, encoder_rms, mbc_norm, mbc_rms)):
                raise RuntimeError("Exp2-A gradient diagnostics found non-finite owner gradients")
            self.gradient_records.append({"encoder_norm": encoder_norm, "encoder_rms": encoder_rms, "l1_mbc_norm": mbc_norm, "l1_mbc_rms": mbc_rms})
            optimizer.step(); self.global_step += 1; losses.append(float(total.detach()))
        self.model.zero_grad(set_to_none=True)
        diagnostics = boundary_motion_diagnostics(self.model, self.sampler, stage=segment.stage)
        gradient_summary = {key: {"first": self.gradient_records[0][key], "last": self.gradient_records[-1][key], "mean": float(sum(item[key] for item in self.gradient_records) / len(self.gradient_records))} for key in ("encoder_norm", "encoder_rms", "l1_mbc_norm", "l1_mbc_rms")}
        report = {"segment": segment.name, "stage": segment.stage, "segment_step": segment.steps, "global_step": self.global_step, "owners": sorted(segment.owners), "loss_last": losses[-1], "loss_mean": sum(losses) / len(losses), "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}, "gradient_summary": gradient_summary, "canonical_grad_none_all_steps": not any("canonical" == item for item in self.frozen_gradient_violations), "inactive_mbc_grad_none_all_steps": not any("L2_MBC" == item for item in self.frozen_gradient_violations), **diagnostics, **self._parameter_norms()}
        self.completed.append(report); self.last_optimizer_state = optimizer.state_dict()
        return report

    def state_dict(self) -> dict[str, Any]:
        return {"experiment": EXP2A_PROTOCOL["experiment"], "segment": "exp2a_step_200", "steps": 200, "stage": "stage2a", "active_levels": 1, "owners": ["respiratory_encoder", "respiratory_mbc"], "canonical_frozen": True, "global_step": self.global_step, "completed": list(self.completed), "sampler_state": self.sampler.state_dict(), "gradient_records": list(self.gradient_records)}

    def load_boundary_state(self, state: Mapping[str, Any]) -> None:
        expected = {"experiment": EXP2A_PROTOCOL["experiment"], "segment": "exp2a_step_200", "steps": 200, "stage": "stage2a", "active_levels": 1, "canonical_frozen": True}
        if any(state.get(key) != value for key, value in expected.items()) or state.get("owners") != ["respiratory_encoder", "respiratory_mbc"]:
            raise ValueError("invalid Exp2-A runner schedule/state")
        self.global_step = int(state["global_step"]); self.completed = list(state["completed"]); self.gradient_records = list(state.get("gradient_records", [])); self.sampler.load_state_dict(state["sampler_state"])


def exp2a_trainable_parameter_groups(parameters: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
    """Return exactly the Exp2-A owners, excluding canonical and inactive MBC levels."""
    return {"film": list(parameters.get("film", [])), "respiratory_mbc": list(parameters.get("respiratory_mbc", []))[:1]}


def validate_exp2a_protocol(config: Mapping[str, Any]) -> None:
    for key, expected in EXP2A_PROTOCOL.items():
        if key in config and config[key] != expected:
            raise ValueError(f"Exp2-A protocol mismatch for {key}: expected {expected!r}")


def exp2a_provenance(*, manifest: Path, qc_table: Path, canonical_domain: Path, static_manifest: Path, mean_manifest: Path, config: Mapping[str, Any], static_model: Path | None = None) -> dict[str, Any]:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"protocol": dict(EXP2A_PROTOCOL), "manifest_sha256": digest(manifest), "qc_table_sha256": digest(qc_table), "canonical_domain_sha256": digest(canonical_domain), "static_manifest_sha256": digest(static_manifest), "mean_manifest_sha256": digest(mean_manifest), "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    if static_model is not None:
        result["static_model_sha256"] = digest(static_model)
    return result


def exp2a_input_bindings(*, manifest: Path, qc_table: Path, canonical_domain: Path, static_manifest: Path, mean_manifest: Path, static_model: Path, corrected_pose_manifest: Path) -> dict[str, Any]:
    """Create the small immutable input binding consumed by the V9 adapter."""
    def artifact(path: Path) -> dict[str, str]:
        path = Path(path).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    mean_payload = json.loads(Path(mean_manifest).read_text())
    stack_paths = {}
    for record in mean_payload.get("locations", []):
        if record.get("real", True) is False or str(record.get("slice_id", "")).upper().startswith("MISSING"):
            continue
        stack = (Path(mean_manifest).resolve().parent / str(record["stack_file"])).resolve()
        stack_paths[stack.name] = artifact(stack)
    return {"schema_version": 1, "manifest": artifact(manifest), "qc_table": artifact(qc_table), "canonical_domain": artifact(canonical_domain), "static_manifest": artifact(static_manifest), "mean_manifest": artifact(mean_manifest), "static_model": artifact(static_model), "corrected_pose_manifest": artifact(corrected_pose_manifest), "mean_stacks": stack_paths}


def verify_exp2a_input_bindings(bindings: Mapping[str, Any]) -> None:
    """Fail closed if any immutable Exp2-A input changed after training."""
    artifacts = dict(bindings)
    for name, item in artifacts.items():
        if name == "schema_version":
            continue
        if name == "mean_stacks":
            for stack_name, stack_item in dict(item).items():
                path = Path(stack_item["path"])
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != stack_item["sha256"]:
                    raise ValueError(f"Exp2-A input binding changed: mean_stacks/{stack_name}")
            continue
        path = Path(item["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Exp2-A input binding changed: {name}")


def exp2a_parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Freeze canonical and inactive levels; return only Exp2-A optimizer owners."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.respiratory_mbc.set_active_levels(1)
    film = list(model.film_encoder.parameters())
    mbc = list(model.respiratory_mbc.levels[0].parameters())
    for parameter in film + mbc:
        parameter.requires_grad_(True)
    return {"film": film, "respiratory_mbc": mbc}


def exp2a_gauge_diagnostics(model: Any, sampler: Any, *, grid_shape: tuple[int, int, int] = (16, 16, 16)) -> dict[str, Any]:
    """Compute raw/reference/effective score and DVF gauges on one fixed grid."""
    cohort = tuple((view, slice_id, tuple(sorted(frames, key=lambda item: item.timestamp_s))) for view in sorted(sampler._grouped) for slice_id, frames in sorted(sampler._grouped[view].items()) if frames)
    if not cohort:
        raise ValueError("Exp2-A gauge diagnostics require a non-empty cohort")
    grid, _ = fixed_physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, grid_shape)
    raw_basis = model.respiratory_mbc.raw_active(grid, active_levels=1).detach()
    score_values = {name: [] for name in ("raw", "reference", "effective")}; dvf_values = {name: [] for name in score_values}
    with torch.no_grad():
        for _, _, frames in cohort:
            scores_by_kind = {"raw": [], "reference": [], "effective": []}
            for observation in frames:
                raw = model.encode_motion_scores_raw(observation)["resp_scores"][:, :1]
                reference = model.encode_reference_scores(observation)["resp_scores"][:, :1]
                effective = raw - reference
                scores_by_kind["raw"].append(raw[0]); scores_by_kind["reference"].append(reference[0]); scores_by_kind["effective"].append(effective[0])
            for name, values in scores_by_kind.items():
                sequence = torch.stack(values); score_values[name].append(sequence)
                dvf_values[name].append(torch.stack([weighted_dvf_from_precomputed_basis(raw_basis, value[None])[0] for value in values]))
    result: dict[str, Any] = {"location_count": len(cohort), "frame_count": sum(len(frames) for _, _, frames in cohort)}
    by_view: dict[str, dict[str, list[torch.Tensor]]] = {}
    for index, (view, _, _) in enumerate(cohort):
        by_view.setdefault(view, {name: [] for name in score_values})
        for name in score_values:
            by_view[view][name].append(score_values[name][index])
    by_view_metrics: dict[str, dict[str, float]] = {}
    for name in ("raw", "reference", "effective"):
        score_metrics = location_local_rms_metrics(score_values[name]); dvf_metrics = location_local_rms_metrics(dvf_values[name], kind="dvf")
        result.update({f"{name}_score_static_rms": score_metrics["static_rms"], f"{name}_score_dynamic_rms": score_metrics["dynamic_rms"], f"{name}_score_static_dynamic_ratio": score_metrics["static_dynamic_ratio"], f"{name}_score_total_rms": score_metrics["total_rms"], f"{name}_dvf_static_rms_mm": dvf_metrics["static_rms_mm"], f"{name}_dvf_dynamic_rms_mm": dvf_metrics["dynamic_rms_mm"], f"{name}_dvf_static_dynamic_ratio": dvf_metrics["static_dynamic_ratio"], f"{name}_dvf_total_rms_mm": dvf_metrics["total_rms_mm"]})
        for view, values in by_view.items():
            view_scores = values[name]
            view_dvfs = [dvf_values[name][index] for index, item in enumerate(cohort) if item[0] == view]
            score_view = location_local_rms_metrics(view_scores); dvf_view = location_local_rms_metrics(view_dvfs, kind="dvf")
            result.update({f"{name}_score_{view}_static_rms": score_view["static_rms"], f"{name}_score_{view}_dynamic_rms": score_view["dynamic_rms"], f"{name}_dvf_{view}_static_rms_mm": dvf_view["static_rms_mm"], f"{name}_dvf_{view}_dynamic_rms_mm": dvf_view["dynamic_rms_mm"]})
            by_view_metrics.setdefault(view, {}).update({f"{name}_score_static_rms": score_view["static_rms"], f"{name}_score_dynamic_rms": score_view["dynamic_rms"], f"{name}_score_static_dynamic_ratio": score_view["static_dynamic_ratio"], f"{name}_dvf_static_rms_mm": dvf_view["static_rms_mm"], f"{name}_dvf_dynamic_rms_mm": dvf_view["dynamic_rms_mm"], f"{name}_dvf_static_dynamic_ratio": dvf_view["static_dynamic_ratio"]})
    result["by_view"] = by_view_metrics
    return result
