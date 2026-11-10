"""Isolated Exp2-B fixed score-gain ablation protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import (
    REFERENCE_LOSS_WEIGHTS, ReferenceSegment, S2VDREMEReferenceRunner,
    boundary_motion_diagnostics, configure_reference_segment,
    deterministic_diagnostic_cohort,
)
from cardioresp4d.visualization.dynamic_audit import (
    location_local_rms_metrics, mbc_points_from_world_grid,
    weighted_dvf_from_precomputed_basis,
)


EXP2B_ALLOWED_SCORE_GAINS = (10.0, 100.0)
EXP2B_PROTOCOL_BASE = {
    "experiment": "respiratory_reference_gauge_exp2b",
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
    "score_policy": "gain_times_raw_minus_reference",
    "allowed_score_gains": list(EXP2B_ALLOWED_SCORE_GAINS),
}


def validate_exp2b_score_gain(score_gain: float) -> float:
    score_gain = float(score_gain)
    if score_gain not in EXP2B_ALLOWED_SCORE_GAINS:
        raise ValueError(f"Exp2-B score_gain must be one of {EXP2B_ALLOWED_SCORE_GAINS}, got {score_gain!r}")
    return score_gain


def exp2b_protocol(score_gain: float) -> dict[str, Any]:
    score_gain = validate_exp2b_score_gain(score_gain)
    return {**EXP2B_PROTOCOL_BASE, "score_gain": score_gain}


def validate_exp2b_protocol(config: Mapping[str, Any]) -> None:
    exp2a = config.get("exp2a", {})
    expected_exp2a = {"steps": 200, "pixel_samples": 256, "active_respiratory_levels": 1, "score_policy": "raw_minus_same_encoder_reference_mean", "canonical_frozen": True}
    if any(exp2a.get(key) != value for key, value in expected_exp2a.items()):
        raise ValueError("Exp2-B requires the unchanged Exp2-A hard reference protocol")
    exp2b = config.get("exp2b", {})
    expected_exp2b = {"steps": 200, "pixel_samples": 256, "active_respiratory_levels": 1, "score_policy": "gain_times_raw_minus_reference", "canonical_frozen": True}
    if any(exp2b.get(key) != value for key, value in expected_exp2b.items()) or tuple(float(value) for value in exp2b.get("allowed_score_gains", ())) != EXP2B_ALLOWED_SCORE_GAINS:
        raise ValueError("Exp2-B config does not match the fixed score-gain ablation protocol")
    training = config.get("training", {})
    if training.get("observations_per_update") != 32 or training.get("seed") != 20260823 or training.get("normalization", {}).get("mode") != "per_series":
        raise ValueError("Exp2-B requires seed 20260823, 32 observations/update, and per_series normalization")
    if training.get("optimizer") != {"canonical_lr": None, "respiratory_encoder_lr": 5e-4, "respiratory_mbc_lr": 5e-4}:
        raise ValueError("Exp2-B requires the unchanged Exp2-A optimizer learning rates")
    if training.get("schedule") != [{"level": 1, "warm_steps": 200, "joint_steps": 0}]:
        raise ValueError("Exp2-B requires the fixed 200-step L1 schedule")


def reference_loss_components(components: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the unchanged reference objective as auditable weighted terms."""
    result = {
        "data": components["data"],
        "image": REFERENCE_LOSS_WEIGHTS["image"] * components["image"],
        "mbc_normalization": REFERENCE_LOSS_WEIGHTS["mbc_normalization"] * components["mbc_normalization"],
        "smooth_resp": REFERENCE_LOSS_WEIGHTS["smooth_resp"] * components["smooth_resp"],
    }
    result["total"] = result["data"] + result["image"] + result["mbc_normalization"] + result["smooth_resp"]
    return result


def _metric_fields(prefix: str, values: list[torch.Tensor], *, kind: str = "score") -> dict[str, float]:
    metrics = location_local_rms_metrics(values, kind=kind)
    suffix = "_mm" if kind == "dvf" else ""
    return {f"{prefix}_static_rms{suffix}": metrics["static_rms_mm" if kind == "dvf" else "static_rms"], f"{prefix}_dynamic_rms{suffix}": metrics["dynamic_rms_mm" if kind == "dvf" else "dynamic_rms"], f"{prefix}_static_dynamic_ratio": metrics["static_dynamic_ratio"], f"{prefix}_total_rms{suffix}": metrics["total_rms_mm" if kind == "dvf" else "total_rms"]}


def exp2b_gauge_diagnostics(model: Any, sampler: Any, *, score_gain: float | None = None) -> dict[str, Any]:
    """Compute raw/reference/unscaled/used score and DVF diagnostics."""
    gain = validate_exp2b_score_gain(getattr(model, "reference_score_gain", score_gain if score_gain is not None else 1.0))
    if score_gain is not None and gain != validate_exp2b_score_gain(score_gain):
        raise ValueError("model reference score gain differs from requested Exp2-B gain")
    cohort = deterministic_diagnostic_cohort(sampler)
    if not cohort:
        raise ValueError("Exp2-B gauge diagnostics require a non-empty cohort")
    grid, _ = model._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (4, 4, 4))
    raw_basis = model.respiratory_mbc.raw_active(mbc_points_from_world_grid(grid), active_levels=1).detach()
    score_values = {name: [] for name in ("raw", "reference", "unscaled_effective", "used")}
    dvf_values = {name: [] for name in ("unscaled_effective", "used")}
    score_error = 0.0
    dvf_error = 0.0
    with torch.no_grad():
        for _, _, frames in cohort:
            location_scores = {name: [] for name in score_values}
            location_dvfs = {name: [] for name in dvf_values}
            for observation in frames:
                raw = model.encode_motion_scores_raw(observation)["resp_scores"]
                reference = model.encode_reference_scores(observation)["resp_scores"]
                unscaled = raw - reference
                used = gain * unscaled
                score_error = max(score_error, float((used - gain * unscaled).abs().max()))
                unscaled_dvf = weighted_dvf_from_precomputed_basis(raw_basis, unscaled[:, :1])[0]
                used_dvf = weighted_dvf_from_precomputed_basis(raw_basis, used[:, :1])[0]
                dvf_error = max(dvf_error, float((used_dvf - gain * unscaled_dvf).abs().max()))
                location_scores["raw"].append(raw[0]); location_scores["reference"].append(reference[0]); location_scores["unscaled_effective"].append(unscaled[0]); location_scores["used"].append(used[0])
                location_dvfs["unscaled_effective"].append(unscaled_dvf); location_dvfs["used"].append(used_dvf)
            for name in score_values:
                score_values[name].append(torch.stack(location_scores[name]))
            for name in dvf_values:
                dvf_values[name].append(torch.stack(location_dvfs[name]))
    result: dict[str, Any] = {"score_gain": gain, "location_count": len(cohort), "frame_count": sum(len(frames) for _, _, frames in cohort), "score_gain_algebra_max_abs_error": score_error, "dvf_gain_algebra_max_abs_error": dvf_error}
    result.update({**_metric_fields("raw_score", score_values["raw"]), **_metric_fields("reference_score", score_values["reference"]), **_metric_fields("unscaled_effective_score", score_values["unscaled_effective"]), **_metric_fields("used_score", score_values["used"]), **_metric_fields("unscaled_effective_dvf", dvf_values["unscaled_effective"], kind="dvf"), **_metric_fields("used_dvf", dvf_values["used"], kind="dvf")})
    reconstruction = boundary_motion_diagnostics(model, sampler, stage="stage2a")
    result.update({key: value for key, value in reconstruction.items() if key.startswith("reconstruction_")})
    return result


class Exp2BReferenceRunner(S2VDREMEReferenceRunner):
    """Fixed-gain 200-step runner with component and gradient diagnostics."""

    def __init__(self, *args, score_gain: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_gain = validate_exp2b_score_gain(score_gain)
        self.gradient_records: list[dict[str, float]] = []
        self.loss_component_records: list[dict[str, float]] = []
        self.frozen_gradient_violations: list[str] = []

    @staticmethod
    def _gradient_summary(parameters) -> tuple[float, float]:
        values = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
        if not values:
            return 0., 0.
        flat = torch.cat(values)
        return float(torch.linalg.vector_norm(flat)), float(flat.square().mean().sqrt())

    def _summary(self, records: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
        return {key: {"first": records[0][key], "last": records[-1][key], "mean": float(sum(item[key] for item in records) / len(records))} for key in keys}

    def run_segment(self, segment: ReferenceSegment) -> dict[str, Any]:
        expected = ReferenceSegment("exp2b_step_200", "stage2a", 200, 1, frozenset({"respiratory_encoder", "respiratory_mbc"}))
        if segment != expected:
            raise ValueError("Exp2-B accepts only the fixed exp2b_step_200 segment")
        optimizer = configure_reference_segment(self.model, segment, optimizer_config=self.optimizer_config)
        encoder_parameters = list(self.model.film_encoder.parameters())
        l1_parameters = list(self.model.respiratory_mbc.levels[0].parameters())
        canonical_parameters = list(self.model.canonical.parameters())
        inactive_parameters = [parameter for level in self.model.respiratory_mbc.levels[1:] for parameter in level.parameters()]
        self.model.train()
        for _ in range(segment.steps):
            self.model.zero_grad(set_to_none=True)
            selected = self.sampler.sample_step(observations_per_update=self.observations_per_update)
            batch: dict[str, torch.Tensor] | None = None
            for observation in selected:
                terms = reference_loss_components(self._components(observation, segment.stage))
                averaged = {key: value / len(selected) for key, value in terms.items()}
                batch = averaged if batch is None else {key: batch[key] + averaged[key] for key in batch}
            if batch is None:
                raise RuntimeError("Exp2-B sampler returned no observations")
            total = batch["total"]
            total.backward()
            for name, parameters in (("canonical", canonical_parameters), ("L2_MBC", inactive_parameters)):
                if any(parameter.grad is not None for parameter in parameters):
                    self.frozen_gradient_violations.append(name)
            encoder_norm, encoder_rms = self._gradient_summary(encoder_parameters)
            mbc_norm, mbc_rms = self._gradient_summary(l1_parameters)
            if not all(torch.isfinite(torch.tensor(value)) for value in (encoder_norm, encoder_rms, mbc_norm, mbc_rms)):
                raise RuntimeError("Exp2-B gradient diagnostics found non-finite owner gradients")
            self.gradient_records.append({"encoder_norm": encoder_norm, "encoder_rms": encoder_rms, "l1_mbc_norm": mbc_norm, "l1_mbc_rms": mbc_rms})
            self.loss_component_records.append({key: float(value.detach()) for key, value in batch.items()})
            optimizer.step(); self.global_step += 1
        self.model.zero_grad(set_to_none=True)
        diagnostics = exp2b_gauge_diagnostics(self.model, self.sampler, score_gain=self.score_gain)
        loss_summary = self._summary(self.loss_component_records, ("data", "image", "mbc_normalization", "smooth_resp", "total"))
        report = {"segment": segment.name, "stage": segment.stage, "segment_step": segment.steps, "global_step": self.global_step, "score_gain": self.score_gain, "owners": sorted(segment.owners), "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}, "gradient_summary": self._summary(self.gradient_records, ("encoder_norm", "encoder_rms", "l1_mbc_norm", "l1_mbc_rms")), "loss_components": loss_summary, "loss_data": loss_summary["data"], "loss_image": loss_summary["image"], "loss_mbc_normalization": loss_summary["mbc_normalization"], "loss_smooth_resp": loss_summary["smooth_resp"], "loss_total": loss_summary["total"], "canonical_grad_none_all_steps": not any(item == "canonical" for item in self.frozen_gradient_violations), "inactive_mbc_grad_none_all_steps": not any(item == "L2_MBC" for item in self.frozen_gradient_violations), **diagnostics, **self._parameter_norms()}
        self.completed.append(report); self.last_optimizer_state = optimizer.state_dict()
        return report

    def state_dict(self) -> dict[str, Any]:
        return {"experiment": EXP2B_PROTOCOL_BASE["experiment"], "segment": "exp2b_step_200", "stage": "stage2a", "steps": 200, "active_levels": 1, "owners": ["respiratory_encoder", "respiratory_mbc"], "canonical_frozen": True, "score_gain": self.score_gain, "global_step": self.global_step, "completed": list(self.completed), "sampler_state": self.sampler.state_dict(), "gradient_records": list(self.gradient_records), "loss_component_records": list(self.loss_component_records)}

    def load_boundary_state(self, state: Mapping[str, Any]) -> None:
        expected = {"experiment": EXP2B_PROTOCOL_BASE["experiment"], "segment": "exp2b_step_200", "stage": "stage2a", "steps": 200, "active_levels": 1, "owners": ["respiratory_encoder", "respiratory_mbc"], "canonical_frozen": True, "score_gain": self.score_gain}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("invalid Exp2-B runner score-gain provenance")
        self.global_step = int(state["global_step"]); self.completed = list(state["completed"]); self.gradient_records = list(state.get("gradient_records", [])); self.loss_component_records = list(state.get("loss_component_records", [])); self.sampler.load_state_dict(state["sampler_state"])


def exp2b_provenance(*, manifest: Path, qc_table: Path, canonical_domain: Path, static_manifest: Path, mean_manifest: Path, config: Mapping[str, Any], score_gain: float, static_model: Path | None = None) -> dict[str, Any]:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    gain = validate_exp2b_score_gain(score_gain)
    result = {"protocol": exp2b_protocol(gain), "score_gain": gain, "manifest_sha256": digest(manifest), "qc_table_sha256": digest(qc_table), "canonical_domain_sha256": digest(canonical_domain), "static_manifest_sha256": digest(static_manifest), "mean_manifest_sha256": digest(mean_manifest), "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    if static_model is not None:
        result["static_model_sha256"] = digest(static_model)
    return result
