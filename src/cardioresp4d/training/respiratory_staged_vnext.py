"""Fixed, non-joint respiratory-only production-vNext schedule."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from cardioresp4d.diagnostics.stage_visual import dvf_static_dynamic_metrics
from cardioresp4d.losses.motion_loss import respiratory_pca_waveform_subspace_loss
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.stage_contract import stage_contract


@dataclass(frozen=True)
class RespiratoryStagedSegment:
    name: str
    stage: str
    steps: int
    optimizer_key: str
    owners: frozenset[str]
    active_respiratory_levels: int
    respiratory_pca: bool = False
    data_loss: bool = True
    motion_regularization: bool = False


def respiratory_staged_vnext_segments() -> tuple[RespiratoryStagedSegment, ...]:
    return (
        RespiratoryStagedSegment("resp_static_bootstrap_a1", "stage1", 500, "resp_a1", frozenset({"canonical"}), 0),
        RespiratoryStagedSegment("resp_static_bootstrap_a2", "stage1", 1300, "resp_a2", frozenset({"canonical"}), 0),
        RespiratoryStagedSegment("resp_pca_warmup", "stage2a", 1000, "resp_score", frozenset({"respiratory_encoder"}), 1, respiratory_pca=True, data_loss=False),
        RespiratoryStagedSegment("resp_mbc_bootstrap_c1", "stage2a", 250, "resp_mbc", frozenset({"respiratory_mbc"}), 1, motion_regularization=True),
        RespiratoryStagedSegment("resp_mbc_bootstrap_c2", "stage2b", 250, "resp_mbc", frozenset({"respiratory_mbc"}), 2, motion_regularization=True),
        RespiratoryStagedSegment("resp_mbc_bootstrap_c3", "stage2c", 250, "resp_mbc", frozenset({"respiratory_mbc"}), 3, motion_regularization=True),
        RespiratoryStagedSegment("resp_score_refine", "stage2c", 1500, "resp_score", frozenset({"respiratory_encoder"}), 3, respiratory_pca=True),
        RespiratoryStagedSegment("resp_canonical_refine", "stage2c", 2000, "resp_e", frozenset({"canonical"}), 3),
    )


def validate_respiratory_staged_vnext_config(config: Mapping[str, Any]) -> None:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("respiratory vNext requires training configuration")
    weights, optimizer = training.get("loss_weights"), training.get("optimizer")
    if not isinstance(weights, Mapping) or not isinstance(optimizer, Mapping):
        raise ValueError("respiratory vNext requires loss_weights and optimizer mappings")
    pca_weight = weights.get("respiratory_pca_waveform")
    if isinstance(pca_weight, bool) or not isinstance(pca_weight, (int, float)) or not math.isfinite(float(pca_weight)) or float(pca_weight) != 1e-3:
        raise ValueError("respiratory_pca_waveform must be the fixed formal value 1e-3")
    zero_mean_weight = weights.get("zero_mean_score")
    if isinstance(zero_mean_weight, bool) or not isinstance(zero_mean_weight, (int, float)) or not math.isfinite(float(zero_mean_weight)) or float(zero_mean_weight) != 1e-2:
        raise ValueError("zero_mean_score must be the fixed formal value 1e-2")
    expected = {"resp_a1": {"canonical_lr": 2e-4}, "resp_a2": {"canonical_lr": 5e-5}, "resp_score": {"film_lr": 5e-4}, "resp_mbc": {"respiratory_mbc_lr": 5e-4}, "resp_e": {"canonical_lr": 5e-5}}
    for key, values in expected.items():
        if not isinstance(optimizer.get(key), Mapping) or any(float(optimizer[key].get(name, float("nan"))) != value for name, value in values.items()):
            raise ValueError(f"respiratory vNext requires fixed {key} learning rates")


def multilevel_respiratory_pca_waveform_loss(respiratory_scores: torch.Tensor, waveform: torch.Tensor | None, *, ridge: float = 1e-4, min_frames: int = 8) -> dict[str, object]:
    """Anchor each of the three MBC score levels to the same respiratory PC."""
    if respiratory_scores.ndim != 3 or respiratory_scores.shape[1:] != (3, 3):
        raise ValueError("respiratory PCA anchoring requires respiratory_scores [T,3,3]")
    zero = respiratory_scores.sum() * 0.
    if waveform is None:
        return {"loss": zero, "r2": zero, "loss_by_level": (zero, zero, zero), "r2_by_level": (zero, zero, zero), "matched": False}
    losses = tuple(respiratory_pca_waveform_subspace_loss(respiratory_scores[:, level, :], waveform, ridge=ridge, min_frames=min_frames) for level in range(3))
    loss_by_level = tuple(item["loss"] for item in losses); r2_by_level = tuple(item["r2"] for item in losses)
    return {"loss": torch.stack(loss_by_level).mean(), "r2": torch.stack(r2_by_level).mean(), "loss_by_level": loss_by_level, "r2_by_level": r2_by_level, "matched": True}


def respiratory_score_rms_metrics(scores: torch.Tensor) -> dict[str, float]:
    """Separate respiratory score DC gauge from its temporal variation."""
    if scores.ndim < 2 or not scores.numel() or not torch.isfinite(scores).all():
        raise ValueError("score diagnostics require a non-empty finite [T,...] tensor")
    mean = scores.mean(dim=0, keepdim=True)
    total_rms = float(scores.square().mean().sqrt())
    mean_rms = float(mean.square().mean().sqrt())
    dynamic_rms = float((scores - mean).square().mean().sqrt())
    return {"respiratory_score_temporal_rms": total_rms, "respiratory_score_static_rms": mean_rms, "respiratory_score_total_rms": total_rms, "respiratory_score_mean_rms": mean_rms, "respiratory_score_dynamic_rms": dynamic_rms}


def configure_respiratory_staged_segment(model, segment: RespiratoryStagedSegment, optimizer_config: Mapping[str, Mapping[str, float]]) -> torch.optim.Adam:
    """Apply one vNext ownership boundary and construct a fresh optimizer."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.respiratory_mbc.set_active_levels(segment.active_respiratory_levels)
    groups: list[dict[str, object]] = []
    settings = optimizer_config[segment.optimizer_key]
    if "canonical" in segment.owners:
        parameters = list(model.canonical.parameters())
        for parameter in parameters: parameter.requires_grad_(True)
        groups.append({"params": parameters, "lr": float(settings["canonical_lr"]), "name": "canonical"})
    if "respiratory_encoder" in segment.owners:
        parameters = []
        for module in (model.film_encoder.image, model.film_encoder.geometry_mlp, model.film_encoder.head, model.film_encoder.resp):
            parameters.extend(module.parameters())
        for parameter in parameters: parameter.requires_grad_(True)
        groups.append({"params": parameters, "lr": float(settings["film_lr"]), "name": "respiratory_encoder"})
    if "respiratory_mbc" in segment.owners:
        parameters = []
        for level in model.respiratory_mbc.levels[:segment.active_respiratory_levels]:
            parameters.extend(level.parameters())
        for parameter in parameters: parameter.requires_grad_(True)
        groups.append({"params": parameters, "lr": float(settings["respiratory_mbc_lr"]), "name": "respiratory_mbc"})
    if len(groups) != 1:
        raise RuntimeError("respiratory vNext segment must own exactly one parameter factor")
    return torch.optim.Adam(groups)


class RespiratoryStagedVNextRunner:
    """Small production runner that enforces the five non-joint vNext updates."""
    def __init__(self, trainer, *, respiratory_pca_waveform_prior, loss_weights: Mapping[str, float]) -> None:
        self.trainer = trainer
        self.prior = respiratory_pca_waveform_prior
        self.loss_weights = loss_weights
        self.global_step = 0
        self.completed: list[dict[str, object]] = []
        self.last_optimizer_state: dict[str, object] | None = None

    def run_segment(self, segment: RespiratoryStagedSegment) -> dict[str, object]:
        optimizer = configure_respiratory_staged_segment(self.trainer.model, segment, self.trainer.optimizer_config)
        self.trainer.model.zero_grad(set_to_none=True)
        before = {name: parameter.detach().cpu().clone() for name, parameter in self.trainer.model.named_parameters()}
        stage = segment.stage; contract = stage_contract(stage); losses: list[float] = []
        waveform_losses: list[list[float]] = [[], [], []]; waveform_r2: list[list[float]] = [[], [], []]
        pca_matched_steps = 0; pca_skipped_steps = 0
        for step in range(1, segment.steps + 1):
            self.trainer.model.zero_grad(set_to_none=True)
            total = next(self.trainer.model.parameters()).sum() * 0.
            if segment.data_loss:
                rows = [self.trainer._observation_components(observation, stage) for observation in self.trainer.sampler.sample_step(observations_per_update=self.trainer.observations_per_update)]
                for components in rows:
                    value = components["data"] + self.trainer.loss_weights["image"] * components["image"]
                    if segment.motion_regularization:
                        value = value + self.trainer.loss_weights["mbc_normalization"] * components["mbc_normalization"] + self.trainer.loss_weights["smooth_resp"] * components["smooth_resp"]
                    total = total + value / len(rows)
            if segment.respiratory_pca:
                temporal = self.trainer._temporal_components(stage)
                total = total + self.trainer.loss_weights["zero_mean_score"] * temporal["zero_mean_score"] + self.trainer.loss_weights["cardiac_leakage_in_resp"] * temporal["cardiac_leakage_in_resp"]
                sequence = self.trainer.sampler.temporal_batch(max_items=self.trainer.temporal_batch_size)
                if self.prior is None:
                    raise ValueError("respiratory vNext PCA segments require reliable respiratory PCA provenance")
                timestamps = torch.tensor([item.timestamp_s for item in sequence], device=sequence[0].image.device, dtype=torch.float64)
                matched = self.prior.match(sequence[0].view, sequence[0].slice_id, timestamps, device=sequence[0].image.device, dtype=sequence[0].image.dtype)
                scores = torch.cat([self.trainer.model.encode_motion_scores(item)["resp_scores"] for item in sequence], 0)
                waveform = multilevel_respiratory_pca_waveform_loss(scores, None if matched is None else matched.waveform, ridge=self.trainer.pca_waveform_ridge, min_frames=self.trainer.pca_waveform_min_frames)
                total = total + float(self.loss_weights["respiratory_pca_waveform"]) * waveform["loss"]
                if waveform["matched"]:
                    pca_matched_steps += 1
                    for level, value in enumerate(waveform["loss_by_level"]): waveform_losses[level].append(float(value.detach()))
                    for level, value in enumerate(waveform["r2_by_level"]): waveform_r2[level].append(float(value.detach()))
                else:
                    pca_skipped_steps += 1
            total.backward()
            frozen = [name for name, parameter in self.trainer.model.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
            if frozen:
                raise RuntimeError("respiratory vNext frozen-gradient audit failed: " + ", ".join(frozen))
            optimizer.step(); self.global_step += 1; losses.append(float(total.detach()))
        changed = [name for name, parameter in self.trainer.model.named_parameters() if not torch.equal(before[name], parameter.detach().cpu())]
        owner_prefixes = {"canonical": ("canonical.",), "respiratory_encoder": ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."), "respiratory_mbc": ("respiratory_mbc.levels.",)}
        allowed = owner_prefixes[next(iter(segment.owners))]
        unexpected = [name for name in changed if not name.startswith(allowed)]
        if unexpected:
            raise RuntimeError("respiratory vNext ownership audit failed: " + ", ".join(unexpected))
        report = {"stage": stage, "segment": segment.name, "global_step": self.global_step, "segment_step": segment.steps, "trainable_modules": sorted(segment.owners), "learning_rates": [group["lr"] for group in optimizer.param_groups], "loss_last": losses[-1], "loss_mean": sum(losses) / len(losses), "unexpected_parameter_changes": unexpected}
        if segment.respiratory_pca:
            level_losses = [None if not values else sum(values) / len(values) for values in waveform_losses]
            level_r2 = [None if not values else sum(values) / len(values) for values in waveform_r2]
            report.update({"respiratory_pca_matched_steps": pca_matched_steps, "respiratory_pca_skipped_steps": pca_skipped_steps, **{f"respiratory_pca_loss_level{level}": level_losses[level] for level in range(3)}, **{f"respiratory_pca_r2_level{level}": level_r2[level] for level in range(3)}, "respiratory_pca_loss_mean": None if not pca_matched_steps else sum(value for value in level_losses if value is not None) / 3, "respiratory_pca_r2_mean": None if not pca_matched_steps else sum(value for value in level_r2 if value is not None) / 3, "respiratory_pca_loss_last": None if not pca_matched_steps else sum(values[-1] for values in waveform_losses) / 3, "respiratory_pca_r2_last": None if not pca_matched_steps else sum(values[-1] for values in waveform_r2) / 3, **self._score_diagnostics(3)})
        if segment.name.startswith(("resp_mbc_bootstrap", "resp_score_refine", "resp_canonical_refine")):
            report.update(self._dvf_diagnostics(stage))
        self.last_optimizer_state = optimizer.state_dict()
        self.completed.append(report)
        return report

    def _diagnostic_cohort(self) -> tuple[tuple[str, str, tuple[object, ...]], ...]:
        """Read a sorted full-span observation cohort without sampling RNG."""
        grouped = self.trainer.sampler._grouped
        return tuple((view, slice_id, tuple(sorted(frames, key=lambda item: item.timestamp_s))) for view in sorted(grouped) for slice_id, frames in sorted(grouped[view].items()) if frames)

    def _score_diagnostics(self, active_levels: int) -> dict[str, float]:
        cohort = self._diagnostic_cohort()
        with torch.no_grad():
            scores = torch.cat([self.trainer.model.encode_motion_scores(item)["resp_scores"][:, :active_levels] for _, _, frames in cohort for item in frames], 0)
        return respiratory_score_rms_metrics(scores)

    def _dvf_diagnostics(self, stage: str) -> dict[str, float]:
        cohort = self._diagnostic_cohort()
        if not cohort:
            return {}
        model = self.trainer.model
        with torch.no_grad():
            grid, _ = model._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (4, 4, 4))
            values = []
            for _, _, frames in cohort:
              for observation in frames:
                score = model.encode_motion_scores(observation)["resp_scores"][:, :stage_contract(stage).active_respiratory_levels]
                values.append(ScoreWeightedMBCField(model.respiratory_mbc, score)(grid)[0])
            _, _, metrics = dvf_static_dynamic_metrics(torch.stack(values))
        total_rms = float(torch.stack(values).square().mean().sqrt())
        return {"respiratory_dvf_rms_mm": total_rms, "respiratory_dvf_total_rms_mm": total_rms, "respiratory_dvf_static_rms_mm": metrics["mean_DVF_RMS_mm"], "respiratory_dvf_dynamic_rms_mm": metrics["dynamic_DVF_temporal_RMS_mm"], "diagnostic_location_count": len(cohort), "diagnostic_frame_count": sum(len(frames) for _, _, frames in cohort)}

    def state_dict(self) -> dict[str, object]:
        return {"global_step": self.global_step, "completed": list(self.completed), "current_segment": None if not self.completed else self.completed[-1]["segment"], "segment_step": 0 if not self.completed else self.completed[-1]["segment_step"], "sampler_state": self.trainer.sampler.state_dict(), "optimizer": self.last_optimizer_state, "schedule": [segment.name for segment in respiratory_staged_vnext_segments()]}

    def load_boundary_state(self, state: Mapping[str, object]) -> None:
        self.global_step = int(state["global_step"])
        self.completed = list(state["completed"])
        self.last_optimizer_state = state.get("optimizer")  # type: ignore[assignment]
        self.trainer.sampler.load_state_dict(state["sampler_state"])  # type: ignore[arg-type]
