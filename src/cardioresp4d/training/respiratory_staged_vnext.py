"""Fixed, non-joint respiratory-only production-vNext schedule."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from cardioresp4d.losses.motion_loss import respiratory_pca_waveform_subspace_loss
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
    expected = {"resp_a1": {"canonical_lr": 2e-4}, "resp_a2": {"canonical_lr": 5e-5}, "resp_score": {"film_lr": 5e-4}, "resp_mbc": {"respiratory_mbc_lr": 5e-4}, "resp_e": {"canonical_lr": 5e-5}}
    for key, values in expected.items():
        if not isinstance(optimizer.get(key), Mapping) or any(float(optimizer[key].get(name, float("nan"))) != value for name, value in values.items()):
            raise ValueError(f"respiratory vNext requires fixed {key} learning rates")


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

    def run_segment(self, segment: RespiratoryStagedSegment) -> dict[str, object]:
        optimizer = configure_respiratory_staged_segment(self.trainer.model, segment, self.trainer.optimizer_config)
        before = {name: parameter.detach().cpu().clone() for name, parameter in self.trainer.model.named_parameters()}
        stage = segment.stage; contract = stage_contract(stage); losses: list[float] = []
        for step in range(1, segment.steps + 1):
            optimizer.zero_grad(set_to_none=True)
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
                if matched is None:
                    raise ValueError("respiratory vNext PCA segment has no reliable local respiratory selected-PC")
                scores = torch.cat([self.trainer.model.encode_motion_scores(item)["resp_scores"][:, 0, :] for item in sequence], 0)
                waveform = respiratory_pca_waveform_subspace_loss(scores, matched.waveform, ridge=self.trainer.pca_waveform_ridge, min_frames=self.trainer.pca_waveform_min_frames)
                total = total + float(self.loss_weights["respiratory_pca_waveform"]) * waveform["loss"]
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
        report = {"stage": stage, "segment": segment.name, "global_step": self.global_step, "segment_step": segment.steps, "trainable_modules": sorted(segment.owners), "learning_rates": [group["lr"] for group in optimizer.param_groups], "loss_last": losses[-1], "unexpected_parameter_changes": unexpected}
        self.completed.append(report)
        return report

    def state_dict(self) -> dict[str, object]:
        return {"global_step": self.global_step, "completed": list(self.completed), "schedule": [segment.name for segment in respiratory_staged_vnext_segments()]}
