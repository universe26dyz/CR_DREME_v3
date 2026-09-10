"""Unified Stage 1 / Stage 2a-b-c / Stage 3 trainer without initial V."""
from __future__ import annotations

import torch

from cardioresp4d.losses.motion_loss import dreme_mbc_normalization, dreme_zero_mean_scores
from cardioresp4d.losses.frequency_loss import dreme_cardiac_leakage_in_resp, dreme_respiratory_leakage_in_card
from .model import SourceFirstDynamicModel
from .sampler import DynamicObservation, ViewLocationBalancedSampler


class UnifiedProgressiveTrainer:
    """Stage-aware optimizer, losses, sampling and true-timestamp score auxiliary."""
    def __init__(self, model: SourceFirstDynamicModel, sampler: ViewLocationBalancedSampler, *, pixel_samples: int = 256, learning_rate: float = 1e-3, loss_weights: dict[str, float] | None = None, frequency_prior=None, temporal_every: int = 1, temporal_batch_size: int = 50, cardiac_sampling_fraction: float = .8) -> None:
        if pixel_samples <= 0 or learning_rate <= 0 or not 0 <= cardiac_sampling_fraction <= 1:
            raise ValueError("pixel_samples, learning_rate, and cardiac_sampling_fraction are invalid")
        self.model, self.sampler, self.pixel_samples, self.learning_rate = model, sampler, pixel_samples, learning_rate
        self.loss_weights = {"image": 1e-4, "mbc_normalization": 1e-5, "smooth_resp": 1e-5, "smooth_card": 1e-5, "zero_mean_score": 1e-5, "cardiac_leakage_in_resp": 1e-4, "respiratory_leakage_in_card": 1e-4, **(loss_weights or {})}
        self.frequency_prior = frequency_prior
        self.temporal_every, self.temporal_batch_size, self.cardiac_sampling_fraction = temporal_every, temporal_batch_size, cardiac_sampling_fraction
        self.optimizer: torch.optim.Optimizer | None = None

    def _configure_stage(self, stage: str) -> None:
        active_resp = {"stage1": 0, "stage2a": 1, "stage2b": 2, "stage2c": 3, "stage3": 3}[stage]
        groups = {"canonical": self.model.canonical, "film": self.model.film_encoder, "resp": self.model.respiratory_mbc, "card": self.model.cardiac_mbc, "uncertainty": self.model.uncertainty}
        enabled = {"stage1": {"canonical"}, "stage2a": {"canonical", "film", "resp"}, "stage2b": {"canonical", "film", "resp"}, "stage2c": {"canonical", "film", "resp"}, "stage3": set(groups)}[stage]
        for name, module in groups.items():
            for parameter in module.parameters():
                parameter.requires_grad_(name in enabled)
        for index, level in enumerate(self.model.respiratory_mbc.levels):
            for parameter in level.parameters():
                parameter.requires_grad_("resp" in enabled and index < active_resp)
        self.model.respiratory_mbc.level_gates.requires_grad_("resp" in enabled)
        self.model.respiratory_mbc.activate_levels(active_resp)
        self.optimizer = torch.optim.Adam([parameter for parameter in self.model.parameters() if parameter.requires_grad], lr=self.learning_rate)

    def run_stage(self, stage: str, *, steps: int) -> dict[str, object]:
        if stage not in {"stage1", "stage2a", "stage2b", "stage2c", "stage3"} or steps <= 0:
            raise ValueError("invalid progressive stage or step count")
        self._configure_stage(stage)
        assert self.optimizer is not None
        losses: list[float] = []; components_last: dict[str, float] = {}; gradients: dict[str, bool] = {}
        self.model.train()
        for step in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            rows = [self._observation_components(observation, stage) for observation in self.sampler.sample_step()]
            components = {key: torch.stack([row[key] for row in rows]).mean() for key in rows[0]}
            if stage != "stage1" and step % self.temporal_every == 0:
                components.update(self._temporal_components(stage))
            total = components["data"] + self.loss_weights["image"] * components["image"]
            if stage != "stage1":
                total = total + self.loss_weights["mbc_normalization"] * components["mbc_normalization"] + self.loss_weights["smooth_resp"] * components["smooth_resp"] + self.loss_weights["zero_mean_score"] * components.get("zero_mean_score", total * 0.) + self.loss_weights["cardiac_leakage_in_resp"] * components.get("cardiac_leakage_in_resp", total * 0.)
            if stage == "stage3":
                total = total + self.loss_weights["smooth_card"] * components["smooth_card"] + self.loss_weights["respiratory_leakage_in_card"] * components.get("respiratory_leakage_in_card", total * 0.)
            total.backward()
            gradients = {"inr": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.canonical.parameters()), "film": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.film_encoder.parameters()), "respiratory_sinr": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.respiratory_mbc.parameters()), "cardiac_sinr": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.cardiac_mbc.parameters()), "uncertainty": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.uncertainty.parameters())}
            self.optimizer.step(); losses.append(float(total.detach())); components_last = {key: float(value.detach()) for key, value in components.items()}
        return {"stage": stage, "steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "loss_components": components_last, "gradient_non_none": gradients}

    def sample_pixels(self, observation: DynamicObservation) -> torch.Tensor:
        """80/20 cardiac-priority/global sampling in patient-world geometry."""
        height, width = observation.image.shape[-2:]; count = min(self.pixel_samples, height * width)
        linear_all = torch.arange(height * width, device=observation.image.device)
        uv_all = torch.stack((linear_all.remainder(width), torch.div(linear_all, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)
        points = self.model._pixel_world(observation, uv_all)
        inside = ((points >= self.model.cardiac_lower_world_mm.to(points)) & (points <= self.model.cardiac_upper_world_mm.to(points))).all(-1)
        cardiac_indices = linear_all[inside]
        cardiac_count = int(round(count * self.cardiac_sampling_fraction)) if cardiac_indices.numel() else 0
        # Pixel sampling is with replacement when a small cardiac intersection
        # contains fewer than its requested quota; otherwise 80/20 would silently
        # collapse on tiny long-axis views.
        chosen = [cardiac_indices[torch.randint(cardiac_indices.numel(), (cardiac_count,), device=linear_all.device)]] if cardiac_count else []
        chosen.append(linear_all[torch.randint(linear_all.numel(), (count - cardiac_count,), device=linear_all.device)])
        linear = torch.cat(chosen)
        return torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)

    def _observation_components(self, observation: DynamicObservation, stage: str) -> dict[str, torch.Tensor]:
        pixel_uv = self.sample_pixels(observation)
        target = observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]
        render = self.model.predict(observation, pixel_uv, stage); prediction = render["predicted_intensity"]
        data = self.model.uncertainty.nll(prediction, target, render["uncertainty"]["variance"]) if stage == "stage3" else (prediction - target).square().mean()
        image = self.model.canonical.image_regularization(render["intensity_samples"], render["reference_samples_world_mm"], mode="edge")
        motion = self.model.motion_regularizers(render["scores"])
        return {"data": data, "image": image, "mbc_normalization": motion["mbc_normalization"], "smooth_resp": motion["smooth_resp"], "smooth_card": motion["smooth_card"]}

    def _temporal_components(self, stage: str) -> dict[str, torch.Tensor]:
        sequence = self.sampler.temporal_batch(max_items=self.temporal_batch_size)
        if len(sequence) < 3 or self.frequency_prior is None:
            zero = next(self.model.parameters()).sum() * 0.
            return {"zero_mean_score":zero,"cardiac_leakage_in_resp":zero,"respiratory_leakage_in_card":zero}
        scores = [self.model.film_encoder(item.image.unsqueeze(0), center_mm=item.center_mm[None], row_direction=item.row_direction[None], column_direction=item.column_direction[None], normal=item.normal[None], pixel_spacing_mm=item.pixel_spacing_mm[None], slice_thickness_mm=torch.tensor([[item.slice_thickness_mm]], device=item.image.device, dtype=item.image.dtype)) for item in sequence]
        timestamps = torch.tensor([item.timestamp_s for item in sequence], device=sequence[0].image.device, dtype=sequence[0].image.dtype)
        resp = torch.cat([score["resp_scores"] for score in scores], 0); card = torch.cat([score["card_scores"] for score in scores], 0)
        result = {"zero_mean_score":dreme_zero_mean_scores(resp if stage!='stage3' else torch.cat((resp,card),1)), "cardiac_leakage_in_resp":dreme_cardiac_leakage_in_resp(resp,timestamps,self.frequency_prior.cardiac_bands_hz,self.frequency_prior.baseline_bands_hz), "respiratory_leakage_in_card":resp.sum()*0.}
        if stage == "stage3": result["respiratory_leakage_in_card"] = dreme_respiratory_leakage_in_card(card,timestamps,self.frequency_prior.respiratory_bands_hz)
        return result
