"""Unified Stage 1 / Stage 2a-b-c / Stage 3 trainer without initial V."""
from __future__ import annotations

import torch

from cardioresp4d.losses.motion_loss import dreme_mbc_normalization, dreme_zero_mean_scores
from cardioresp4d.losses.frequency_loss import cardiac_target_band_concentration, dreme_cardiac_leakage_in_resp, dreme_respiratory_leakage_in_card
from .model import SourceFirstDynamicModel
from .sampler import DynamicObservation, ViewLocationBalancedSampler
from .stage_contract import progressive_stage_order, stage_contract


class UnifiedProgressiveTrainer:
    """Stage-aware optimizer, losses, sampling and true-timestamp score auxiliary."""
    def __init__(self, model: SourceFirstDynamicModel, sampler: ViewLocationBalancedSampler, *, pixel_samples: int = 256, learning_rate: float = 1e-3, optimizer_config: dict[str, dict[str, float]] | None = None, loss_weights: dict[str, float] | None = None, frequency_prior=None, temporal_every: int = 1, temporal_batch_size: int = 50, cardiac_sampling_fraction: float = .8) -> None:
        if pixel_samples <= 0 or learning_rate <= 0 or not 0 <= cardiac_sampling_fraction <= 1:
            raise ValueError("pixel_samples, learning_rate, and cardiac_sampling_fraction are invalid")
        self.model, self.sampler, self.pixel_samples, self.learning_rate = model, sampler, pixel_samples, learning_rate
        self.optimizer_config = optimizer_config or {}
        self.loss_weights = {"image": 1e-4, "mbc_normalization": 1e-5, "smooth_resp": 1e-5, "smooth_card": 1e-5, "zero_mean_score": 1e-5, "cardiac_leakage_in_resp": 1e-4, "respiratory_leakage_in_card": 1e-4, "cardiac_target_concentration": 0., **(loss_weights or {})}
        self.frequency_prior = frequency_prior
        self.temporal_every, self.temporal_batch_size, self.cardiac_sampling_fraction = temporal_every, temporal_batch_size, cardiac_sampling_fraction
        self.optimizer: torch.optim.Optimizer | None = None
        self._optimizer_parameter_ids: set[int] = set()
        self.current_stage: str | None = None
        self.global_step = 0
        self.stage_step = 0
        self.metrics: list[dict[str, object]] = []

    def _configure_stage(self, stage: str) -> None:
        contract = stage_contract(stage)
        groups = {"canonical": self.model.canonical, "film": self.model.film_encoder, "respiratory_mbc": self.model.respiratory_mbc, "cardiac_mbc": self.model.cardiac_mbc, "uncertainty": self.model.uncertainty}
        enabled = contract.trainable_modules
        for name, module in groups.items():
            for parameter in module.parameters():
                parameter.requires_grad_(name in enabled)
        if contract.film_train_mode == "card_head_only":
            # Stage3a intentionally leaves the Stage2c shared FiLM/respiratory
            # representation immutable while its cardiac output head warms up.
            for parameter in self.model.film_encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.model.film_encoder.card.parameters():
                parameter.requires_grad_(True)
        for index, level in enumerate(self.model.respiratory_mbc.levels):
            for parameter in level.parameters():
                parameter.requires_grad_(contract.train_respiratory_mbc and index < contract.active_respiratory_levels)
        self.model.respiratory_mbc.set_active_levels(contract.active_respiratory_levels)
        stage_key = "stage2" if stage.startswith("stage2") else stage
        aliases = {"canonical": "canonical_lr", "film": "film_lr", "respiratory_mbc": "respiratory_mbc_lr", "cardiac_mbc": "cardiac_mbc_lr", "uncertainty": "uncertainty_lr"}
        configured = self.optimizer_config.get(stage_key, {})
        active_by_module: dict[str, list[torch.nn.Parameter]] = {}
        for name, module in groups.items():
            parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
            if parameters:
                active_by_module[name] = parameters
        if self.optimizer is None:
            first_name, first_parameters = next(iter(active_by_module.items()))
            self.optimizer = torch.optim.Adam([{"params": first_parameters, "lr": float(configured.get(aliases[first_name], self.learning_rate)), "name": first_name}])
            self._optimizer_parameter_ids.update(id(parameter) for parameter in first_parameters)
            active_by_module.pop(first_name)
        assert self.optimizer is not None
        for name, parameters in active_by_module.items():
            fresh = [parameter for parameter in parameters if id(parameter) not in self._optimizer_parameter_ids]
            if fresh:
                self.optimizer.add_param_group({"params": fresh, "lr": float(configured.get(aliases[name], self.learning_rate)), "name": name})
                self._optimizer_parameter_ids.update(id(parameter) for parameter in fresh)
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name in aliases:
                group["lr"] = float(configured.get(aliases[name], self.learning_rate))
        self.current_stage = stage
        self.stage_step = 0

    def run_stage(self, stage: str, *, steps: int) -> dict[str, object]:
        try:
            contract = stage_contract(stage)
        except ValueError as exc:
            raise ValueError("invalid progressive stage or step count") from exc
        if steps <= 0:
            raise ValueError("invalid progressive stage or step count")
        self._configure_stage(stage)
        assert self.optimizer is not None
        losses: list[float] = []; components_last: dict[str, float] = {}; gradients: dict[str, bool] = {}
        self.model.train()
        for step in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            rows = [self._observation_components(observation, stage) for observation in self.sampler.sample_step()]
            components = {key: torch.stack([row[key] for row in rows]).mean() for key in rows[0]}
            if contract.enable_motion and step % self.temporal_every == 0:
                components.update(self._temporal_components(stage))
            total = components["data"] + self.loss_weights["image"] * components["image"]
            if contract.enable_motion:
                total = total + self.loss_weights["mbc_normalization"] * components["mbc_normalization"] + self.loss_weights["smooth_resp"] * components["smooth_resp"] + self.loss_weights["zero_mean_score"] * components.get("zero_mean_score", total * 0.) + self.loss_weights["cardiac_leakage_in_resp"] * components.get("cardiac_leakage_in_resp", total * 0.)
            if contract.enable_cardiac:
                total = total + self.loss_weights["smooth_card"] * components["smooth_card"] + self.loss_weights["respiratory_leakage_in_card"] * components.get("respiratory_leakage_in_card", total * 0.) + self.loss_weights["cardiac_target_concentration"] * components.get("cardiac_target_concentration", total * 0.)
            total.backward()
            gradients = {"inr": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.canonical.parameters()), "film": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.film_encoder.parameters()), "respiratory_sinr": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.respiratory_mbc.parameters()), "cardiac_sinr": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.cardiac_mbc.parameters()), "uncertainty": any(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.uncertainty.parameters())}
            self.optimizer.step(); losses.append(float(total.detach())); components_last = {key: float(value.detach()) for key, value in components.items()}
            self.global_step += 1; self.stage_step += 1
            self.metrics.append({"global_step": self.global_step, "stage_step": self.stage_step, "stage": stage, "total": float(total.detach()), **components_last, "learning_rates": [group["lr"] for group in self.optimizer.param_groups], "resp_active_levels": contract.active_respiratory_levels})
        return {"stage": stage, "steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "loss_mean": sum(losses) / len(losses), "loss_min": min(losses), "loss_max": max(losses), "loss_components": components_last, "gradient_non_none": gradients, "global_step": self.global_step}

    def training_state_dict(self) -> dict[str, object]:
        if self.optimizer is None or self.current_stage is None:
            raise RuntimeError("cannot checkpoint before a configured training stage")
        return {"current_stage": self.current_stage, "global_step": self.global_step, "stage_step": self.stage_step, "optimizer": self.optimizer.state_dict(), "sampler": self.sampler.state_dict(), "metrics": list(self.metrics)}

    def load_training_state_dict(self, state: dict[str, object]) -> None:
        stage = str(state["current_stage"])
        if stage == "stage3":
            raise ValueError("legacy v3 Stage3 checkpoint is not compatible with v3_change4 schedule; resume from the Stage2c checkpoint")
        order = progressive_stage_order()
        if stage not in order:
            raise ValueError("checkpoint has unsupported current stage")
        for name in order[:order.index(stage) + 1]:
            self._configure_stage(name)
        assert self.optimizer is not None
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        self.sampler.load_state_dict(state["sampler"])  # type: ignore[arg-type]
        self.global_step = int(state["global_step"])
        self.stage_step = int(state["stage_step"])
        self.metrics = list(state.get("metrics", []))

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
        contract = stage_contract(stage)
        data = self.model.uncertainty.nll(prediction, target, render["uncertainty"]["variance"]) if contract.data_term == "gaussian_nll" else (prediction - target).square().mean()
        image = self.model.canonical.image_regularization(render["intensity_samples"], render["reference_samples_world_mm"], mode=self.model.image_regularization_mode, delta=self.model.image_regularization_delta)
        if not contract.enable_motion:
            return {"data": data, "image": image}
        motion = self.model.motion_regularizers(render["scores"], stage)
        result = {"data": data, "image": image, "mbc_normalization": motion["mbc_normalization"], "smooth_resp": motion["smooth_resp"]}
        result.update({key: value for key, value in motion.items() if key not in result})
        result["resp_score_mean"] = render["scores"]["resp_scores"].mean()
        result["resp_score_std"] = render["scores"]["resp_scores"].std(unbiased=False)
        if contract.enable_cardiac:
            result["smooth_card"] = motion["smooth_card"]
            result["card_score_mean"] = render["scores"]["card_scores"].mean()
            result["card_score_std"] = render["scores"]["card_scores"].std(unbiased=False)
        return result

    def _temporal_components(self, stage: str) -> dict[str, torch.Tensor]:
        sequence = self.sampler.temporal_batch(max_items=self.temporal_batch_size)
        contract = stage_contract(stage)
        if len(sequence) < 3 or self.frequency_prior is None:
            zero = next(self.model.parameters()).sum() * 0.
            result = {"zero_mean_score":zero,"cardiac_leakage_in_resp":zero,"respiratory_leakage_in_card":zero}
            if contract.enable_cardiac:
                result.update({"cardiac_target_concentration": zero, "cardiac_target_fraction": zero, "cardiac_target_power": zero, "cardiac_total_power": zero})
            return result
        scores = [self.model.film_encoder(item.image.unsqueeze(0), center_mm=item.center_mm[None], row_direction=item.row_direction[None], column_direction=item.column_direction[None], normal=item.normal[None], pixel_spacing_mm=item.pixel_spacing_mm[None], slice_thickness_mm=torch.tensor([[item.slice_thickness_mm]], device=item.image.device, dtype=item.image.dtype)) for item in sequence]
        timestamps = torch.tensor([item.timestamp_s for item in sequence], device=sequence[0].image.device, dtype=torch.float64)
        resp = torch.cat([score["resp_scores"][:, :contract.active_respiratory_levels] for score in scores], 0); card = torch.cat([score["card_scores"] for score in scores], 0)
        prior = self.frequency_prior.for_location(sequence[0].view, sequence[0].slice_id) if hasattr(self.frequency_prior, "for_location") else self.frequency_prior
        result = {"zero_mean_score":dreme_zero_mean_scores(resp if not contract.enable_cardiac else torch.cat((resp,card),1)), "cardiac_leakage_in_resp":dreme_cardiac_leakage_in_resp(resp,timestamps,prior.cardiac_baseline_pairs), "respiratory_leakage_in_card":resp.sum()*0.}
        if contract.enable_cardiac:
            # Phase-1 respiratory occupancy is location-specific when reliable;
            # using the aggregate here would make Eq.9 suppress the wrong band.
            result["respiratory_leakage_in_card"] = dreme_respiratory_leakage_in_card(card, timestamps, prior.respiratory_bands_hz)
            concentration = cardiac_target_band_concentration(card, timestamps, prior.cardiac_bands_hz)
            result.update({"cardiac_target_concentration": concentration["loss"], "cardiac_target_fraction": concentration["fraction"], "cardiac_target_power": concentration["target_power"], "cardiac_total_power": concentration["total_power"]})
        return result
