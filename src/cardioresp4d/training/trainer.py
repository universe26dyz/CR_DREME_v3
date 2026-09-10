"""Unified Stage 1 / Stage 2a-b-c / Stage 3 trainer without initial V."""
from __future__ import annotations

import torch

from .model import SourceFirstDynamicModel
from .sampler import DynamicObservation, ViewLocationBalancedSampler


class UnifiedProgressiveTrainer:
    def __init__(self, model: SourceFirstDynamicModel, sampler: ViewLocationBalancedSampler, *, pixel_samples: int = 256, learning_rate: float = 1e-3, image_regularization_weight: float = 1e-4) -> None:
        if pixel_samples <= 0 or learning_rate <= 0: raise ValueError("pixel_samples and learning_rate must be positive")
        self.model, self.sampler, self.pixel_samples, self.image_regularization_weight = model, sampler, pixel_samples, image_regularization_weight
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def run_stage(self, stage: str, *, steps: int) -> dict[str, object]:
        if stage not in {"stage1", "stage2a", "stage2b", "stage2c", "stage3"} or steps <= 0: raise ValueError("invalid progressive stage or step count")
        losses: list[float] = []; gradients: dict[str, bool] = {}
        self.model.train()
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            view_losses = [self._observation_loss(observation, stage) for observation in self.sampler.sample_step()]
            loss = torch.stack(view_losses).mean(); loss.backward()
            gradients = {"inr": any(parameter.grad is not None for parameter in self.model.canonical.parameters()), "film": any(parameter.grad is not None for parameter in self.model.film_encoder.parameters()), "respiratory_sinr": any(parameter.grad is not None for parameter in self.model.respiratory_mbc.parameters()), "cardiac_sinr": any(parameter.grad is not None for parameter in self.model.cardiac_mbc.parameters()), "uncertainty": any(parameter.grad is not None for parameter in self.model.uncertainty.parameters())}
            self.optimizer.step(); losses.append(float(loss.detach()))
        return {"stage": stage, "steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "gradient_non_none": gradients}

    def _observation_loss(self, observation: DynamicObservation, stage: str) -> torch.Tensor:
        height, width = observation.image.shape[-2:]
        count = min(self.pixel_samples, height * width)
        linear = torch.randperm(height * width, device=observation.image.device)[:count]
        pixel_uv = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(dtype=observation.image.dtype)
        target = observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]
        render = self.model.predict(observation, pixel_uv, stage)
        prediction = render["predicted_intensity"]
        data_loss = self.model.uncertainty.nll(prediction, target, render["uncertainty"]["variance"].mean(-1)) if stage == "stage3" else (prediction - target).square().mean()
        image_reg = self.model.canonical.image_regularization(render["intensity_samples"], render["reference_samples_world_mm"], mode="edge")
        return data_loss + self.image_regularization_weight * image_reg
