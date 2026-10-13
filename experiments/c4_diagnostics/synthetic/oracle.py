"""Encoder-free production-MBC utilities used exclusively by Exp1-B."""
from __future__ import annotations

import math

import torch
from torch import nn

from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.adapters.sinr_mbc import CardiacSINRMBCAdapter, RespiratorySINRMBCAdapter
from cardioresp4d.losses.motion_loss import dreme_mbc_normalization
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.sampler import DynamicObservation

from .motion import GroundTruthMotion
from .render import PhantomCanonicalAdapter


class _ZeroDisplacement(nn.Module):
    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(points_mm)


class OracleScoreGenerator:
    """Fixed timestamp-only score mapping: each Cartesian component shares r/c."""
    def __init__(self, motion: GroundTruthMotion, *, active_respiratory_levels: int) -> None:
        if not 1 <= active_respiratory_levels <= 3:
            raise ValueError("active respiratory levels must be 1..3")
        self.motion = motion
        self.active_respiratory_levels = active_respiratory_levels

    def __call__(self, timestamps_s: torch.Tensor, *, image: torch.Tensor | None = None, geometry: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del image, geometry
        timestamps = torch.as_tensor(timestamps_s).reshape(-1)
        resp = self.motion.respiratory_score(timestamps)[:, None, None].expand(-1, self.active_respiratory_levels, 3)
        resp = resp / math.sqrt(self.active_respiratory_levels)
        card = self.motion.cardiac_score(timestamps)[:, None, None].expand(-1, 1, 3)
        return {"resp_scores": resp, "card_scores": card}


class OracleMBCMotionModel(nn.Module):
    """Production MBC/PSF path with the encoder intentionally absent."""
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, cardiac_lower: torch.Tensor, cardiac_upper: torch.Tensor, canonical: nn.Module, *, active_respiratory_levels: int, hidden_dim: int, respiratory_shapes: tuple[tuple[int, int, int], ...], cardiac_shape: tuple[int, int, int], psf_samples: int) -> None:
        super().__init__()
        self.canonical = canonical
        self.respiratory_mbc = RespiratorySINRMBCAdapter(lower, upper, logical_control_shapes=respiratory_shapes, hidden_dim=hidden_dim)
        self.cardiac_mbc = CardiacSINRMBCAdapter(cardiac_lower, cardiac_upper, logical_control_shape=cardiac_shape, hidden_dim=hidden_dim)
        self.psf = NeSVoRPSFAdapter(psf_samples)
        self.active_respiratory_levels = int(active_respiratory_levels)
        self.respiratory_mbc.set_active_levels(self.active_respiratory_levels)
        self.register_buffer("lower_world_mm", lower)
        self.register_buffer("upper_world_mm", upper)
        self.register_buffer("cardiac_lower_world_mm", cardiac_lower)
        self.register_buffer("cardiac_upper_world_mm", cardiac_upper)
        for parameter in self.canonical.parameters():
            parameter.requires_grad_(False)

    def configure_trainable(self, branch: str) -> None:
        if branch not in {"resp", "card", "combined"}:
            raise ValueError("branch must be resp, card, or combined")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if branch in {"resp", "combined"}:
            for parameter in self.respiratory_mbc.parameters():
                parameter.requires_grad_(True)
        if branch in {"card", "combined"}:
            for parameter in self.cardiac_mbc.parameters():
                parameter.requires_grad_(True)

    def motion(self, scores: dict[str, torch.Tensor], branch: str) -> SequentialPullbackMotion:
        respiratory: nn.Module = _ZeroDisplacement()
        cardiac: nn.Module = _ZeroDisplacement()
        if branch in {"resp", "combined"}:
            respiratory = ScoreWeightedMBCField(self.respiratory_mbc, scores["resp_scores"][:, : self.active_respiratory_levels])
        if branch in {"card", "combined"}:
            cardiac = ScoreWeightedMBCField(self.cardiac_mbc, scores["card_scores"])
        return SequentialPullbackMotion(respiratory, cardiac)

    def render(self, observation: DynamicObservation, scores: dict[str, torch.Tensor], branch: str, pixel_uv: torch.Tensor) -> torch.Tensor:
        points = SourceFirstDynamicModel._pixel_world(observation, pixel_uv.to(observation.image))
        spacing = observation.pixel_spacing_mm.to(points)
        resolution = torch.stack((spacing[1], spacing[0], points.new_tensor(observation.slice_thickness_mm))).expand(points.shape[0], 3)
        axes = {"row_direction": observation.row_direction.to(points).expand(points.shape[0], -1), "column_direction": observation.column_direction.to(points).expand(points.shape[0], -1), "normal": observation.normal.to(points).expand(points.shape[0], -1)}
        return self.psf(self.canonical, points, resolution, **axes, motion=self.motion(scores, branch))["predicted_intensity"]

    def normalization(self, branch: str, *, grid_shape: tuple[int, int, int] = (5, 5, 5)) -> torch.Tensor:
        resp_grid, _ = SourceFirstDynamicModel._physical_grid(self.lower_world_mm, self.upper_world_mm, grid_shape)
        terms: list[torch.Tensor] = []
        if branch in {"resp", "combined"}:
            terms.append(self.respiratory_mbc.raw_active(resp_grid, active_levels=self.active_respiratory_levels))
        if branch in {"card", "combined"}:
            card_grid, _ = SourceFirstDynamicModel._physical_grid(self.cardiac_lower_world_mm, self.cardiac_upper_world_mm, grid_shape)
            terms.append(self.cardiac_mbc(card_grid)[:, None])
        return dreme_mbc_normalization(tuple(terms))


def build_oracle_motion_model(domain: dict[str, object], phantom: nn.Module, *, device: torch.device, active_respiratory_levels: int = 3, hidden_dim: int = 64, logical_shape: tuple[int, int, int] = (8, 8, 8), respiratory_shapes: tuple[tuple[int, int, int], ...] | None = None, cardiac_shape: tuple[int, int, int] | None = None, psf_samples: int = 8) -> OracleMBCMotionModel:
    cardiac = domain["cardiac_box"]
    model = OracleMBCMotionModel(
        torch.as_tensor(domain["world_min_mm"], device=device, dtype=torch.float32),
        torch.as_tensor(domain["world_max_mm"], device=device, dtype=torch.float32),
        torch.as_tensor(cardiac["min_mm"], device=device, dtype=torch.float32),
        torch.as_tensor(cardiac["max_mm"], device=device, dtype=torch.float32),
        PhantomCanonicalAdapter(phantom).to(device),
        active_respiratory_levels=active_respiratory_levels,
        hidden_dim=hidden_dim,
        respiratory_shapes=(logical_shape,) * 3 if respiratory_shapes is None else respiratory_shapes,
        cardiac_shape=logical_shape if cardiac_shape is None else cardiac_shape,
        psf_samples=psf_samples,
    ).to(device)
    return model


def _grid(lower: torch.Tensor, upper: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return SourceFirstDynamicModel._physical_grid(lower, upper, shape)[0]


class DirectMBCFit:
    """Direct mm-DVF fit at a +1 analytic phase; only one MBC branch is mutable."""
    def __init__(self, model: OracleMBCMotionModel, motion: GroundTruthMotion, *, branch: str, grid_shape: tuple[int, int, int], lr: float) -> None:
        if branch not in {"resp", "card"}:
            raise ValueError("direct fit branch must be resp or card")
        self.model, self.motion, self.branch = model, motion, branch
        model.configure_trainable(branch)
        lower, upper = (model.lower_world_mm, model.upper_world_mm) if branch == "resp" else (model.cardiac_lower_world_mm, model.cardiac_upper_world_mm)
        self.points = _grid(lower, upper, grid_shape)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=lr)

    def _target(self) -> torch.Tensor:
        if self.branch == "resp":
            t = self.points.new_tensor([1.0 / (4.0 * self.motion.respiratory_frequency_hz)])
            return self.motion.respiratory_dvf_mm(self.points, t)
        t = self.points.new_tensor([(math.pi / 2.0 - .37) / (2.0 * math.pi * self.motion.cardiac_frequency_hz)])
        return self.motion.cardiac_dvf_mm(self.points, t)

    def predicted(self) -> torch.Tensor:
        if self.branch == "resp":
            scores = self.points.new_ones((1, self.model.active_respiratory_levels, 3)) / math.sqrt(self.model.active_respiratory_levels)
            return ScoreWeightedMBCField(self.model.respiratory_mbc, scores)(self.points)
        return ScoreWeightedMBCField(self.model.cardiac_mbc, self.points.new_ones((1, 1, 3)))(self.points)

    def step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.predicted() - self._target()).square().mean() + 1e-4 * self.model.normalization(self.branch, grid_shape=(3, 3, 3))
        loss.backward(); self.optimizer.step()
        return float(loss.detach())

    def metrics(self) -> dict[str, float]:
        with torch.no_grad():
            prediction, target = self.predicted(), self._target()
            error = (prediction - target).square().sum(-1).sqrt()
            cosine = (prediction * target).sum() / (prediction.norm() * target.norm()).clamp_min(1e-12)
            return {"epe_mm": float(error.mean()), "spatial_cosine": float(cosine), "rms_ratio": float(prediction.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-12)), "max_error_mm": float(error.max()), "mbc_normalization": float(self.model.normalization(self.branch, grid_shape=(3, 3, 3)))}
