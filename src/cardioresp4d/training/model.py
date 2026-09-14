"""Source-first dynamic model: NeSVoR canonical INR plus SINR DREME motion."""
from __future__ import annotations

import torch
from torch import nn
from cardioresp4d.losses.motion_loss import dreme_mbc_normalization

from cardioresp4d.adapters.nesvor_inr import NeSVoRCanonicalAdapter
from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.adapters.nesvor_uncertainty import NeSVoRDynamicFrameUncertainty
from cardioresp4d.adapters.sinr_mbc import CardiacSINRMBCAdapter, RespiratorySINRMBCAdapter
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.models.film_motion_encoder import GeometryFiLMMotionEncoder
from .sampler import DynamicObservation
from .stage_contract import stage_contract


class _ZeroDisplacement(nn.Module):
    """Stage-local identity branch that never calls an upstream motion module."""
    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(points_mm)


class SourceFirstDynamicModel(nn.Module):
    """No initial volume or mean-slice supervision; all inputs are acquired frames."""
    def __init__(self, lower_world_mm: torch.Tensor, upper_world_mm: torch.Tensor, *, cardiac_lower_world_mm: torch.Tensor, cardiac_upper_world_mm: torch.Tensor, n_dynamic_frames: int, inr_width: int = 64, inr_depth: int = 2, latent_dim: int = 16, motion_hidden_dim: int = 64, respiratory_grid_shapes: tuple[tuple[int, int, int], ...] = ((8, 8, 8), (12, 12, 12), (16, 16, 16)), cardiac_grid_shape: tuple[int, int, int] = (16, 16, 16), psf_samples: int = 8, canonical_coarsest_resolution: float = 8., canonical_finest_resolution: float = 1., canonical_level_scale: float = 1.5, canonical_features_per_level: int = 2, canonical_log2_hashmap_size: int = 19, canonical_spatial_scaling: float = 1., canonical_encoding_backend: str | None = None, respiratory_cps: int | tuple[int, int, int] = 2, cardiac_cps: int | tuple[int, int, int] = 2, cardiac_taper_mm: float = 4., film_channels: int | None = None, film_position_bands: int = 4, uncertainty_frame_embedding_dim: int = 8, uncertainty_width: int | None = None, uncertainty_depth: int | None = None, respiratory_smoothness_shape: tuple[int, int, int] = (16, 16, 16), cardiac_smoothness_shape: tuple[int, int, int] = (16, 16, 16), image_regularization_mode: str = "edge", image_regularization_delta: float = 1.) -> None:
        super().__init__()
        bounds = torch.stack((lower_world_mm, upper_world_mm))
        self.register_buffer("canonical_lower_world_mm", lower_world_mm.to(dtype=torch.float32))
        self.register_buffer("canonical_upper_world_mm", upper_world_mm.to(dtype=torch.float32))
        self.canonical = NeSVoRCanonicalAdapter(bounds, coarsest_resolution=canonical_coarsest_resolution, finest_resolution=canonical_finest_resolution, level_scale=canonical_level_scale, n_features_per_level=canonical_features_per_level, log2_hashmap_size=canonical_log2_hashmap_size, width=inr_width, depth=inr_depth, n_features_z=latent_dim, spatial_scaling=canonical_spatial_scaling, encoding_backend=canonical_encoding_backend)
        self.film_encoder = GeometryFiLMMotionEncoder(channels=motion_hidden_dim if film_channels is None else film_channels, canonical_lower_world_mm=lower_world_mm, canonical_upper_world_mm=upper_world_mm, position_bands=film_position_bands)
        self.respiratory_mbc = RespiratorySINRMBCAdapter(lower_world_mm, upper_world_mm, grid_shapes=respiratory_grid_shapes, cps=respiratory_cps, hidden_dim=motion_hidden_dim)
        if torch.any(cardiac_lower_world_mm < lower_world_mm) or torch.any(cardiac_upper_world_mm > upper_world_mm) or torch.any(cardiac_upper_world_mm <= cardiac_lower_world_mm):
            raise ValueError("cardiac MBC box must be a proper local subdomain of canonical full-FOV bounds")
        self.register_buffer("cardiac_lower_world_mm", cardiac_lower_world_mm.to(dtype=torch.float32)); self.register_buffer("cardiac_upper_world_mm", cardiac_upper_world_mm.to(dtype=torch.float32))
        self.cardiac_mbc = CardiacSINRMBCAdapter(cardiac_lower_world_mm, cardiac_upper_world_mm, grid_shape=cardiac_grid_shape, cps=cardiac_cps, hidden_dim=motion_hidden_dim, taper_mm=cardiac_taper_mm)
        self.psf = NeSVoRPSFAdapter(psf_samples)
        self.uncertainty = NeSVoRDynamicFrameUncertainty(latent_dim=latent_dim, n_dynamic_frames=n_dynamic_frames, frame_embedding_dim=uncertainty_frame_embedding_dim, width=inr_width if uncertainty_width is None else uncertainty_width, depth=inr_depth if uncertainty_depth is None else uncertainty_depth)
        self.respiratory_smoothness_shape = tuple(int(value) for value in respiratory_smoothness_shape)
        self.cardiac_smoothness_shape = tuple(int(value) for value in cardiac_smoothness_shape)
        self.image_regularization_mode = image_regularization_mode
        self.image_regularization_delta = float(image_regularization_delta)

    def predict(self, observation: DynamicObservation, pixel_uv: torch.Tensor, stage: str) -> dict[str, torch.Tensor]:
        contract = stage_contract(stage)
        image = observation.image.unsqueeze(0)
        device, dtype = image.device, image.dtype
        geometry = {"center_mm": observation.center_mm.to(device=device, dtype=dtype)[None], "row_direction": observation.row_direction.to(device=device, dtype=dtype)[None], "column_direction": observation.column_direction.to(device=device, dtype=dtype)[None], "normal": observation.normal.to(device=device, dtype=dtype)[None], "pixel_spacing_mm": observation.pixel_spacing_mm.to(device=device, dtype=dtype)[None], "slice_thickness_mm": torch.tensor([[observation.slice_thickness_mm]], device=device, dtype=dtype)}
        points = self._pixel_world(observation, pixel_uv.to(device=device, dtype=dtype))
        resolution = torch.tensor([observation.pixel_spacing_mm[1], observation.pixel_spacing_mm[0], observation.slice_thickness_mm], device=device, dtype=dtype).expand(points.shape[0], 3)
        plane_batch = points.shape[0]
        motion = None
        scores: dict[str, torch.Tensor] = {}
        if contract.enable_film:
            encoded = self.film_encoder(image, **geometry)
            self.respiratory_mbc.set_active_levels(contract.active_respiratory_levels)
            respiratory_scores = encoded["resp_scores"][:, :contract.active_respiratory_levels]
            respiratory = ScoreWeightedMBCField(self.respiratory_mbc, respiratory_scores)
            cardiac: nn.Module = _ZeroDisplacement()
            scores["resp_scores"] = respiratory_scores
            if contract.enable_cardiac:
                cardiac_scores = encoded["card_scores"]
                cardiac = ScoreWeightedMBCField(self.cardiac_mbc, cardiac_scores)
                scores["card_scores"] = cardiac_scores
            motion = SequentialPullbackMotion(respiratory, cardiac)
        render = self.psf(self.canonical, points, resolution, row_direction=observation.row_direction.to(device=device, dtype=dtype).expand(plane_batch, -1), column_direction=observation.column_direction.to(device=device, dtype=dtype).expand(plane_batch, -1), normal=observation.normal.to(device=device, dtype=dtype).expand(plane_batch, -1), motion=motion)
        render["scores"] = scores
        if contract.enable_uncertainty:
            render["uncertainty"] = self.uncertainty(render["latent_samples"], torch.tensor([observation.dynamic_frame_id], device=device, dtype=torch.long))
        return render

    @staticmethod
    def _pixel_world(observation: DynamicObservation, pixel_uv: torch.Tensor) -> torch.Tensor:
        height, width = observation.image.shape[-2:]
        u, v = pixel_uv[:, 0], pixel_uv[:, 1]
        return observation.center_mm.to(pixel_uv) + (u - (width - 1) / 2.)[:, None] * observation.pixel_spacing_mm[1].to(pixel_uv) * observation.row_direction.to(pixel_uv) + (v - (height - 1) / 2.)[:, None] * observation.pixel_spacing_mm[0].to(pixel_uv) * observation.column_direction.to(pixel_uv)

    @staticmethod
    def _physical_grid(lower: torch.Tensor, upper: torch.Tensor, shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError("physical smoothness shape must contain three values >=2")
        axes = [torch.linspace(lo, hi, count, device=lower.device, dtype=lower.dtype) for lo, hi, count in zip(lower, upper, shape)]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3), (upper - lower) / torch.tensor([count - 1 for count in shape], device=lower.device, dtype=lower.dtype)

    @staticmethod
    def _smoothness(field: torch.Tensor, shape: tuple[int, int, int], spacing_mm: torch.Tensor) -> torch.Tensor:
        dvf = field.reshape(1, *shape, 3)
        return sum((dvf.diff(dim=axis).div(spacing_mm[axis - 1]).square().mean()) for axis in (1, 2, 3))

    def motion_regularizers(self, scores: dict[str, torch.Tensor], stage: str, *, respiratory_shape: tuple[int, int, int] | None = None, cardiac_shape: tuple[int, int, int] | None = None) -> dict[str, torch.Tensor]:
        """Compute only regularizers owned by ``stage`` on physical domains."""
        contract = stage_contract(stage)
        if not contract.enable_motion:
            raise ValueError("Stage1 owns no motion regularizers")
        respiratory_shape = self.respiratory_smoothness_shape if respiratory_shape is None else respiratory_shape
        cardiac_shape = self.cardiac_smoothness_shape if cardiac_shape is None else cardiac_shape
        resp_grid, resp_spacing = self._physical_grid(self.canonical_lower_world_mm, self.canonical_upper_world_mm, respiratory_shape)
        raw_resp = self.respiratory_mbc.raw_active(resp_grid, active_levels=contract.active_respiratory_levels)
        resp_dvf = ScoreWeightedMBCField(self.respiratory_mbc, scores["resp_scores"])(resp_grid)
        result = {
            "mbc_normalization": dreme_mbc_normalization(raw_resp),
            "smooth_resp": self._smoothness(resp_dvf, respiratory_shape, resp_spacing),
            "resp_dvf_rms_mm": resp_dvf.square().mean().sqrt(),
            "resp_dvf_max_mm": resp_dvf.abs().max(),
        }
        if contract.enable_cardiac:
            card_grid, card_spacing = self._physical_grid(self.cardiac_lower_world_mm, self.cardiac_upper_world_mm, cardiac_shape)
            raw_card = self.cardiac_mbc(card_grid)[:, None]
            card_dvf = ScoreWeightedMBCField(self.cardiac_mbc, scores["card_scores"])(card_grid)
            result["mbc_normalization"] = dreme_mbc_normalization((raw_resp, raw_card))
            result["smooth_card"] = self._smoothness(card_dvf, cardiac_shape, card_spacing)
            result["card_dvf_rms_mm"] = card_dvf.square().mean().sqrt()
            result["card_dvf_max_mm"] = card_dvf.abs().max()
        return result
