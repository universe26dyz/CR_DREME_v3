"""Source-first dynamic model: NeSVoR canonical INR plus SINR DREME motion."""
from __future__ import annotations

import torch
from torch import nn

from cardioresp4d.adapters.nesvor_inr import NeSVoRCanonicalAdapter
from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.adapters.nesvor_uncertainty import NeSVoRDynamicFrameUncertainty
from cardioresp4d.adapters.sinr_mbc import CardiacSINRMBCAdapter, RespiratorySINRMBCAdapter
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.models.film_motion_encoder import GeometryFiLMMotionEncoder
from .sampler import DynamicObservation


class SourceFirstDynamicModel(nn.Module):
    """No initial volume or mean-slice supervision; all inputs are acquired frames."""
    def __init__(self, lower_world_mm: torch.Tensor, upper_world_mm: torch.Tensor, *, cardiac_lower_world_mm: torch.Tensor, cardiac_upper_world_mm: torch.Tensor, n_dynamic_frames: int, inr_width: int = 64, inr_depth: int = 2, latent_dim: int = 16, motion_hidden_dim: int = 64, respiratory_grid_shapes: tuple[tuple[int, int, int], ...] = ((8, 8, 8), (12, 12, 12), (16, 16, 16)), cardiac_grid_shape: tuple[int, int, int] = (16, 16, 16), psf_samples: int = 8) -> None:
        super().__init__()
        bounds = torch.stack((lower_world_mm, upper_world_mm))
        self.register_buffer("canonical_lower_world_mm", lower_world_mm.to(dtype=torch.float32))
        self.register_buffer("canonical_upper_world_mm", upper_world_mm.to(dtype=torch.float32))
        self.canonical = NeSVoRCanonicalAdapter(bounds, width=inr_width, depth=inr_depth, n_features_z=latent_dim)
        self.film_encoder = GeometryFiLMMotionEncoder(channels=motion_hidden_dim, canonical_lower_world_mm=lower_world_mm, canonical_upper_world_mm=upper_world_mm)
        self.respiratory_mbc = RespiratorySINRMBCAdapter(lower_world_mm, upper_world_mm, grid_shapes=respiratory_grid_shapes, hidden_dim=motion_hidden_dim)
        if torch.any(cardiac_lower_world_mm < lower_world_mm) or torch.any(cardiac_upper_world_mm > upper_world_mm) or torch.any(cardiac_upper_world_mm <= cardiac_lower_world_mm):
            raise ValueError("cardiac MBC box must be a proper local subdomain of canonical full-FOV bounds")
        self.register_buffer("cardiac_lower_world_mm", cardiac_lower_world_mm.to(dtype=torch.float32)); self.register_buffer("cardiac_upper_world_mm", cardiac_upper_world_mm.to(dtype=torch.float32))
        self.cardiac_mbc = CardiacSINRMBCAdapter(cardiac_lower_world_mm, cardiac_upper_world_mm, grid_shape=cardiac_grid_shape, hidden_dim=motion_hidden_dim)
        self.psf = NeSVoRPSFAdapter(psf_samples)
        self.uncertainty = NeSVoRDynamicFrameUncertainty(latent_dim=latent_dim, n_dynamic_frames=n_dynamic_frames, width=inr_width, depth=inr_depth)

    def predict(self, observation: DynamicObservation, pixel_uv: torch.Tensor, stage: str) -> dict[str, torch.Tensor]:
        image = observation.image.unsqueeze(0)
        device, dtype = image.device, image.dtype
        geometry = {"center_mm": observation.center_mm.to(device=device, dtype=dtype)[None], "row_direction": observation.row_direction.to(device=device, dtype=dtype)[None], "column_direction": observation.column_direction.to(device=device, dtype=dtype)[None], "normal": observation.normal.to(device=device, dtype=dtype)[None], "pixel_spacing_mm": observation.pixel_spacing_mm.to(device=device, dtype=dtype)[None], "slice_thickness_mm": torch.tensor([[observation.slice_thickness_mm]], device=device, dtype=dtype)}
        scores = self.film_encoder(image, **geometry)
        respiratory_scores, cardiac_scores = scores["resp_scores"], scores["card_scores"]
        if stage == "stage1": respiratory_scores = respiratory_scores * 0.; cardiac_scores = cardiac_scores * 0.
        elif stage.startswith("stage2"):
            active = {"stage2a": 1, "stage2b": 2, "stage2c": 3}.get(stage, 3)
            respiratory_scores = torch.cat((respiratory_scores[:, :active], respiratory_scores[:, active:] * 0.), dim=1); cardiac_scores = cardiac_scores * 0.
        respiratory = ScoreWeightedMBCField(self.respiratory_mbc, respiratory_scores)
        cardiac = ScoreWeightedMBCField(self.cardiac_mbc, cardiac_scores)
        points = self._pixel_world(observation, pixel_uv.to(device=device, dtype=dtype))
        resolution = torch.tensor([observation.pixel_spacing_mm[1], observation.pixel_spacing_mm[0], observation.slice_thickness_mm], device=device, dtype=dtype).expand(points.shape[0], 3)
        plane_batch = points.shape[0]
        render = self.psf(self.canonical, points, resolution, row_direction=observation.row_direction.to(device=device, dtype=dtype).expand(plane_batch, -1), column_direction=observation.column_direction.to(device=device, dtype=dtype).expand(plane_batch, -1), normal=observation.normal.to(device=device, dtype=dtype).expand(plane_batch, -1), motion=SequentialPullbackMotion(respiratory, cardiac))
        render["scores"] = {"resp_scores": respiratory_scores, "card_scores": cardiac_scores}
        if stage == "stage3": render["uncertainty"] = self.uncertainty(render["latent_samples"], torch.tensor([observation.dynamic_frame_id], device=device, dtype=torch.long))
        return render

    @staticmethod
    def _pixel_world(observation: DynamicObservation, pixel_uv: torch.Tensor) -> torch.Tensor:
        height, width = observation.image.shape[-2:]
        u, v = pixel_uv[:, 0], pixel_uv[:, 1]
        return observation.center_mm.to(pixel_uv) + (u - (width - 1) / 2.)[:, None] * observation.pixel_spacing_mm[1].to(pixel_uv) * observation.row_direction.to(pixel_uv) + (v - (height - 1) / 2.)[:, None] * observation.pixel_spacing_mm[0].to(pixel_uv) * observation.column_direction.to(pixel_uv)

    def motion_regularizers(self, scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Evaluate source-SINR DVF regularizers on a small physical dense grid."""
        axes = [torch.linspace(lower, upper, 4, device=self.canonical_lower_world_mm.device) for lower, upper in zip(self.canonical_lower_world_mm, self.canonical_upper_world_mm)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)
        respiratory = ScoreWeightedMBCField(self.respiratory_mbc, scores["resp_scores"])(grid)
        cardiac = ScoreWeightedMBCField(self.cardiac_mbc, scores["card_scores"])(grid)
        dvf = (respiratory + cardiac).reshape(1, 4, 4, 4, 3)
        spacing = (self.canonical_upper_world_mm - self.canonical_lower_world_mm) / 3.
        smooth = sum((dvf.diff(dim=axis).div(spacing[axis - 1]).square().mean()) for axis in (1, 2, 3))
        return {"mbc": dvf.square().mean(), "smooth": smooth}
