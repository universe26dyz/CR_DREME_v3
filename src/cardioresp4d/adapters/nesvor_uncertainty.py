"""NeSVoR-derived uncertainty with dynamic acquired-frame IDs."""
from __future__ import annotations

import torch
from torch import nn
from ._upstream import add_upstream_to_path

add_upstream_to_path("NeSVoR")
from nesvor.inr.models import build_network  # noqa: E402


class NeSVoRDynamicFrameUncertainty(nn.Module):
    """Use NeSVoR's sigma-network builder; only slice-ID -> frame-ID changes."""
    def __init__(self, *, latent_dim: int, n_dynamic_frames: int, frame_embedding_dim: int = 8, width: int = 64, depth: int = 2) -> None:
        super().__init__()
        if min(latent_dim, n_dynamic_frames, frame_embedding_dim, width) <= 0 or depth < 0: raise ValueError("uncertainty dimensions invalid")
        self.frame_embedding = nn.Embedding(n_dynamic_frames, frame_embedding_dim)
        self.log_var_frame = nn.Parameter(torch.zeros(n_dynamic_frames, dtype=torch.float32))
        self.sigma_net = build_network(n_input_dims=latent_dim + frame_embedding_dim, n_output_dims=1, activation="ReLU", output_activation="None", n_neurons=width, n_hidden_layers=depth, dtype=torch.float32)

    def forward(self, latent_z: torch.Tensor, dynamic_frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if dynamic_frame_ids.dtype != torch.long: raise ValueError("dynamic_frame_ids must be torch.long")
        frame = self.frame_embedding(dynamic_frame_ids)
        while frame.ndim < latent_z.ndim: frame = frame.unsqueeze(-2)
        frame = frame.expand(*latent_z.shape[:-1], frame.shape[-1])
        pixel_scale = self.sigma_net(torch.cat((latent_z, frame), dim=-1)).squeeze(-1).exp()
        frame_variance = self.log_var_frame.exp()[dynamic_frame_ids]
        while frame_variance.ndim < pixel_scale.ndim: frame_variance = frame_variance.unsqueeze(-1)
        variance = pixel_scale.square() + frame_variance
        return {"pixel_scale": pixel_scale, "frame_variance": frame_variance, "variance": variance}

    @staticmethod
    def nll(prediction: torch.Tensor, target: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return torch.nn.GaussianNLLLoss()(prediction, target, variance)
