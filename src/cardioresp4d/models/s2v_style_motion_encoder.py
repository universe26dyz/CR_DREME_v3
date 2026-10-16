"""S2V-DREME-style single-frame encoder with view and slice-position FiLM."""
from __future__ import annotations

import math

import torch
from torch import nn


class S2VStyleSingleFrameFiLMEncoder(nn.Module):
    """Infer motion scores from one image conditioned only on view and stack position."""

    def __init__(self, *, position_bands: int = 4, view_embedding_dim: int = 8, condition_dim: int = 64, block_channels: tuple[int, int, int, int] = (32, 64, 64, 64), block_strides: tuple[int, int, int, int] = (1, 2, 1, 2)) -> None:
        super().__init__()
        if position_bands <= 0 or view_embedding_dim <= 0 or condition_dim <= 0:
            raise ValueError("position_bands, view_embedding_dim, and condition_dim must be positive")
        if len(block_channels) != 4 or len(block_strides) != 4 or min(block_channels) <= 0 or min(block_strides) <= 0:
            raise ValueError("S2V encoder requires four positive block channels and strides")
        self.position_bands = int(position_bands)
        self.position_feature_dim = 1 + 2 * self.position_bands
        self.view_embedding = nn.Embedding(3, view_embedding_dim)
        self.condition_mlp = nn.Sequential(nn.Linear(self.position_feature_dim + view_embedding_dim, condition_dim), nn.ReLU(), nn.Linear(condition_dim, condition_dim), nn.ReLU())
        inputs = (1, *block_channels[:-1])
        self.blocks = nn.ModuleList(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1) for in_channels, out_channels, stride in zip(inputs, block_channels, block_strides))
        self.film_heads = nn.ModuleList(nn.Linear(condition_dim, 2 * channels) for channels in block_channels)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(block_channels[-1], block_channels[-1]), nn.ReLU())
        self.resp = nn.Linear(block_channels[-1], 9)
        self.card = nn.Linear(block_channels[-1], 3)

    def encode_position(self, normalized_slice_position: torch.Tensor) -> torch.Tensor:
        if normalized_slice_position.ndim != 2 or normalized_slice_position.shape[1] != 1 or not torch.isfinite(normalized_slice_position).all():
            raise ValueError("normalized_slice_position must be finite [B,1]")
        bands = math.pi * (2.0 ** torch.arange(self.position_bands, dtype=normalized_slice_position.dtype, device=normalized_slice_position.device))
        phase = normalized_slice_position * bands
        return torch.cat((normalized_slice_position, torch.sin(phase), torch.cos(phase)), dim=-1)

    def _condition(self, normalized_slice_position: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
        if view_id.ndim != 1 or view_id.shape[0] != normalized_slice_position.shape[0] or view_id.dtype not in (torch.int32, torch.int64):
            raise ValueError("view_id must be integer [B] matching normalized_slice_position")
        if (view_id < 0).any() or (view_id >= 3).any():
            raise ValueError("view_id must be SAX=0, 2CH=1, or 4CH=2")
        return self.condition_mlp(torch.cat((self.encode_position(normalized_slice_position), self.view_embedding(view_id)), dim=-1))

    def _features(self, image: torch.Tensor, normalized_slice_position: torch.Tensor, view_id: torch.Tensor, *, capture: bool) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if image.ndim == 3:
            image = image[:, None]
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must be [B,1,H,W]")
        if normalized_slice_position.shape[0] != image.shape[0]:
            raise ValueError("image and normalized_slice_position batch sizes must match")
        condition = self._condition(normalized_slice_position.to(image), view_id.to(image.device))
        feature, records = image, []
        for block, film_head in zip(self.blocks, self.film_heads):
            pre_film = block(feature)
            gamma, beta = film_head(condition).chunk(2, dim=-1)
            post_film = pre_film * (1.0 + gamma[..., None, None]) + beta[..., None, None]
            feature = torch.relu(post_film)
            if capture:
                records.append({"pre_film": pre_film, "gamma": gamma, "beta": beta, "post_film": post_film})
        return feature, records

    def forward(self, image: torch.Tensor, *, normalized_slice_position: torch.Tensor, view_id: torch.Tensor) -> dict[str, torch.Tensor]:
        feature, _ = self._features(image, normalized_slice_position, view_id, capture=False)
        feature = self.head(feature)
        return {"resp_scores": self.resp(feature).reshape(image.shape[0], 3, 3), "card_scores": self.card(feature).reshape(image.shape[0], 1, 3)}

    def inspect_film(self, image: torch.Tensor, *, normalized_slice_position: torch.Tensor, view_id: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return per-block pre/post FiLM features for offline diagnostics."""
        _, records = self._features(image, normalized_slice_position, view_id, capture=True)
        return records
