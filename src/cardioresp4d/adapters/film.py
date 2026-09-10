"""Direct wrapper for the pinned upstream FiLM modulation primitive."""
from __future__ import annotations

from torch import nn
from ._upstream import add_upstream_to_path

add_upstream_to_path("film")
from vr.models.filmed_net import FiLM  # noqa: E402


class FiLMAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.film = FiLM()

    def forward(self, feature, gamma, beta):
        return self.film(feature, gamma, beta)
