"""DREME-MR Eq.6/7 losses; fixed-grid norm is the SINR-hybrid adaptation."""
from __future__ import annotations
import torch
def dreme_mbc_normalization(mbcs: torch.Tensor) -> torch.Tensor:
    """Eq.6: mean over level/component of (physical discrete L2 norm² - 1)²."""
    if mbcs.ndim < 3 or mbcs.shape[-1] != 3: raise ValueError('mbcs must end in Cartesian xyz')
    # Spatial mean is volume-normalized discrete L2², stable across fixed grids.
    norm_sq=mbcs.square().mean(dim=tuple(range(mbcs.ndim-2)))
    return (norm_sq-1.).square().mean()
def dreme_zero_mean_scores(scores: torch.Tensor) -> torch.Tensor:
    """Eq.7: score-channel temporal mean, never instantaneous score amplitude."""
    if scores.ndim < 2: raise ValueError('scores require time and channel axes')
    return scores.mean(dim=0).square().mean()
