"""DREME-MR Eq.6/7 losses; fixed-grid norm is the SINR-hybrid adaptation."""
from __future__ import annotations
import torch


def dreme_mbc_normalization(mbcs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    """Eq.6 over active raw MBCs, retaining level and Cartesian axes first.

    Each tensor is ``[batch, level, spatial..., xyz]``.  The hybrid SINR
    adaptation uses a voxel-volume-normalized discrete mean-square norm, so
    only batch and spatial axes are reduced; level and xyz are then averaged
    exactly once.  Separate respiratory/cardiac grids may be supplied as a
    sequence without resampling either physical domain.
    """
    fields = (mbcs,) if isinstance(mbcs, torch.Tensor) else tuple(mbcs)
    if not fields:
        raise ValueError("Eq.6 requires at least one active MBC field")
    terms: list[torch.Tensor] = []
    for field in fields:
        if field.ndim < 4 or field.shape[-1] != 3:
            raise ValueError("mbcs must be [batch, level, spatial..., xyz]")
        if field.shape[1] == 0:
            continue
        # dim=0 is batch; 2..ndim-2 are all spatial axes.  Keep [level, xyz].
        norm_sq = field.square().mean(dim=(0, *range(2, field.ndim - 1)))
        terms.append((norm_sq - 1.).square().reshape(-1))
    if not terms:
        raise ValueError("Eq.6 requires at least one active MBC level")
    return torch.cat(terms).mean()
def dreme_zero_mean_scores(scores: torch.Tensor) -> torch.Tensor:
    """Eq.7: score-channel temporal mean, never instantaneous score amplitude."""
    if scores.ndim < 2: raise ValueError('scores require time and channel axes')
    return scores.mean(dim=0).square().mean()
