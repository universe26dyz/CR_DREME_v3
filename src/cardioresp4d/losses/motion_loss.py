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


def cardiac_pca_waveform_subspace_loss(card_scores: torch.Tensor, target_waveform: torch.Tensor, *, ridge: float = 1e-4, eps: float = 1e-8, min_frames: int = 8) -> dict[str, torch.Tensor]:
    """Sign/scale/basis-invariant agreement with one local image-derived PC.

    The 3-D cardiac score space is fitted with a per-location differentiable
    ridge projection.  Neither the PCA signal nor the projection introduces a
    trainable parameter, so score-basis permutations and rotations remain free.
    """
    if ridge < 0 or eps <= 0 or min_frames < 3:
        raise ValueError("PCA waveform ridge, eps, or min_frames is invalid")
    scores = card_scores[:, 0, :] if card_scores.ndim == 3 and card_scores.shape[1] == 1 else card_scores
    target = torch.as_tensor(target_waveform, device=scores.device, dtype=scores.dtype).reshape(-1).detach()
    if scores.ndim != 2 or scores.shape[1] != 3 or scores.shape[0] != target.numel():
        raise ValueError("card_scores must be [T,3] or [T,1,3] aligned with target_waveform")
    zero = scores.sum() * 0.
    count = scores.new_tensor(float(scores.shape[0]))
    skipped = scores.new_tensor(1.)
    if scores.shape[0] < min_frames or not torch.isfinite(scores).all() or not torch.isfinite(target).all():
        return {"loss": zero, "r2": zero, "skipped": skipped, "matched_frame_count": count}
    x = scores - scores.mean(dim=0, keepdim=True)
    y = target - target.mean()
    target_energy = y.square().sum()
    if target_energy <= eps:
        return {"loss": zero, "r2": zero, "skipped": skipped, "matched_frame_count": count}
    gram = x.transpose(0, 1) @ x / scores.shape[0]
    cross = x.transpose(0, 1) @ y / scores.shape[0]
    ridge_eff = ridge * gram.diagonal().sum() / 3. + eps
    coefficients = torch.linalg.solve(gram + ridge_eff * torch.eye(3, device=scores.device, dtype=scores.dtype), cross)
    prediction = x @ coefficients
    prediction_energy = prediction.square().sum()
    if prediction_energy <= eps:
        return {"loss": zero, "r2": zero, "skipped": skipped, "matched_frame_count": count}
    correlation = (prediction * y).sum() / torch.sqrt(prediction_energy * target_energy + eps)
    r2 = correlation.square().clamp(0., 1.)
    return {"loss": 1. - r2, "r2": r2, "skipped": zero, "matched_frame_count": count}
