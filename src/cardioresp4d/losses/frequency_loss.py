"""DREME Eq.8/9 on true timestamps with explicit numerical semantics."""
from __future__ import annotations

from typing import Any, Iterable

import torch


def nonuniform_dft_at_frequencies(scores: torch.Tensor, timestamps_s: torch.Tensor, frequencies_hz: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Mean-centred, sample-count-normalized equal-weight NUDFT.

    Equal observation weight is the project adaptation for QC-valid acquired
    frames. Time is made relative to its first sample, avoiding loss of phase
    precision from DICOM seconds-since-midnight. The score mean is removed so
    a constant/DC component cannot become a nonzero-frequency leakage term.
    """
    if scores.shape[0] != timestamps_s.numel() or timestamps_s.numel() < 3:
        raise ValueError("need >=3 timestamped scores")
    if not torch.isfinite(timestamps_s).all() or not torch.isfinite(scores).all():
        raise ValueError("timestamps and scores must be finite")
    freq = torch.as_tensor(frequencies_hz, dtype=scores.dtype, device=scores.device).reshape(-1)
    if not freq.numel() or torch.any(freq < 0) or not torch.isfinite(freq).all():
        raise ValueError("frequencies must be finite non-empty nonnegative Hz")
    t64 = timestamps_s.to(device=scores.device, dtype=torch.float64)
    t_rel = t64 - t64[0]
    centred = scores - scores.mean(dim=0, keepdim=True)
    phase = torch.exp(-2j * torch.pi * t_rel[:, None] * freq.to(torch.float64)[None]).to(torch.complex64)
    return torch.einsum("t...,tf->f...", centred.to(torch.complex64), phase) / scores.shape[0]


def resolved_band_frequencies(bands_hz: Iterable[tuple[float, float]] | Iterable[list[float]], timestamps_s: torch.Tensor) -> list[float]:
    """Resolve each interval at temporal resolution; never collapse it to centre."""
    duration = float((timestamps_s.max() - timestamps_s.min()).detach().cpu())
    if not duration > 0:
        raise ValueError("frequency evaluation requires positive timestamp duration")
    resolution = 1. / duration
    result: list[float] = []
    for lower, upper in bands_hz:
        lower, upper = float(lower), float(upper)
        if upper < lower or lower < 0:
            raise ValueError("invalid frequency band")
        if upper == lower:
            result.append(lower)
            continue
        count = max(2, int(round((upper - lower) / resolution)) + 1)
        result.extend(torch.linspace(lower, upper, count).tolist())
    return sorted(set(result))


def _pairs(value: Any, baseline_bands_hz: Any | None) -> list[dict[str, Any]]:
    if baseline_bands_hz is not None:  # compatibility for existing callers/tests
        return [{"cardiac_band_hz": list(c), "baseline_band_hz": list(b)} for c, b in zip(value, baseline_bands_hz)]
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise ValueError("Eq.8 requires explicit cardiac_baseline_pairs")
    return value


def dreme_cardiac_leakage_in_resp(scores: torch.Tensor, timestamps_s: torch.Tensor, cardiac_baseline_pairs: Any, baseline_bands_hz: Any | None = None) -> torch.Tensor:
    """Eq.8: complex coefficient subtraction over explicit paired bands."""
    terms: list[torch.Tensor] = []
    for pair in _pairs(cardiac_baseline_pairs, baseline_bands_hz):
        cardiac = nonuniform_dft_at_frequencies(scores, timestamps_s, resolved_band_frequencies([pair["cardiac_band_hz"]], timestamps_s))
        baseline = nonuniform_dft_at_frequencies(scores, timestamps_s, resolved_band_frequencies([pair["baseline_band_hz"]], timestamps_s))
        terms.append((cardiac[:, None] - baseline[None, :]).abs().square().mean())
    return torch.stack(terms).mean()


def dreme_respiratory_leakage_in_card(scores: torch.Tensor, timestamps_s: torch.Tensor, respiratory_bands_hz: list[tuple[float, float]]) -> torch.Tensor:
    frequencies = resolved_band_frequencies(respiratory_bands_hz, timestamps_s)
    return nonuniform_dft_at_frequencies(scores, timestamps_s, frequencies).abs().square().mean()
