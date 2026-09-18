"""DREME Eq.8/9 on true timestamps with explicit numerical semantics."""
from __future__ import annotations

from typing import Any, Iterable

import torch


def _timestamp_frequency_limits(timestamps_s: torch.Tensor) -> tuple[float, float]:
    """Return span-based resolution and median-spacing Nyquist limit."""
    timestamps = torch.as_tensor(timestamps_s, dtype=torch.float64).reshape(-1)
    if timestamps.numel() < 3 or not torch.isfinite(timestamps).all():
        raise ValueError("need >=3 finite timestamps for frequency evaluation")
    ordered = timestamps.sort().values
    duration = float(ordered[-1] - ordered[0])
    deltas = ordered[1:] - ordered[:-1]
    if not duration > 0 or torch.any(deltas < 0):
        raise ValueError("frequency evaluation requires ordered timestamp support")
    median_delta = float(deltas.median())
    if not median_delta > 0:
        raise ValueError("frequency evaluation requires positive median timestamp spacing")
    nyquist = .5 / median_delta
    if not torch.isfinite(torch.tensor(nyquist)) or not nyquist > 0:
        raise ValueError("frequency evaluation has invalid Nyquist limit")
    return 1. / duration, nyquist


def non_dc_frequency_grid(timestamps_s: torch.Tensor) -> torch.Tensor:
    """Positive span-resolution frequencies through timestamp-derived Nyquist.

    This deliberately differs from the diagnostic's dense scan: it is the
    compact training denominator grid, with no DC coefficient.
    """
    resolution, nyquist = _timestamp_frequency_limits(timestamps_s)
    count = int(torch.floor(torch.tensor(nyquist / resolution)).item())
    if count < 1:
        raise ValueError("timestamp support cannot form a non-DC frequency grid")
    grid = torch.arange(1, count + 1, dtype=torch.float64, device=timestamps_s.device) * resolution
    return grid[grid <= nyquist + torch.finfo(grid.dtype).eps * max(1., nyquist)]


def resolve_target_frequency_mask(grid_hz: torch.Tensor, bands_hz: Iterable[tuple[float, float]] | Iterable[list[float]], *, nyquist_hz: float | None = None) -> torch.Tensor:
    """Resolve local positive target bands against one shared denominator grid.

    A phase-1 band may fall between span-resolution grid centres.  When it is
    otherwise valid and within the timestamp support, the closest centre is
    selected exactly once rather than silently dropping its target power.
    """
    grid = torch.as_tensor(grid_hz, dtype=torch.float64).reshape(-1)
    if not grid.numel() or not torch.isfinite(grid).all() or torch.any(grid <= 0) or torch.any(grid[1:] <= grid[:-1]):
        raise ValueError("target resolution requires a finite strictly increasing non-DC grid")
    inferred_nyquist = float(grid[-1]) + (float((grid[1:] - grid[:-1]).min()) if grid.numel() > 1 else float(grid[-1]))
    upper_limit = float(nyquist_hz) if nyquist_hz is not None else inferred_nyquist
    if not torch.isfinite(torch.tensor(upper_limit)) or upper_limit <= 0:
        raise ValueError("target resolution requires a finite positive Nyquist limit")
    mask = torch.zeros_like(grid, dtype=torch.bool)
    seen = False
    for item in bands_hz:
        try:
            lower, upper = float(item[0]), float(item[1])
        except (TypeError, IndexError) as exc:
            raise ValueError("frequency target band must contain [lower, upper]") from exc
        if not torch.isfinite(torch.tensor((lower, upper))).all() or lower <= 0 or upper < lower or lower > upper_limit:
            raise ValueError("frequency target band is malformed or outside timestamp Nyquist support")
        upper = min(upper, upper_limit)
        inside = (grid >= lower) & (grid <= upper)
        if inside.any():
            mask |= inside
        else:
            # Valid non-DC support but no centre fell in the narrow phase-1
            # band: preserve it by the documented nearest-centre adaptation.
            mask[(grid - ((lower + upper) / 2.)).abs().argmin()] = True
        seen = True
    if not seen:
        raise ValueError("frequency target requires at least one local cardiac band")
    return mask


def cardiac_target_band_concentration(scores: torch.Tensor, timestamps_s: torch.Tensor, cardiac_bands_hz: Iterable[tuple[float, float]] | Iterable[list[float]], *, epsilon: float = 1e-12) -> dict[str, torch.Tensor]:
    """Ratio of local cardiac score power in Phase-1 target bands.

    Power is summed over every cardiac score channel before one target/total
    ratio is formed.  This is a project-specific image-domain adaptation after
    Change4, not a replacement for DREME Eq.8/Eq.9 crossover suppression.
    """
    if not epsilon > 0 or not torch.isfinite(torch.tensor(epsilon)):
        raise ValueError("concentration epsilon must be finite and positive")
    resolution, nyquist = _timestamp_frequency_limits(timestamps_s)
    del resolution  # Documents that grid and Nyquist are derived together.
    grid = non_dc_frequency_grid(timestamps_s).to(device=scores.device, dtype=scores.dtype)
    target_mask = resolve_target_frequency_mask(grid, cardiac_bands_hz, nyquist_hz=nyquist)
    spectrum = nonuniform_dft_at_frequencies(scores, timestamps_s, grid)
    power = spectrum.abs().square().flatten(1).sum(1)
    target_power = power[target_mask].sum()
    total_power = power.sum()
    fraction = target_power / (total_power + scores.new_tensor(epsilon))
    return {
        "loss": 1. - fraction,
        "fraction": fraction,
        "target_power": target_power,
        "total_power": total_power,
        "frequencies_hz": grid,
        "target_mask": target_mask,
        "target_frequencies_hz": grid[target_mask],
    }


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
