"""Extract fixed-slice temporal PCA modes and auditable frequency candidates.

This necessary local adaptation uses image-domain, mean-centred NumPy SVD and a
one-sided SciPy periodogram.  It only accepts ordinary FFT sampling when
AcquisitionTime intervals are uniform to the stated relative tolerance; it
deliberately provides no irregular-sampling fallback.

Example:
    python -m cardioresp4d.frequency.pca_motion --manifest results/dicom_manifest.csv \\
        --output-dir results/frequency
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.frequency.frequency_bands import aggregate_frequency_bands


UNIFORM_RTOL = 1e-3
# DICOM AcquisitionTime in this dataset is serialised to milliseconds, so a
# single quantised interval may differ by 1 ms even when sampling is regular.
UNIFORM_ATOL_S = 1.1e-3
RESPIRATORY_BAND_HZ = (0.10, 0.60)
CARDIAC_BAND_HZ = (0.80, 2.00)
DOMINANCE_THRESHOLD = 2.2
MAXIMUM_DF_HZ = 0.20


class SamplingIrregularError(ValueError):
    """Raised when ordinary FFT sampling is unsafe for the supplied timestamps."""


def validate_uniform_sampling(
    timestamps_s: np.ndarray,
    relative_tolerance: float = UNIFORM_RTOL,
    absolute_tolerance_s: float = UNIFORM_ATOL_S,
) -> tuple[float, float]:
    """Return median sampling interval and largest deviation after a uniformity check.

    Intervals must be strictly positive and their absolute deviations from the
    median must not exceed ``max(absolute_tolerance_s,
    relative_tolerance * median_dt)``.  The explicit tolerance is 0.1% or
    1.1 ms (to accommodate millisecond DICOM time quantisation), whichever is
    larger, because no Lomb--Scargle or resampling fallback is implemented.
    """
    times = np.asarray(timestamps_s, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("timestamps_s must be a one-dimensional array with at least two samples")
    if not np.isfinite(times).all():
        raise ValueError("timestamps_s must be finite")
    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise SamplingIrregularError("AcquisitionTime timestamps must be strictly increasing and uniform")
    median_dt = float(np.median(intervals))
    max_deviation = float(np.max(np.abs(intervals - median_dt)))
    tolerance = max(float(absolute_tolerance_s), float(relative_tolerance) * median_dt)
    if max_deviation > tolerance:
        raise SamplingIrregularError(
            "AcquisitionTime sampling is not uniform for ordinary FFT/periodogram "
            f"(max deviation {max_deviation:.9g} s exceeds tolerance {tolerance:.9g} s)"
        )
    return median_dt, max_deviation


def analyze_image_series(
    images: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    relative_tolerance: float = UNIFORM_RTOL,
) -> dict[str, Any]:
    """Perform time-by-pixels PCA and a one-sided PSD for one fixed image slice.

    The returned dictionary contains machine-readable arrays plus scalar QC and
    candidate fields.  Temporal PCA scores are exactly ``U * S`` from the thin
    SVD of the per-pixel mean-centred time-by-pixels matrix.
    """
    image_array = np.asarray(images, dtype=np.float32)
    times = np.asarray(timestamps_s, dtype=float)
    if image_array.ndim != 3:
        raise ValueError("images must have shape (time, rows, columns)")
    if image_array.shape[0] != times.size:
        raise ValueError("images and timestamps_s must have the same number of frames")
    if image_array.shape[0] < 2:
        raise ValueError("at least two image frames are required")
    if not np.isfinite(image_array).all():
        raise ValueError("images must be finite")

    median_dt, max_deviation = validate_uniform_sampling(times, relative_tolerance)
    n_frames = image_array.shape[0]
    data_matrix = image_array.reshape(n_frames, -1).astype(np.float64, copy=False)
    data_matrix = data_matrix - data_matrix.mean(axis=0, keepdims=True)
    temporal_left, singular_values, _ = np.linalg.svd(data_matrix, full_matrices=False)
    temporal_pcs = temporal_left * singular_values[None, :]
    variance = singular_values ** 2
    explained_variance = variance / variance.sum() if variance.sum() > 0.0 else np.zeros_like(variance)

    sampling_frequency_hz = 1.0 / median_dt
    frequencies_hz, pc_psd = periodogram(
        temporal_pcs,
        fs=sampling_frequency_hz,
        axis=0,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    duration_span_s = float(times[-1] - times[0])
    duration_n_dt_s = float(n_frames * median_dt)
    df_hz = sampling_frequency_hz / n_frames
    respiratory_candidate = _select_band_candidate(
        frequencies_hz, pc_psd, RESPIRATORY_BAND_HZ, df_hz
    )
    cardiac_candidate = _select_band_candidate(
        frequencies_hz, pc_psd, CARDIAC_BAND_HZ, df_hz
    )
    return {
        "timestamps_s": times,
        "temporal_pcs": temporal_pcs,
        "explained_variance": explained_variance,
        "frequencies_hz": frequencies_hz,
        "pc_psd": pc_psd,
        "n_frames": int(n_frames),
        "median_dt_s": median_dt,
        "max_dt_deviation_s": max_deviation,
        "duration_span_s": duration_span_s,
        "duration_n_dt_s": duration_n_dt_s,
        "df_hz": float(df_hz),
        "nyquist_hz": float(frequencies_hz[-1]),
        "uniform_relative_tolerance": float(relative_tolerance),
        "respiratory_candidate": respiratory_candidate,
        "cardiac_candidate": cardiac_candidate,
    }


def _select_band_candidate(
    frequencies_hz: np.ndarray,
    pc_psd: np.ndarray,
    band_hz: tuple[float, float],
    df_hz: float,
) -> dict[str, Any]:
    """Choose the strongest PC peak in one physiological search band with QC.

    Reliability is determined by positive peak power and the explicit dominance
    threshold.  Observation duration is reported separately as a frequency
    resolution limitation, not as a hard candidate-rejection threshold.
    """
    empty = {
        "frequency_hz": None,
        "selected_pc": None,
        "peak_power": None,
        "dominance": None,
        "reliable": False,
        "reason": "no_frequency_bin_in_search_band",
        "search_band_hz": list(band_hz),
    }
    if df_hz > MAXIMUM_DF_HZ:
        empty["reason"] = "insufficient_frequency_resolution"
        return empty
    mask = (frequencies_hz >= band_hz[0]) & (frequencies_hz <= band_hz[1])
    if not np.any(mask):
        return empty
    band_powers = pc_psd[mask, :]
    best_flat_index = int(np.argmax(band_powers))
    frequency_index, pc_index = np.unravel_index(best_flat_index, band_powers.shape)
    selected_frequency = float(frequencies_hz[mask][frequency_index])
    peak_power = float(band_powers[frequency_index, pc_index])
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        empty["reason"] = "no_positive_peak_power"
        return empty
    comparison = pc_psd[frequencies_hz > 0.0, pc_index]
    noise_floor = float(np.median(comparison))
    dominance = peak_power / max(noise_floor, np.finfo(float).tiny)
    candidate = {
        "frequency_hz": selected_frequency,
        "selected_pc": int(pc_index + 1),
        "peak_power": peak_power,
        "dominance": float(dominance),
        "reliable": False,
        "reason": "insufficient_peak_dominance",
        "search_band_hz": list(band_hz),
    }
    if dominance < DOMINANCE_THRESHOLD:
        candidate["reason"] = "insufficient_peak_dominance"
    else:
        candidate["reliable"] = True
        candidate["reason"] = "peak_dominance_at_least_2.2"
    return candidate


def write_slice_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    """Write NPZ, temporal-PC CSV, PSD CSV, and a compact PCA/PSD PNG for one slice."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "pca_psd.npz",
        timestamps_s=result["timestamps_s"],
        temporal_pcs=result["temporal_pcs"],
        explained_variance=result["explained_variance"],
        frequencies_hz=result["frequencies_hz"],
        pc_psd=result["pc_psd"],
    )
    _write_matrix_csv(destination / "temporal_pc.csv", "timestamp_s", result["timestamps_s"], result["temporal_pcs"])
    _write_matrix_csv(destination / "spectrum.csv", "frequency_hz", result["frequencies_hz"], result["pc_psd"])
    _write_qc_png(result, destination / "pca_qc.png")


def _write_matrix_csv(path: Path, first_column: str, axis: np.ndarray, matrix: np.ndarray) -> None:
    """Write a numeric coordinate plus all PCA columns without lossy rounding."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([first_column, *[f"pc_{index + 1}" for index in range(matrix.shape[1])]])
        for coordinate, values in zip(axis, matrix):
            writer.writerow([f"{float(coordinate):.12g}", *[f"{float(value):.12g}" for value in values]])


def _write_qc_png(result: dict[str, Any], path: Path) -> None:
    """Plot leading temporal PCs and their PSDs for visual output QC."""
    count = min(3, result["temporal_pcs"].shape[1])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for index in range(count):
        axes[0].plot(result["timestamps_s"], result["temporal_pcs"][:, index], label=f"PC {index + 1}")
        axes[1].plot(result["frequencies_hz"], result["pc_psd"][:, index], label=f"PC {index + 1}")
    axes[0].set(xlabel="AcquisitionTime (s)", ylabel="temporal score", title="Temporal PCA scores")
    axes[1].set(xlabel="frequency (Hz)", ylabel="PSD", title="One-sided periodogram", xlim=(0.0, result["nyquist_hz"]))
    axes[0].legend()
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    relative_tolerance: float = UNIFORM_RTOL,
) -> dict[str, Any]:
    """Analyze all complete 50-frame manifest slices and write aggregate frequency JSON."""
    dataset = CardioRespDataset(manifest_path)
    grouped_indices: dict[tuple[str, str], list[int]] = {}
    for index in range(len(dataset)):
        row = dataset._rows[index]
        grouped_indices.setdefault((row["view"], row["slice_id"]), []).append(index)
    results: list[dict[str, Any]] = []
    root = Path(output_dir)
    for (view, slice_id), indices in sorted(grouped_indices.items()):
        if len(indices) != 50:
            raise ValueError(f"{view}/{slice_id} has {len(indices)} frames; fixed-slice analysis requires exactly 50")
        indices.sort(key=lambda index: float(dataset._rows[index]["timestamp_s"]))
        samples = [dataset[index] for index in indices]
        result = analyze_image_series(
            np.stack([sample["image"] for sample in samples]),
            np.asarray([sample["timestamp_s"] for sample in samples]),
            relative_tolerance=relative_tolerance,
        )
        result["view"] = view
        result["slice_id"] = slice_id
        result["slice_key"] = f"{view}/{slice_id}"
        write_slice_outputs(result, root / _safe_path_component(view) / _safe_path_component(slice_id))
        results.append(result)
    aggregate = aggregate_frequency_bands(results)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "frequency_bands.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
        handle.write("\n")
    return aggregate


def _safe_path_component(value: str) -> str:
    """Create a conservative output-directory component from manifest metadata."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unnamed"


def main() -> None:
    """Run fixed-slice PCA/PSD analysis over the authoritative DICOM manifest."""
    parser = argparse.ArgumentParser(description="Run fixed-slice image PCA and frequency analysis.")
    parser.add_argument("--manifest", required=True, type=Path, help="Authoritative DICOM CSV manifest")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for per-slice outputs and aggregate JSON")
    parser.add_argument("--uniform-rtol", type=float, default=UNIFORM_RTOL, help="Maximum relative interval deviation (default: 0.001)")
    args = parser.parse_args()
    aggregate = analyze_manifest(args.manifest, args.output_dir, relative_tolerance=args.uniform_rtol)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
