"""功能：在每个固定 slice 的 50 帧内标记明显 acquisition corruption。
论文来源：DSVR-style NCC/local rejection 与 NeSVoR robust weighting 思路；阈值为本项目必要适配。
输入：完成 modality scaling 后的 50 帧图像及 reader rescale status。
输出：独立 PHI-free QC table，含 valid/reason/NCC/global scale/scale-corrected residual。
主要步骤：temporal median reference、robust scale、NCC/residual、block 内 median+MAD 判定。
是否属于原论文直接实现 / 必要适配 / 可选实验：necessary adaptation；不删除 DICOM。
CLI：内部 library，无独立 CLI，由 Phase-1 runner 调用。
"""
from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from cardioresp4d.data.dataset import CardioRespDataset

QC_COLUMNS = ("source_file_token", "view", "slice_id", "frame_index", "qc_valid",
              "qc_reason", "qc_ncc", "qc_intensity_scale", "qc_residual", "rescale_status")


def analyze_frame_block(images: np.ndarray, rescale_statuses: Iterable[str], *,
                        ncc_mad_threshold: float = 6.0, scale_mad_threshold: float = 6.0,
                        residual_mad_threshold: float = 6.0, mad_floor: float = 1e-6) -> list[dict[str, Any]]:
    """Return transparent per-frame metrics and robust corruption flags."""
    values = np.asarray(images, dtype=np.float64)
    statuses = list(rescale_statuses)
    if values.ndim != 3 or values.shape[0] != 50 or not np.isfinite(values).all():
        raise ValueError("images must be a finite 50-frame (time,row,column) block")
    if len(statuses) != 50:
        raise ValueError("rescale_statuses must contain exactly 50 entries")
    thresholds = (ncc_mad_threshold, scale_mad_threshold, residual_mad_threshold, mad_floor)
    if any(not np.isfinite(x) or x <= 0 for x in thresholds):
        raise ValueError("QC thresholds and mad_floor must be finite and positive")
    reference = np.median(values, axis=0)
    mask = np.abs(reference) > max(float(np.percentile(np.abs(reference), 25)), np.finfo(float).eps)
    if not np.any(mask):
        mask = np.ones(reference.shape, dtype=bool)
    ref_vector = reference[mask]
    ref_centered = ref_vector - np.mean(ref_vector)
    ref_norm = max(float(np.linalg.norm(ref_centered)), np.finfo(float).eps)
    scales, nccs, residuals = [], [], []
    for frame in values:
        frame_vector = frame[mask]
        ratios = frame_vector / np.where(np.abs(ref_vector) > np.finfo(float).eps, ref_vector, np.nan)
        finite_ratios = ratios[np.isfinite(ratios)]
        scale = float(np.median(finite_ratios)) if finite_ratios.size else 1.0
        scale = max(scale, np.finfo(float).eps)
        centered = frame_vector - np.mean(frame_vector)
        denom = max(float(np.linalg.norm(centered)) * ref_norm, np.finfo(float).eps)
        ncc = float(np.dot(centered, ref_centered) / denom)
        corrected = frame / scale
        residual = float(np.mean(np.abs(corrected - reference)) /
                         max(float(np.mean(np.abs(reference))), np.finfo(float).eps))
        scales.append(scale); nccs.append(ncc); residuals.append(residual)
    log_scales = np.log(np.asarray(scales))
    ncc_array, residual_array = np.asarray(nccs), np.asarray(residuals)
    scale_center, scale_spread = _median_mad(log_scales, max(mad_floor, 0.01))
    ncc_center, ncc_spread = _median_mad(ncc_array, max(mad_floor, 0.005))
    residual_center, residual_spread = _median_mad(residual_array, max(mad_floor, 0.005))
    output = []
    for index, (scale, ncc, residual) in enumerate(zip(scales, nccs, residuals)):
        reasons = []
        # Require a large absolute scale excursion as well as MAD evidence; small
        # first-frame/transient physiology is not acquisition corruption.
        if abs(np.log(scale) - scale_center) > max(scale_mad_threshold * scale_spread, 0.25):
            reasons.append("global_intensity_scale")
        if ncc < ncc_center - ncc_mad_threshold * ncc_spread:
            reasons.append("low_ncc")
        if residual > residual_center + residual_mad_threshold * residual_spread:
            reasons.append("scale_corrected_residual")
        output.append({"qc_valid": not reasons, "qc_reason": "valid" if not reasons else ";".join(reasons),
                       "qc_ncc": ncc, "qc_intensity_scale": scale, "qc_residual": residual,
                       "rescale_status": str(statuses[index])})
    return output


def write_qc_table(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Write the independent path-free acquisition QC table."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row); payload["qc_valid"] = int(bool(payload["qc_valid"]))
            writer.writerow(payload)
    return path


def run_acquisition_qc(manifest_path: str | Path, output_dir: str | Path, *,
                       slice_keys: set[str] | None = None, ncc_mad_threshold: float = 6.0,
                       scale_mad_threshold: float = 6.0, residual_mad_threshold: float = 6.0,
                       mad_floor: float = 1e-6) -> tuple[Path, Path, Path]:
    """Evaluate all or selected complete blocks and write table/summary/compact plot."""
    dataset = CardioRespDataset(manifest_path, valid_only=False)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset._rows):
        key = (row["view"], row["slice_id"])
        if slice_keys is None or f"{key[0]}/{key[1]}" in slice_keys:
            groups[key].append(index)
    if not groups:
        raise ValueError("No manifest slice blocks selected for acquisition QC")
    all_rows, slice_means = [], {}
    for (view, slice_id), indices in sorted(groups.items()):
        indices.sort(key=lambda i: int(dataset._rows[i]["frame_index"]))
        if len(indices) != 50:
            raise ValueError(f"{view}/{slice_id} must contain exactly 50 frames before QC")
        samples = [dataset[i] for i in indices]
        metrics = analyze_frame_block(np.stack([s["rescaled_image"] for s in samples]),
                                      [s["rescale_status"] for s in samples],
                                      ncc_mad_threshold=ncc_mad_threshold,
                                      scale_mad_threshold=scale_mad_threshold,
                                      residual_mad_threshold=residual_mad_threshold, mad_floor=mad_floor)
        valid_images = [sample["image"] for sample, metric in zip(samples, metrics) if metric["qc_valid"]]
        if not valid_images:
            raise ValueError(f"Acquisition QC rejected every frame in {view}/{slice_id}; no fabricated replacement is allowed")
        slice_means[(view, slice_id)] = np.mean(np.stack(valid_images), axis=0)
        for sample, metric, index in zip(samples, metrics, indices):
            row = dataset._rows[index]
            metric.update(source_file_token=row.get("source_file_token", ""), view=view, slice_id=slice_id,
                          frame_index=int(row["frame_index"]))
            all_rows.append(metric)
    _annotate_slice_level(all_rows, slice_means)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    table = write_qc_table(all_rows, output / "acquisition_qc.csv")
    summary = output / "acquisition_qc.json"
    payload = {"schema_version": 1, "method": "within-block median+MAD acquisition corruption QC",
               "evaluated_frames": len(all_rows), "valid_frames": sum(row["qc_valid"] for row in all_rows),
               "invalid_frames": sum(not row["qc_valid"] for row in all_rows),
               "evaluated_slices": len(groups), "slice_keys": [f"{v}/{s}" for v, s in sorted(groups)]}
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    image = output / "acquisition_qc.png"; _render_metrics(all_rows, image)
    return table, summary, image


def _annotate_slice_level(rows: list[dict[str, Any]], slice_means: dict[tuple[str, str], np.ndarray]) -> None:
    """Conservatively flag only joint intensity-and-neighbour-consistency extremes."""
    for view in sorted({key[0] for key in slice_means}):
        keys = sorted(key for key in slice_means if key[0] == view)
        if len(keys) < 3: continue
        intensities = np.asarray([float(np.median(slice_means[key])) for key in keys])
        center, spread = _median_mad(intensities, 0.01)
        for index in range(1, len(keys) - 1):
            key, image = keys[index], slice_means[keys[index]]
            neighbour = (slice_means[keys[index - 1]] + slice_means[keys[index + 1]]) / 2.0
            ncc = np.corrcoef(image.ravel(), neighbour.ravel())[0, 1]
            if abs(intensities[index] - center) > 8.0 * spread and np.isfinite(ncc) and ncc < 0.2:
                for row in rows:
                    if (row["view"], row["slice_id"]) == key:
                        row["qc_valid"] = False; row["qc_reason"] += ";slice_location_outlier"


def _render_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib; matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    x = np.arange(len(rows)); valid = np.asarray([row["qc_valid"] for row in rows])
    for axis, field, title in zip(axes, ("qc_ncc", "qc_intensity_scale", "qc_residual"), ("NCC", "global scale", "corrected residual")):
        values = np.asarray([row[field] for row in rows]); axis.plot(x, values, linewidth=1); axis.scatter(x[~valid], values[~valid], color="red", s=18); axis.set_ylabel(title)
    axes[-1].set_xlabel("selected-frame table index"); figure.savefig(path, dpi=160); plt.close(figure)


def _median_mad(values: np.ndarray, floor: float) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center))) * 1.4826
    return center, max(mad, floor)
