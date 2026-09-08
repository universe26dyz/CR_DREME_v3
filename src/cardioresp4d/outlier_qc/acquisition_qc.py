"""功能：Phase-1 acquisition outlier QC（逐帧 + 整个 fixed-slice location + 手动排除）。

核心目标
--------
1. 保留原有的 frame-level QC：
   - 50 帧 temporal median reference
   - NCC
   - global intensity scale
   - scale-corrected residual
   - block 内 median + MAD

2. 修正 whole-slice-location QC：
   - 不再使用 whole-image median + 全 stack 8 MAD
   - 对每个 fixed slice 的 temporal mean 构建相邻层 local reference
   - 使用 foreground-aware local intensity scale
   - 使用 local NCC
   - 使用 scale-corrected mean residual + p95 residual
   - 支持连续 2 个异常层（例如 SAX s16/s17）：
     默认使用 ±3 层邻域并对邻层图像做 pixelwise median

3. 新增手动排除接口：
   - 支持 view/slice：SAX/s16
   - 支持 slice 名：s16、SAX_s16
   - 支持文件夹名：例如 s16
   - 支持完整/部分目录路径
   - 支持完整文件路径 / 文件名 / stem
   - 支持 glob：例如 */s16/* 或 *SAX_s16*
   - 可通过重复 --manual-exclude 或文本文件 --manual-exclude-file 传入
   - 手动路径只在 runtime 中用于匹配，不写入 PHI-free QC CSV/JSON

4. 不删除、不修改任何原始 DICOM。
5. 整个 slice 被判 invalid 时，不填 0、不复制邻层、不伪造 replacement。
   本文件只生成 QC decision；后续 reference/training 应只消费 valid observations。

论文来源
--------
- DSVR-style robust slice/voxel rejection
- NeSVoR-style robust weighting / intensity handling
- 本项目 image-domain acquisition corruption 的 necessary adaptation

CLI 示例
--------
仅自动 QC：
python -m cardioresp4d.outlier_qc.acquisition_qc \
  --manifest results/dicom_manifest.csv \
  --output-dir results/acquisition_qc

手动排除 SAX s16/s17：
python -m cardioresp4d.outlier_qc.acquisition_qc \
  --manifest results/dicom_manifest.csv \
  --output-dir results/acquisition_qc \
  --manual-exclude SAX/s16 \
  --manual-exclude SAX/s17

使用文本文件：
python -m cardioresp4d.outlier_qc.acquisition_qc \
  --manifest results/dicom_manifest.csv \
  --output-dir results/acquisition_qc \
  --manual-exclude-file configs/manual_exclusions.txt

定向检查：
python -m cardioresp4d.outlier_qc.acquisition_qc \
  --manifest results/dicom_manifest.csv \
  --output-dir results/acquisition_qc_debug \
  --slice-key SAX/s13 --slice-key SAX/s14 --slice-key SAX/s15 \
  --slice-key SAX/s16 --slice-key SAX/s17 --slice-key SAX/s18 \
  --slice-key SAX/s19 --slice-key SAX/s20

注意：
定向 subset QC 仅用于诊断，不应直接作为完整 downstream QC table。
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cardioresp4d.data.dataset import CardioRespDataset, validate_qc_table_coverage


# -----------------------------------------------------------------------------
# 置 invalid 的 reason 白名单
# -----------------------------------------------------------------------------
# 只有这里列出的 reason 会把 qc_valid / slice_qc_valid 置为 False；
# 其他 reason 只写入 qc_reason / slice_qc_reason 做诊断记录，但保持 valid。
INVALID_REASON_WHITELIST = frozenset({
    "slice_local_scale_absolute",
    "manual_exclusion",
})


def is_hard_invalid_reason_text(reason_text: str | None) -> bool:
    """Return whether a semicolon-delimited QC reason triggers hard exclusion."""
    if reason_text in (None, "", "valid"):
        return False
    return any(reason in INVALID_REASON_WHITELIST for reason in str(reason_text).split(";"))


# -----------------------------------------------------------------------------
# 输出 schema
# -----------------------------------------------------------------------------

QC_COLUMNS = (
    "source_file_token",
    "view",
    "slice_id",
    "frame_index",
    "qc_valid",
    "qc_reason",
    "qc_ncc",
    "qc_intensity_scale",
    "qc_residual",
    "rescale_status",
)

SLICE_QC_COLUMNS = (
    "view",
    "slice_id",
    "position_mm",
    "n_total_frames",
    "n_frame_qc_valid",
    "n_manual_excluded_frames",
    "n_effective_valid_frames",
    "foreground_intensity",
    "local_reference_intensity",
    "local_scale",
    "local_ncc",
    "local_residual_mean",
    "local_residual_p95",
    "neighbor_count",
    "slice_qc_valid",
    "slice_qc_reason",
    "manual_slice_excluded",
)


# -----------------------------------------------------------------------------
# Frame-level QC
# -----------------------------------------------------------------------------

def analyze_frame_block(
    images: np.ndarray,
    rescale_statuses: Iterable[str],
    *,
    ncc_mad_threshold: float = 6.0,
    scale_mad_threshold: float = 6.0,
    residual_mad_threshold: float = 6.0,
    mad_floor: float = 1e-6,
) -> list[dict[str, Any]]:
    """对单个 fixed slice 的 50 帧执行逐帧 acquisition corruption QC。

    这一层只负责检测“50 帧中少数 frame 异常”。
    对于“整个 location 的 50 帧一起偏暗/一起 corrupted”，
    必须依赖后面的 slice-location QC。
    """
    values = np.asarray(images, dtype=np.float64)
    statuses = list(rescale_statuses)

    if values.ndim != 3 or values.shape[0] != 50 or not np.isfinite(values).all():
        raise ValueError("images must be a finite 50-frame (time,row,column) block")
    if len(statuses) != 50:
        raise ValueError("rescale_statuses must contain exactly 50 entries")

    thresholds = (
        ncc_mad_threshold,
        scale_mad_threshold,
        residual_mad_threshold,
        mad_floor,
    )
    if any(not np.isfinite(x) or x <= 0 for x in thresholds):
        raise ValueError("QC thresholds and mad_floor must be finite and positive")

    # 50 帧自己的 temporal median reference：
    # 适合少量坏帧；不适合 whole-slice-location 一起坏。
    reference = np.median(values, axis=0)

    # 去掉最低信号背景，避免纯背景主导 NCC / scale。
    abs_ref = np.abs(reference)
    threshold = max(float(np.percentile(abs_ref, 25.0)), np.finfo(float).eps)
    mask = abs_ref > threshold
    if not np.any(mask):
        mask = np.ones(reference.shape, dtype=bool)

    ref_vector = reference[mask]
    ref_centered = ref_vector - np.mean(ref_vector)
    ref_norm = max(float(np.linalg.norm(ref_centered)), np.finfo(float).eps)

    scales: list[float] = []
    nccs: list[float] = []
    residuals: list[float] = []

    for frame in values:
        frame_vector = frame[mask]

        ratios = frame_vector / np.where(
            np.abs(ref_vector) > np.finfo(float).eps,
            ref_vector,
            np.nan,
        )
        finite_ratios = ratios[np.isfinite(ratios)]
        scale = float(np.median(finite_ratios)) if finite_ratios.size else 1.0
        scale = max(scale, np.finfo(float).eps)

        centered = frame_vector - np.mean(frame_vector)
        denom = max(
            float(np.linalg.norm(centered)) * ref_norm,
            np.finfo(float).eps,
        )
        ncc = float(np.dot(centered, ref_centered) / denom)

        corrected = frame / scale
        residual = float(
            np.mean(np.abs(corrected - reference))
            / max(float(np.mean(np.abs(reference))), np.finfo(float).eps)
        )

        scales.append(scale)
        nccs.append(ncc)
        residuals.append(residual)

    log_scales = np.log(np.asarray(scales, dtype=np.float64))
    ncc_array = np.asarray(nccs, dtype=np.float64)
    residual_array = np.asarray(residuals, dtype=np.float64)

    scale_center, scale_spread = _median_mad(
        log_scales, max(mad_floor, 0.01)
    )
    ncc_center, ncc_spread = _median_mad(
        ncc_array, max(mad_floor, 0.005)
    )
    residual_center, residual_spread = _median_mad(
        residual_array, max(mad_floor, 0.005)
    )

    output: list[dict[str, Any]] = []
    for index, (scale, ncc, residual) in enumerate(
        zip(scales, nccs, residuals)
    ):
        reasons: list[str] = []

        # MAD + 大幅绝对变化 guard。
        if abs(np.log(scale) - scale_center) > max(
            scale_mad_threshold * scale_spread,
            0.25,
        ):
            reasons.append("global_intensity_scale")

        if ncc < ncc_center - ncc_mad_threshold * ncc_spread:
            reasons.append("low_ncc")

        if residual > residual_center + residual_mad_threshold * residual_spread:
            reasons.append("scale_corrected_residual")

        output.append(
            {
                "qc_valid": not is_hard_invalid_reason_text(";".join(reasons)),
                "qc_reason": "valid" if not reasons else ";".join(reasons),
                "qc_ncc": ncc,
                "qc_intensity_scale": scale,
                "qc_residual": residual,
                "rescale_status": str(statuses[index]),
            }
        )

    return output


# -----------------------------------------------------------------------------
# Manual exclusion
# -----------------------------------------------------------------------------

def load_manual_exclusions(
    selectors: Sequence[str] | None = None,
    exclusion_file: str | Path | None = None,
) -> list[str]:
    """加载手动排除 selector。

    exclusion_file:
      - 每行一个 selector
      - 空行忽略
      - # 开头为注释
    """
    merged: list[str] = []

    if selectors:
        merged.extend(str(x).strip() for x in selectors if str(x).strip())

    if exclusion_file is not None:
        path = Path(exclusion_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Manual exclusion file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            merged.append(item)

    # 去重但保序。
    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in merged:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)

    return deduplicated


def _slice_aliases(view: str, slice_id: str) -> set[str]:
    """生成用户容易输入的 slice aliases，例如 SAX_s16 -> s16 / SAX/s16。"""
    view_cf = str(view).strip().casefold()
    sid_cf = str(slice_id).strip().casefold()

    aliases = {
        sid_cf,
        f"{view_cf}/{sid_cf}",
    }

    prefix = f"{view_cf}_"
    if sid_cf.startswith(prefix):
        short = sid_cf[len(prefix):]
        aliases.add(short)
        aliases.add(f"{view_cf}/{short}")

    match = re.search(r"(s\d+)$", sid_cf)
    if match:
        short = match.group(1)
        aliases.add(short)
        aliases.add(f"{view_cf}/{short}")

    return aliases


def _normalise_path_text(value: str) -> str:
    value = value.replace("\\", os.sep).replace("/", os.sep)
    return os.path.normcase(os.path.normpath(value)).casefold()


def _manual_selector_matches(
    selector: str,
    *,
    view: str,
    slice_id: str,
    runtime_path: str | None,
) -> bool:
    """判断一个 selector 是否命中一个 manifest frame。

    匹配原则：
    1. exact slice alias：
       SAX/s16, s16, SAX_s16
    2. bare folder/file component exact match
    3. path selector：
       完整路径或目录 prefix
    4. glob：
       *SAX_s16* / */s16/*
    """
    raw = selector.strip()
    if not raw:
        return False

    selector_cf = raw.casefold()
    aliases = _slice_aliases(view, slice_id)

    # 先匹配 view/slice aliases。
    if selector_cf in aliases:
        return True

    runtime = str(runtime_path or "")
    path_cf = _normalise_path_text(runtime) if runtime else ""
    path_obj = Path(runtime) if runtime else None

    # glob selector
    if any(ch in raw for ch in "*?[]"):
        targets = set(aliases)
        if runtime:
            targets.add(runtime.casefold())
            targets.add(path_cf)
            targets.add(path_obj.name.casefold())
            targets.add(path_obj.stem.casefold())
            targets.update(part.casefold() for part in path_obj.parts)
        return any(fnmatch.fnmatchcase(target, selector_cf) for target in targets)

    # bare token：只做 exact component / filename / stem，不做危险 substring。
    contains_separator = "/" in raw or "\\" in raw
    if not contains_separator:
        if runtime and path_obj is not None:
            if selector_cf == path_obj.name.casefold():
                return True
            if selector_cf == path_obj.stem.casefold():
                return True
            if any(selector_cf == part.casefold() for part in path_obj.parts):
                return True
        return False

    # 路径 selector：
    # - 完整文件路径 exact
    # - 目录 prefix（该目录下所有文件）
    selector_path_cf = _normalise_path_text(raw)
    if path_cf:
        if path_cf == selector_path_cf:
            return True
        prefix = selector_path_cf.rstrip(os.sep) + os.sep
        if path_cf.startswith(prefix):
            return True

    return False


def _prepare_manual_matches(
    dataset: CardioRespDataset,
    selectors: Sequence[str],
) -> dict[int, bool]:
    """提前解析全部 selector；任何 selector 0 命中时直接报错，防止静默漏排。"""
    if not selectors:
        return {}

    matches: dict[int, bool] = {}
    counts = {selector: 0 for selector in selectors}

    for index, row in enumerate(dataset._rows):
        token = row.get("source_file_token", "")
        runtime_path = dataset._token_to_path.get(
            token,
            row.get("dicom_path", ""),
        )

        hit = False
        for selector in selectors:
            if _manual_selector_matches(
                selector,
                view=row["view"],
                slice_id=row["slice_id"],
                runtime_path=runtime_path,
            ):
                counts[selector] += 1
                hit = True
        matches[index] = hit

    missing = [selector for selector, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            "Manual exclusion selector(s) matched zero manifest frames: "
            + ", ".join(missing)
        )

    return matches


def apply_manual_exclusions(
    manifest_path: str | Path,
    existing_qc_table: str | Path,
    output_dir: str | Path,
    *,
    manual_exclude: Sequence[str] | None = None,
    manual_exclude_file: str | Path | None = None,
) -> tuple[Path, Path]:
    """Propagate formal manual exclusions without rerunning automatic QC.

    This is intentionally narrower than :func:`run_acquisition_qc`: it first
    requires an existing complete table, resolves selectors against original
    manifest IDs/runtime paths, and changes only matched rows by appending the
    hard-whitelisted ``manual_exclusion`` reason.
    """
    manifest_path, existing_qc_table, output_dir = Path(manifest_path), Path(existing_qc_table), Path(output_dir)
    validate_qc_table_coverage(manifest_path, existing_qc_table)
    dataset = CardioRespDataset(manifest_path, valid_only=False, use_qc=False)
    selectors = load_manual_exclusions(manual_exclude, manual_exclude_file)
    matches = _prepare_manual_matches(dataset, selectors)
    with existing_qc_table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle)); fields = list(rows[0])
    token_mode = "source_file_token" in dataset._rows[0]
    by_key = {(row.get("source_file_token", ""),) if token_mode else (row["view"], row["slice_id"], row["frame_index"]): row for row in rows}
    changed = 0
    for index, matched in matches.items():
        if not matched: continue
        source = dataset._rows[index]; key = (source.get("source_file_token", ""),) if token_mode else (source["view"], source["slice_id"], source["frame_index"])
        row = by_key[key]; row["qc_valid"] = "False"; row["qc_reason"] = _append_reason(str(row.get("qc_reason", "valid")), "manual_exclusion"); changed += 1
    output_dir.mkdir(parents=True, exist_ok=True); table = output_dir / "acquisition_qc.csv"
    with table.open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = output_dir / "manual_exclusion_summary.json"
    summary.write_text(json.dumps({"selectors": selectors, "matched_frames": changed, "total_frames": len(rows), "method": "formal_manual_exclusion_propagation_without_automatic_qc_rerun"}, indent=2) + "\n", encoding="utf-8")
    return table, summary


# -----------------------------------------------------------------------------
# Slice-location QC
# -----------------------------------------------------------------------------

def _foreground_mask(
    reference: np.ndarray,
    foreground_percentile: float,
) -> np.ndarray:
    """由 local reference 构建 foreground-aware mask。

    不做 per-slice percentile normalization，只用 percentile 决定“在哪些像素上比较”。
    因此不会抹掉真实的 absolute/local intensity drop。
    """
    ref = np.asarray(reference, dtype=np.float64)
    abs_ref = np.abs(ref)
    finite = abs_ref[np.isfinite(abs_ref)]
    if finite.size == 0:
        raise ValueError("Local reference contains no finite pixels")

    threshold = float(np.percentile(finite, foreground_percentile))
    threshold = max(threshold, np.finfo(float).eps)
    mask = abs_ref > threshold

    # 极端情况下至少保留非零像素。
    if np.count_nonzero(mask) < 32:
        mask = abs_ref > np.finfo(float).eps

    if not np.any(mask):
        mask = np.ones(ref.shape, dtype=bool)

    return mask


def _local_pair_metrics(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    foreground_percentile: float,
) -> dict[str, float]:
    """计算一个 slice 与 local neighbor reference 之间的透明指标。"""
    image = np.asarray(image, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if image.shape != reference.shape:
        raise ValueError(
            f"Slice/local-reference shape mismatch: {image.shape} vs {reference.shape}"
        )
    if not np.isfinite(image).all() or not np.isfinite(reference).all():
        raise ValueError("Slice/local-reference images must be finite")

    mask = _foreground_mask(reference, foreground_percentile)

    image_vec = image[mask]
    ref_vec = reference[mask]

    # local intensity scale：同一 foreground 坐标下 image/reference 的 robust median。
    valid_ratio = np.abs(ref_vec) > np.finfo(float).eps
    ratios = image_vec[valid_ratio] / ref_vec[valid_ratio]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    if ratios.size < 16:
        scale = 1.0
    else:
        scale = float(np.median(ratios))
    scale = max(scale, np.finfo(float).eps)

    foreground_intensity = float(np.median(np.abs(image_vec)))
    local_reference_intensity = float(np.median(np.abs(ref_vec)))

    image_centered = image_vec - np.mean(image_vec)
    ref_centered = ref_vec - np.mean(ref_vec)
    denom = max(
        float(np.linalg.norm(image_centered))
        * float(np.linalg.norm(ref_centered)),
        np.finfo(float).eps,
    )
    ncc = float(np.dot(image_centered, ref_centered) / denom)

    corrected_vec = image_vec / scale
    residual_abs = np.abs(corrected_vec - ref_vec)
    reference_scale = max(
        float(np.mean(np.abs(ref_vec))),
        np.finfo(float).eps,
    )

    residual_mean = float(np.mean(residual_abs) / reference_scale)
    residual_p95 = float(np.percentile(residual_abs, 95.0) / reference_scale)

    return {
        "foreground_intensity": foreground_intensity,
        "local_reference_intensity": local_reference_intensity,
        "local_scale": scale,
        "local_ncc": ncc,
        "local_residual_mean": residual_mean,
        "local_residual_p95": residual_p95,
    }


def _nearest_neighbor_indices(
    n: int,
    index: int,
    *,
    radius: int,
    eligible: Sequence[bool],
    min_neighbors: int,
) -> list[int]:
    """优先使用 ±radius；边界处不足时再补最近 eligible slices。"""
    selected = [
        j
        for j in range(max(0, index - radius), min(n, index + radius + 1))
        if j != index and eligible[j]
    ]

    if len(selected) < min_neighbors:
        extras = sorted(
            (
                j
                for j in range(n)
                if j != index and eligible[j] and j not in selected
            ),
            key=lambda j: (abs(j - index), j),
        )
        for j in extras:
            selected.append(j)
            if len(selected) >= min_neighbors:
                break

    return sorted(selected)


def _append_reason(existing: str, reason: str) -> str:
    parts = [] if existing in ("", "valid", None) else str(existing).split(";")
    if reason not in parts:
        parts.append(reason)
    return "valid" if not parts else ";".join(parts)


def _analyze_slice_locations(
    rows: list[dict[str, Any]],
    slice_means: dict[tuple[str, str], np.ndarray],
    positions_mm: dict[tuple[str, str], float],
    *,
    slice_counts: dict[tuple[str, str], dict[str, int]],
    reference_eligible: dict[tuple[str, str], bool],
    manual_full_slice: set[tuple[str, str]],
    neighbor_radius: int = 3,
    min_neighbors: int = 3,
    foreground_percentile: float = 70.0,
    slice_scale_low: float = 0.70,
    slice_scale_high: float = 1.40,
    slice_scale_mad_threshold: float = 6.0,
    slice_scale_min_log_excursion: float = 0.15,
    slice_ncc_mad_threshold: float = 6.0,
    slice_residual_mad_threshold: float = 6.0,
    mad_floor: float = 1e-6,
) -> list[dict[str, Any]]:
    """真正的 whole-slice-location QC。

    判定规则保持透明：
    A. local_scale 超出硬阈值 [slice_scale_low, slice_scale_high] -> 记录 reason
    B. local_scale 是 robust MAD outlier，且相对 1 的绝对变化至少达到
       slice_scale_min_log_excursion -> 记录 reason
    C. local NCC 显著低 + local p95 residual 显著高，同时成立 -> 记录 reason
    D. 手动完整 slice exclusion -> 记录 reason

    结构指标必须 NCC + residual 同时异常，减少 base/apex 正常解剖变化误判。

    注意：本函数只负责检测并记录所有 reason。最终哪些 reason 会置 invalid
    由 INVALID_REASON_WHITELIST 统一决定（在 run_acquisition_qc 末尾归一化）。
    当前白名单：slice_local_scale_absolute、manual_exclusion。
    """
    if neighbor_radius < 1:
        raise ValueError("neighbor_radius must be >= 1")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1")
    if not (0.0 < foreground_percentile < 100.0):
        raise ValueError("foreground_percentile must be in (0, 100)")
    if not (0.0 < slice_scale_low < 1.0 < slice_scale_high):
        raise ValueError("slice_scale_low/high must satisfy 0 < low < 1 < high")

    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_key[(row["view"], row["slice_id"])].append(row)

    slice_results: list[dict[str, Any]] = []

    for view in sorted({key[0] for key in slice_means}):
        keys = sorted(
            (key for key in slice_means if key[0] == view),
            key=positions_mm.__getitem__,
        )
        n = len(keys)
        if n == 0:
            continue

        eligible = [bool(reference_eligible.get(key, False)) for key in keys]

        provisional: list[dict[str, Any]] = []

        for index, key in enumerate(keys):
            neighbor_indices = _nearest_neighbor_indices(
                n,
                index,
                radius=neighbor_radius,
                eligible=eligible,
                min_neighbors=min_neighbors,
            )

            counts = slice_counts[key]
            base = {
                "view": key[0],
                "slice_id": key[1],
                "position_mm": float(positions_mm[key]),
                "n_total_frames": int(counts["total"]),
                "n_frame_qc_valid": int(counts["frame_qc_valid"]),
                "n_manual_excluded_frames": int(counts["manual_excluded"]),
                "n_effective_valid_frames": int(counts["effective_valid"]),
                "neighbor_count": len(neighbor_indices),
                "manual_slice_excluded": int(key in manual_full_slice),
            }

            if len(neighbor_indices) < min_neighbors:
                provisional.append(
                    {
                        **base,
                        "foreground_intensity": np.nan,
                        "local_reference_intensity": np.nan,
                        "local_scale": np.nan,
                        "local_ncc": np.nan,
                        "local_residual_mean": np.nan,
                        "local_residual_p95": np.nan,
                        "slice_qc_valid": key not in manual_full_slice,
                        "slice_qc_reason": (
                            "manual_exclusion"
                            if key in manual_full_slice
                            else "not_evaluated_insufficient_neighbors"
                        ),
                    }
                )
                continue

            local_reference = np.median(
                np.stack([slice_means[keys[j]] for j in neighbor_indices]),
                axis=0,
            )

            metrics = _local_pair_metrics(
                slice_means[key],
                local_reference,
                foreground_percentile=foreground_percentile,
            )

            provisional.append(
                {
                    **base,
                    **metrics,
                    "slice_qc_valid": True,
                    "slice_qc_reason": "valid",
                }
            )

        # 同一 view 内建立 robust distribution，仅用于 moderate outlier / structure 判定。
        evaluable = [
            item
            for item in provisional
            if np.isfinite(float(item["local_scale"]))
        ]

        if evaluable:
            log_scales = np.log(
                np.asarray([item["local_scale"] for item in evaluable], dtype=float)
            )
            nccs = np.asarray(
                [item["local_ncc"] for item in evaluable], dtype=float
            )
            p95s = np.asarray(
                [item["local_residual_p95"] for item in evaluable], dtype=float
            )

            scale_center, scale_spread = _median_mad(
                log_scales, max(mad_floor, 0.02)
            )
            ncc_center, ncc_spread = _median_mad(
                nccs, max(mad_floor, 0.01)
            )
            p95_center, p95_spread = _median_mad(
                p95s, max(mad_floor, 0.02)
            )
        else:
            scale_center = 0.0
            scale_spread = max(mad_floor, 0.02)
            ncc_center = 1.0
            ncc_spread = max(mad_floor, 0.01)
            p95_center = 0.0
            p95_spread = max(mad_floor, 0.02)

        for item in provisional:
            key = (item["view"], item["slice_id"])
            reasons: list[str] = []

            if key in manual_full_slice:
                reasons.append("manual_exclusion")

            scale = float(item["local_scale"])
            ncc = float(item["local_ncc"])
            p95 = float(item["local_residual_p95"])

            if np.isfinite(scale):
                hard_scale_bad = scale < slice_scale_low or scale > slice_scale_high

                log_scale = float(np.log(max(scale, np.finfo(float).eps)))
                robust_scale_bad = (
                    abs(log_scale - scale_center)
                    > slice_scale_mad_threshold * scale_spread
                    and abs(log_scale) >= slice_scale_min_log_excursion
                )

                if hard_scale_bad:
                    reasons.append("slice_local_scale_absolute")
                elif robust_scale_bad:
                    reasons.append("slice_local_scale_robust")

            if np.isfinite(ncc) and np.isfinite(p95):
                ncc_bad = (
                    ncc
                    < ncc_center
                    - slice_ncc_mad_threshold * ncc_spread
                )
                residual_bad = (
                    p95
                    > p95_center
                    + slice_residual_mad_threshold * p95_spread
                )
                if ncc_bad and residual_bad:
                    reasons.append("slice_local_structure")

            # insufficient neighbors 不自动 reject。
            if item["slice_qc_reason"] == "not_evaluated_insufficient_neighbors":
                if not reasons:
                    item["slice_qc_valid"] = True
                    slice_results.append(item)
                    continue

            # 先记录所有 reason（即使 reason 不在白名单，也要写入 qc_reason 做诊断）。
            if reasons:
                item["slice_qc_reason"] = ";".join(reasons)

            # 只有 INVALID_REASON_WHITELIST 里的 reason 才置 invalid。
            invalid_reasons = [
                r for r in reasons if r in INVALID_REASON_WHITELIST
            ]
            item["slice_qc_valid"] = not invalid_reasons

            # 反向覆盖该 location 的所有 frame：
            # 所有 reason 都追加到 qc_reason（保留诊断），
            # 但只有白名单 reason 才把 qc_valid 置为 False。
            for row in rows_by_key[key]:
                for reason in reasons:
                    row["qc_reason"] = _append_reason(row["qc_reason"], reason)
                if invalid_reasons:
                    row["qc_valid"] = False
            slice_results.append(item)

    return slice_results


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def write_qc_table(
    rows: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """写逐帧 PHI-free QC CSV。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=QC_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["qc_valid"] = int(bool(payload["qc_valid"]))
            writer.writerow(payload)

    return path


def write_slice_qc_table(
    rows: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """写每个 fixed-slice location 一行的诊断表。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SLICE_QC_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            payload = dict(row)
            payload["slice_qc_valid"] = int(bool(payload["slice_qc_valid"]))
            writer.writerow(payload)

    return path


# -----------------------------------------------------------------------------
# Main QC API
# -----------------------------------------------------------------------------

def run_acquisition_qc(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    slice_keys: set[str] | None = None,
    ncc_mad_threshold: float = 6.0,
    scale_mad_threshold: float = 6.0,
    residual_mad_threshold: float = 6.0,
    mad_floor: float = 1e-6,
    manual_exclude: Sequence[str] | None = None,
    manual_exclude_file: str | Path | None = None,
    slice_neighbor_radius: int = 3,
    slice_min_neighbors: int = 3,
    slice_foreground_percentile: float = 70.0,
    slice_scale_low: float = 0.70,
    slice_scale_high: float = 1.40,
    slice_scale_mad_threshold: float = 6.0,
    slice_scale_min_log_excursion: float = 0.15,
    slice_ncc_mad_threshold: float = 6.0,
    slice_residual_mad_threshold: float = 6.0,
) -> tuple[Path, Path, Path]:
    """Phase-1 acquisition QC 主入口。

    Returns 保持旧接口兼容：
        (acquisition_qc.csv, acquisition_qc.json, acquisition_qc.png)

    另外新增输出：
        slice_location_qc.csv
        slice_location_qc.png
    """
    dataset = CardioRespDataset(
        manifest_path,
        valid_only=False,
        use_qc=False,
    )

    selectors = load_manual_exclusions(
        manual_exclude,
        manual_exclude_file,
    )
    manual_by_index = _prepare_manual_matches(dataset, selectors)

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset._rows):
        key = (row["view"], row["slice_id"])

        if (
            slice_keys is None
            or f"{key[0]}/{key[1]}" in slice_keys
            or any(
                candidate in {x.casefold() for x in slice_keys}
                for candidate in _slice_aliases(key[0], key[1])
            )
        ):
            groups[key].append(index)

    if not groups:
        raise ValueError("No manifest slice blocks selected for acquisition QC")

    all_rows: list[dict[str, Any]] = []

    # slice_means 同时用于：
    # - 当前 slice 作为被检测对象
    # - 若 reference_eligible=True，可作为其它 slice 的 local reference
    slice_means: dict[tuple[str, str], np.ndarray] = {}
    reference_eligible: dict[tuple[str, str], bool] = {}
    manual_full_slice: set[tuple[str, str]] = set()
    slice_counts: dict[tuple[str, str], dict[str, int]] = {}

    for (view, slice_id), indices in sorted(groups.items()):
        indices.sort(key=lambda i: int(dataset._rows[i]["frame_index"]))

        if len(indices) != 50:
            raise ValueError(
                f"{view}/{slice_id} must contain exactly 50 frames before QC"
            )

        samples = [dataset[i] for i in indices]
        images = np.stack([sample["rescaled_image"] for sample in samples])

        metrics = analyze_frame_block(
            images,
            [sample["rescale_status"] for sample in samples],
            ncc_mad_threshold=ncc_mad_threshold,
            scale_mad_threshold=scale_mad_threshold,
            residual_mad_threshold=residual_mad_threshold,
            mad_floor=mad_floor,
        )

        # manual exclusion 与 automatic frame QC 分开统计。
        manual_flags = [
            bool(manual_by_index.get(index, False))
            for index in indices
        ]
        auto_valid_flags = [bool(metric["qc_valid"]) for metric in metrics]
        effective_flags = [
            auto_valid and not manual
            for auto_valid, manual in zip(auto_valid_flags, manual_flags)
        ]

        key = (view, slice_id)

        if all(manual_flags):
            manual_full_slice.add(key)

        # 对 local QC：
        # 优先使用最终 effective valid frames。
        # 若整层手动排除，仍使用 auto-valid temporal mean 作为“诊断 target”，
        # 但 reference_eligible=False，绝不会拿它给别的层当正常 reference。
        if any(effective_flags):
            mean_source = [
                image
                for image, valid in zip(images, effective_flags)
                if valid
            ]
            reference_eligible[key] = True
        elif any(auto_valid_flags):
            mean_source = [
                image
                for image, valid in zip(images, auto_valid_flags)
                if valid
            ]
            reference_eligible[key] = False
        else:
            # 50 帧自动 frame-QC 全部失败：
            # 不伪造 replacement；这里只保留 temporal mean 做 QC diagnostic，
            # 并明确不可作为 neighbor reference。
            mean_source = list(images)
            reference_eligible[key] = False

        slice_means[key] = np.mean(np.stack(mean_source), axis=0)

        slice_counts[key] = {
            "total": 50,
            "frame_qc_valid": int(sum(auto_valid_flags)),
            "manual_excluded": int(sum(manual_flags)),
            "effective_valid": int(sum(effective_flags)),
        }

        for metric, index, manual in zip(metrics, indices, manual_flags):
            row = dataset._rows[index]

            if manual:
                metric["qc_valid"] = False
                metric["qc_reason"] = _append_reason(
                    metric["qc_reason"],
                    "manual_exclusion",
                )

            metric.update(
                source_file_token=row.get("source_file_token", ""),
                view=view,
                slice_id=slice_id,
                frame_index=int(row["frame_index"]),
            )
            all_rows.append(metric)

    slice_positions = {
        key: _slice_position_mm(dataset._rows[indices[0]])
        for key, indices in groups.items()
    }

    slice_rows = _analyze_slice_locations(
        all_rows,
        slice_means,
        slice_positions,
        slice_counts=slice_counts,
        reference_eligible=reference_eligible,
        manual_full_slice=manual_full_slice,
        neighbor_radius=slice_neighbor_radius,
        min_neighbors=slice_min_neighbors,
        foreground_percentile=slice_foreground_percentile,
        slice_scale_low=slice_scale_low,
        slice_scale_high=slice_scale_high,
        slice_scale_mad_threshold=slice_scale_mad_threshold,
        slice_scale_min_log_excursion=slice_scale_min_log_excursion,
        slice_ncc_mad_threshold=slice_ncc_mad_threshold,
        slice_residual_mad_threshold=slice_residual_mad_threshold,
        mad_floor=mad_floor,
    )

    # 最终归一化：根据白名单重算所有 frame 的 qc_valid。
    # 内部流程（analyze_frame_block / slice_mean 构建）仍使用原始 qc_valid
    # 保证 slice-level 检测精度，但对外输出时只有白名单 reason 会置 invalid。
    for row in all_rows:
        reason_text = str(row.get("qc_reason", "valid"))
        parts = (
            [] if reason_text in ("", "valid", None) else reason_text.split(";")
        )
        row["qc_valid"] = not is_hard_invalid_reason_text(reason_text)

    # slice-level 同理：根据白名单重算 slice_qc_valid。
    for row in slice_rows:
        reason_text = str(row.get("slice_qc_reason", "valid"))
        parts = (
            [] if reason_text in ("", "valid", None) else reason_text.split(";")
        )
        row["slice_qc_valid"] = not is_hard_invalid_reason_text(reason_text)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 逐帧 QC：保持旧 downstream 文件名。
    table = write_qc_table(
        all_rows,
        output / "acquisition_qc.csv",
    )

    # 新增 slice-location diagnostic。
    slice_table = write_slice_qc_table(
        slice_rows,
        output / "slice_location_qc.csv",
    )

    frame_plot = output / "acquisition_qc.png"
    _render_frame_metrics(all_rows, frame_plot)

    slice_plot = output / "slice_location_qc.png"
    _render_slice_metrics(slice_rows, slice_plot)

    summary = output / "acquisition_qc.json"

    invalid_slice_rows = [
        row for row in slice_rows if not bool(row["slice_qc_valid"])
    ]
    manual_frame_count = sum(
        "manual_exclusion" in str(row["qc_reason"]).split(";")
        for row in all_rows
    )

    payload = {
        "schema_version": 2,
        "method": (
            "frame temporal-median/MAD QC + "
            "foreground-aware local-neighbor slice-location QC"
        ),
        "evaluated_frames": len(all_rows),
        "valid_frames": int(sum(bool(row["qc_valid"]) for row in all_rows)),
        "invalid_frames": int(sum(not bool(row["qc_valid"]) for row in all_rows)),
        "manual_excluded_frames": int(manual_frame_count),
        "evaluated_slices": len(groups),
        "invalid_slices": len(invalid_slice_rows),
        "invalid_slice_keys": [
            f"{row['view']}/{row['slice_id']}"
            for row in invalid_slice_rows
        ],
        "slice_keys": [
            f"{view}/{slice_id}"
            for view, slice_id in sorted(groups)
        ],
        "slice_location_qc": {
            "neighbor_radius": slice_neighbor_radius,
            "min_neighbors": slice_min_neighbors,
            "foreground_percentile": slice_foreground_percentile,
            "hard_scale_range": [slice_scale_low, slice_scale_high],
            "scale_mad_threshold": slice_scale_mad_threshold,
            "scale_min_log_excursion": slice_scale_min_log_excursion,
            "ncc_mad_threshold": slice_ncc_mad_threshold,
            "residual_mad_threshold": slice_residual_mad_threshold,
            "table": slice_table.name,
            "plot": slice_plot.name,
        },
        "manual_exclusion": {
            "enabled": bool(selectors),
            # 只记录 selector 数量，不把可能包含 PHI 的路径写进 durable summary。
            "selector_count": len(selectors),
        },
    }

    summary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return table, summary, frame_plot


# -----------------------------------------------------------------------------
# Geometry helper
# -----------------------------------------------------------------------------

def _slice_position_mm(row: dict[str, str]) -> float:
    """由 DICOM IOP/IPP 计算该 slice 的 patient-world plane position。"""
    orientation = np.asarray(
        json.loads(row["image_orientation_patient"]),
        dtype=float,
    )
    origin = np.asarray(
        json.loads(row["image_position_patient"]),
        dtype=float,
    )

    normal = np.cross(
        orientation[:3],
        orientation[3:],
    )
    norm = float(np.linalg.norm(normal))
    if norm <= np.finfo(float).eps:
        raise ValueError("Invalid ImageOrientationPatient: zero slice normal")

    return float(np.dot(origin, normal / norm))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _render_frame_metrics(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """原有 frame-level NCC / scale / residual 图。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
    )

    x = np.arange(len(rows))
    valid = np.asarray([bool(row["qc_valid"]) for row in rows])

    for axis, field, title in zip(
        axes,
        ("qc_ncc", "qc_intensity_scale", "qc_residual"),
        ("NCC", "global scale", "corrected residual"),
    ):
        values = np.asarray([float(row[field]) for row in rows])
        axis.plot(x, values, linewidth=1)
        axis.scatter(
            x[~valid],
            values[~valid],
            s=18,
        )
        axis.set_ylabel(title)

    axes[-1].set_xlabel("selected-frame table index")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_slice_metrics(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """新增 slice-location QC 图，重点观察 s16/s17 是否明显掉到 ~0.5 scale。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    if not rows:
        return

    # 每个 view 单独按 patient-world position 排序后顺序拼接。
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["view"]),
            float(row["position_mm"]),
        ),
    )

    x = np.arange(len(ordered))
    labels = [
        f"{row['view']}/{row['slice_id']}"
        for row in ordered
    ]
    valid = np.asarray(
        [bool(row["slice_qc_valid"]) for row in ordered],
        dtype=bool,
    )

    metrics = (
        ("local_scale", "local scale"),
        ("local_ncc", "local NCC"),
        ("local_residual_mean", "corrected residual mean"),
        ("local_residual_p95", "corrected residual p95"),
    )

    figure, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
    )

    for axis, (field, ylabel) in zip(axes, metrics):
        values = np.asarray(
            [
                float(row[field])
                if row[field] not in ("", None)
                else np.nan
                for row in ordered
            ],
            dtype=float,
        )
        axis.plot(x, values, linewidth=1)
        finite_bad = (~valid) & np.isfinite(values)
        axis.scatter(
            x[finite_bad],
            values[finite_bad],
            s=28,
        )
        axis.set_ylabel(ylabel)

    # 144 层时全部标签太密，只显示少量 ticks；
    # 定向 s13-s20 时会自动显示大多数。
    if len(labels) <= 24:
        ticks = x
    else:
        step = max(1, len(labels) // 16)
        ticks = x[::step]

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(
        [labels[int(i)] for i in ticks],
        rotation=60,
        ha="right",
    )
    axes[-1].set_xlabel("slice location")

    figure.savefig(path, dpi=160)
    plt.close(figure)


# -----------------------------------------------------------------------------
# Robust stats
# -----------------------------------------------------------------------------

def _median_mad(
    values: np.ndarray,
    floor: float,
) -> tuple[float, float]:
    """返回 median 与 1.4826 * MAD，并设置 spread floor。"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, float(floor)

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center))) * 1.4826
    return center, max(mad, float(floor))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run frame-level and slice-location acquisition corruption QC "
            "with optional manual exclusions."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to dicom_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="QC output directory",
    )

    parser.add_argument(
        "--slice-key",
        action="append",
        default=None,
        help=(
            "Optional diagnostic subset, repeatable. "
            "Example: --slice-key SAX/s16"
        ),
    )

    parser.add_argument(
        "--manual-exclude",
        action="append",
        default=None,
        help=(
            "Manual exclusion selector, repeatable. Supports SAX/s16, s16, "
            "folder name, file name/path, directory path, or glob."
        ),
    )
    parser.add_argument(
        "--manual-exclude-file",
        type=Path,
        default=None,
        help="Text file with one manual exclusion selector per line",
    )

    # frame-level
    parser.add_argument("--ncc-mad-threshold", type=float, default=6.0)
    parser.add_argument("--scale-mad-threshold", type=float, default=6.0)
    parser.add_argument("--residual-mad-threshold", type=float, default=6.0)
    parser.add_argument("--mad-floor", type=float, default=1e-6)

    # slice-level
    parser.add_argument("--slice-neighbor-radius", type=int, default=3)
    parser.add_argument("--slice-min-neighbors", type=int, default=3)
    parser.add_argument(
        "--slice-foreground-percentile",
        type=float,
        default=70.0,
    )
    parser.add_argument("--slice-scale-low", type=float, default=0.70)
    parser.add_argument("--slice-scale-high", type=float, default=1.40)
    parser.add_argument(
        "--slice-scale-mad-threshold",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--slice-scale-min-log-excursion",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--slice-ncc-mad-threshold",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--slice-residual-mad-threshold",
        type=float,
        default=6.0,
    )

    args = parser.parse_args()

    slice_keys = set(args.slice_key) if args.slice_key else None

    table, summary, frame_plot = run_acquisition_qc(
        args.manifest,
        args.output_dir,
        slice_keys=slice_keys,
        ncc_mad_threshold=args.ncc_mad_threshold,
        scale_mad_threshold=args.scale_mad_threshold,
        residual_mad_threshold=args.residual_mad_threshold,
        mad_floor=args.mad_floor,
        manual_exclude=args.manual_exclude,
        manual_exclude_file=args.manual_exclude_file,
        slice_neighbor_radius=args.slice_neighbor_radius,
        slice_min_neighbors=args.slice_min_neighbors,
        slice_foreground_percentile=args.slice_foreground_percentile,
        slice_scale_low=args.slice_scale_low,
        slice_scale_high=args.slice_scale_high,
        slice_scale_mad_threshold=args.slice_scale_mad_threshold,
        slice_scale_min_log_excursion=args.slice_scale_min_log_excursion,
        slice_ncc_mad_threshold=args.slice_ncc_mad_threshold,
        slice_residual_mad_threshold=args.slice_residual_mad_threshold,
    )

    print(
        json.dumps(
            {
                "acquisition_qc_csv": str(table),
                "summary_json": str(summary),
                "frame_qc_plot": str(frame_plot),
                "slice_location_qc_csv": str(
                    args.output_dir / "slice_location_qc.csv"
                ),
                "slice_location_qc_plot": str(
                    args.output_dir / "slice_location_qc.png"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
