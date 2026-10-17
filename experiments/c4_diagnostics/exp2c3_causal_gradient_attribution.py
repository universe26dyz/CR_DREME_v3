#!/usr/bin/env python3
"""Exp2-C3: frozen causal-gradient attribution for motion-only checkpoints.

This is intentionally an offline CPU diagnostic.  It never creates an
optimizer, calls ``backward()``, or updates a checkpoint parameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch

from cardioresp4d.losses.frequency_loss import (
    cardiac_target_band_concentration,
    dreme_cardiac_leakage_in_resp,
    dreme_respiratory_leakage_in_card,
)
from cardioresp4d.losses.motion_loss import dreme_mbc_normalization, dreme_zero_mean_scores
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.trainer import effective_loss_weights
from experiments.c4_diagnostics.exp2_checkpoint_common import (
    assert_parameters_unchanged,
    deterministic_indices,
    full_image_uv,
    parameter_snapshot,
    prepare_frozen_motion_audit,
)
from experiments.c4_diagnostics.exp2c_posthoc_score_centering import _safe_correlation
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior


EPSILON = 1e-12
AGGREGATES = ("mean", "median", "p10", "p90")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _unique(parameters: list[torch.nn.Parameter], label: str) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    result: list[torch.nn.Parameter] = []
    for parameter in parameters:
        if id(parameter) in seen:
            raise ValueError(f"parameter is duplicated within {label}")
        seen.add(id(parameter))
        result.append(parameter)
    if not result:
        raise ValueError(f"required parameter group {label} is empty")
    return result


def parameter_groups(model: SourceFirstDynamicModel) -> dict[str, list[torch.nn.Parameter]]:
    """Return documented non-overlapping leaves plus explicitly overlapping aggregates."""
    encoder = model.film_encoder
    if model.film_implementation == "full_geometry_single_film":
        leaves = {
            "encoder.image": list(encoder.image.parameters()),
            "encoder.geometry_mlp": list(encoder.geometry_mlp.parameters()),
            "encoder.head": list(encoder.head.parameters()),
            "encoder.resp_head": list(encoder.resp.parameters()),
            "encoder.card_head": list(encoder.card.parameters()),
        }
        backbone_names = ("encoder.image", "encoder.geometry_mlp", "encoder.head")
    elif model.film_implementation == "s2v_style_single_frame":
        leaves = {
            "encoder.blocks": list(encoder.blocks.parameters()),
            "encoder.condition_mlp": list(encoder.condition_mlp.parameters()),
            "encoder.view_embedding": list(encoder.view_embedding.parameters()),
            "encoder.film_heads": list(encoder.film_heads.parameters()),
            "encoder.head": list(encoder.head.parameters()),
            "encoder.resp_head": list(encoder.resp.parameters()),
            "encoder.card_head": list(encoder.card.parameters()),
        }
        backbone_names = ("encoder.blocks", "encoder.condition_mlp", "encoder.view_embedding", "encoder.film_heads", "encoder.head")
    else:
        raise ValueError(f"unsupported encoder implementation {model.film_implementation!r}")
    leaves = {name: _unique(values, name) for name, values in leaves.items()}
    all_leaf = [parameter for values in leaves.values() for parameter in values]
    if len({id(parameter) for parameter in all_leaf}) != len(all_leaf):
        raise ValueError("encoder leaf groups overlap")
    respiratory = {f"respiratory_mbc.level{index}": _unique(list(level.parameters()), f"respiratory_mbc.level{index}") for index, level in enumerate(model.respiratory_mbc.levels)}
    if set(respiratory) != {"respiratory_mbc.level0", "respiratory_mbc.level1", "respiratory_mbc.level2"}:
        raise ValueError("C3 requires exactly three respiratory MBC levels")
    cardiac = _unique(list(model.cardiac_mbc.parameters()), "cardiac_mbc")
    result = {**leaves, **respiratory, "cardiac_mbc": cardiac}
    result["encoder_backbone"] = _unique([parameter for name in backbone_names for parameter in leaves[name]], "encoder_backbone")
    result["encoder_all"] = _unique(all_leaf, "encoder_all")
    result["respiratory_mbc_all"] = _unique([parameter for values in respiratory.values() for parameter in values], "respiratory_mbc_all")
    result["motion_all"] = _unique(result["respiratory_mbc_all"] + cardiac, "motion_all")
    return result


def gradient_vector(loss: torch.Tensor, parameters: list[torch.nn.Parameter], *, retain_graph: bool = False) -> torch.Tensor:
    """A stable flattened autograd vector that never writes ``parameter.grad``."""
    parameters = _unique(parameters, "gradient vector")
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(parameter).reshape(-1) for parameter in parameters])
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    return torch.cat([torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1) for parameter, gradient in zip(parameters, gradients)])


def _gradient_vectors(loss: torch.Tensor, groups: dict[str, list[torch.nn.Parameter]], *, retain_graph: bool = False) -> dict[str, torch.Tensor]:
    all_parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for values in groups.values():
        for parameter in values:
            if id(parameter) not in seen:
                seen.add(id(parameter)); all_parameters.append(parameter)
    if loss.requires_grad:
        gradients = torch.autograd.grad(loss, all_parameters, allow_unused=True, retain_graph=retain_graph)
    else:
        gradients = (None,) * len(all_parameters)
    by_id = {id(parameter): torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1) for parameter, gradient in zip(all_parameters, gradients)}
    return {name: torch.cat([by_id[id(parameter)] for parameter in values]) for name, values in groups.items()}


def vector_statistics(vector: torch.Tensor) -> dict[str, float]:
    finite = vector[torch.isfinite(vector)]
    return {
        "gradient_l2_norm": float(finite.norm()) if finite.numel() else float("nan"),
        "gradient_rms_per_parameter": float(finite.square().mean().sqrt()) if finite.numel() else float("nan"),
        "gradient_max_abs": float(finite.abs().max()) if finite.numel() else float("nan"),
        "finite_nonzero_gradient_elements": int((finite != 0).sum()),
        "parameter_count": int(vector.numel()),
    }


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        return float("nan")
    denominator = first.norm() * second.norm()
    return float((first * second).sum() / denominator) if float(denominator) > EPSILON else float("nan")


def directional_effect(metric_gradient: torch.Tensor, loss_gradient: torch.Tensor, *, current_metric: float, epsilon: float = EPSILON) -> dict[str, float | str]:
    """Metric derivative along the unit *loss-descent* direction ``-g_L``."""
    if not torch.isfinite(metric_gradient).all() or not torch.isfinite(loss_gradient).all():
        return {"gradient_cosine": float("nan"), "directional_effect_per_unit_loss_gradient": float("nan"), "fractional_directional_effect": float("nan"), "status": "nonfinite_gradient"}
    norm = float(loss_gradient.norm())
    if norm <= epsilon:
        return {"gradient_cosine": gradient_cosine(metric_gradient, loss_gradient), "directional_effect_per_unit_loss_gradient": float("nan"), "fractional_directional_effect": float("nan"), "status": "zero_loss_gradient"}
    effect = float(torch.dot(metric_gradient, -loss_gradient / norm))
    return {"gradient_cosine": gradient_cosine(metric_gradient, loss_gradient), "directional_effect_per_unit_loss_gradient": effect, "fractional_directional_effect": effect / max(abs(current_metric), epsilon), "status": "defined"}


def _score_state_tensors(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    dynamic = scores - scores.mean(dim=0, keepdim=True)
    static = scores.mean(dim=0).square().mean().sqrt()
    temporal = dynamic.square().mean().sqrt()
    return {"static_rms": static, "temporal_rms": temporal, "static_to_temporal_rms_ratio": static / temporal.clamp_min(EPSILON)}


def score_state_metrics(scores: torch.Tensor) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in _score_state_tensors(scores).items()}


def _respiratory_field(model: SourceFirstDynamicModel, scores: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return ScoreWeightedMBCField(model.respiratory_mbc, scores)(points)


def _dvf_dynamic_tensors(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    dynamic = predicted - predicted.mean(dim=0, keepdim=True)
    target_dynamic = target - target.mean(dim=0, keepdim=True)
    predicted_rms = dynamic.square().mean().sqrt()
    target_rms = target_dynamic.square().mean().sqrt()
    cosine = (dynamic * target_dynamic).sum() / (dynamic.norm() * target_dynamic.norm()).clamp_min(EPSILON)
    return {
        "dynamic_rms": predicted_rms,
        "gt_dynamic_rms": target_rms,
        "dynamic_rms_ratio": predicted_rms / target_rms.clamp_min(EPSILON),
        "dynamic_cosine": cosine,
        "endpoint_error_mm": (predicted - target).square().sum(-1).sqrt().mean(),
        "dynamic_normalized_epe": (dynamic - target_dynamic).square().sum(-1).sqrt().mean() / target_rms.clamp_min(EPSILON),
    }


def _selected_pixels(observation, pixels_per_frame: int) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    linear = torch.linspace(0, height * width - 1, min(pixels_per_frame, height * width), device=observation.image.device).round().long()
    return torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)


@contextmanager
def _diagnostic_psf_rng(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _zero_from(scores: torch.Tensor) -> torch.Tensor:
    return scores.sum() * 0.


def _loss_terms(
    model: SourceFirstDynamicModel,
    sequence: list,
    *,
    objective_view: str,
    data_frame_count: int,
    pixels_per_frame: int,
    prior,
    weights: dict[str, float],
    data_psf_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Production-equivalent scalar terms on fixed support for one location."""
    if objective_view not in {"stage2c_counterfactual", "stage3b_joint"}:
        raise ValueError("unknown objective view")
    stage = "stage2c" if objective_view == "stage2c_counterfactual" else "stage3b"
    encoded = [model.encode_motion_scores(observation) for observation in sequence]
    resp_scores = torch.cat([value["resp_scores"] for value in encoded])
    card_scores = torch.cat([value["card_scores"] for value in encoded])
    timestamps = torch.tensor([observation.timestamp_s for observation in sequence], dtype=torch.float64, device=resp_scores.device)
    selected = deterministic_indices(len(sequence), data_frame_count)
    data_losses: list[torch.Tensor] = []
    image_losses: list[torch.Tensor] = []
    with _diagnostic_psf_rng(data_psf_seed):
        for index in selected:
            observation = sequence[index]
            pixels = _selected_pixels(observation, pixels_per_frame)
            target = observation.image[0, pixels[:, 1].long(), pixels[:, 0].long()]
            render = model.predict(observation, pixels, stage)
            data_losses.append((render["predicted_intensity"] - target).square().mean())
            image_losses.append(model.canonical.image_regularization(render["intensity_samples"], render["reference_samples_world_mm"], mode=model.image_regularization_mode, delta=model.image_regularization_delta))
    local_prior = prior.for_location(sequence[0].view, sequence[0].slice_id)
    resp_grid, _ = model._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, model.respiratory_smoothness_shape)
    resp_raw = model.respiratory_mbc.raw_active(resp_grid, active_levels=3)
    if objective_view == "stage3b_joint":
        card_grid, _ = model._physical_grid(model.cardiac_lower_world_mm, model.cardiac_upper_world_mm, model.cardiac_smoothness_shape)
        card_raw = model.cardiac_mbc(card_grid)[:, None]
        mbc_normalization = dreme_mbc_normalization((resp_raw, card_raw))
        zero_mean = dreme_zero_mean_scores(torch.cat((resp_scores, card_scores), dim=1))
        resp_leakage_in_card = dreme_respiratory_leakage_in_card(card_scores, timestamps, local_prior.respiratory_bands_hz)
        target_concentration = cardiac_target_band_concentration(card_scores, timestamps, local_prior.cardiac_bands_hz)["loss"]
        motion_regularizers = {
            name: torch.stack(values).mean()
            for name, values in {
                name: [model.motion_regularizers({"resp_scores": resp_scores[index:index + 1], "card_scores": card_scores[index:index + 1]}, "stage3b")[name] for index in selected]
                for name in ("smooth_resp", "smooth_card")
            }.items()
        }
    else:
        mbc_normalization = dreme_mbc_normalization(resp_raw)
        zero_mean = dreme_zero_mean_scores(resp_scores)
        resp_leakage_in_card = _zero_from(resp_scores)
        target_concentration = _zero_from(resp_scores)
        motion_regularizers = {
            "smooth_resp": torch.stack([model.motion_regularizers({"resp_scores": resp_scores[index:index + 1]}, "stage2c")["smooth_resp"] for index in selected]).mean()
        }
    cardiac_leakage = dreme_cardiac_leakage_in_resp(resp_scores, timestamps, local_prior.cardiac_baseline_pairs)
    zero = _zero_from(resp_scores)
    raw = {
        "data_mse": torch.stack(data_losses).mean(),
        "image_regularization": torch.stack(image_losses).mean(),
        "mbc_normalization": mbc_normalization,
        "smooth_resp": motion_regularizers["smooth_resp"],
        "smooth_card": motion_regularizers.get("smooth_card", zero),
        "zero_mean_score": zero_mean,
        "cardiac_leakage_in_resp": cardiac_leakage,
        "respiratory_leakage_in_card": resp_leakage_in_card,
        "cardiac_target_concentration": target_concentration,
        "cardiac_pca_waveform": zero,
    }
    weighted = {f"weighted_{name}": raw[name] * (1. if name == "data_mse" else weights.get({"data_mse": "data_mse", "image_regularization": "image", "mbc_normalization": "mbc_normalization", "smooth_resp": "smooth_resp", "smooth_card": "smooth_card", "zero_mean_score": "zero_mean_score", "cardiac_leakage_in_resp": "cardiac_leakage_in_resp", "respiratory_leakage_in_card": "respiratory_leakage_in_card", "cardiac_target_concentration": "cardiac_target_concentration", "cardiac_pca_waveform": "cardiac_pca_waveform"}[name], 0.)) for name in raw}
    regularizer_names = [name for name in weighted if name != "weighted_data_mse"]
    weighted_regularizer_sum = sum((weighted[name] for name in regularizer_names), _zero_from(resp_scores))
    terms = {**raw, **weighted, "weighted_regularizer_sum": weighted_regularizer_sum, "full_objective": raw["data_mse"] + weighted_regularizer_sum}
    return terms, {"resp_scores": resp_scores, "card_scores": card_scores, "timestamps": timestamps}


def _state_rows(model: SourceFirstDynamicModel, sequence: list, motion, *, dvf_frames: int, grid: torch.Tensor) -> list[dict[str, object]]:
    with torch.no_grad():
        encoded = [model.encode_motion_scores(observation) for observation in sequence]
        resp = torch.cat([value["resp_scores"] for value in encoded])
        card = torch.cat([value["card_scores"] for value in encoded])
        timestamps = resp.new_tensor([item.timestamp_s for item in sequence], dtype=torch.float64)
        rows: list[dict[str, object]] = []
        score_values = {f"resp_score_{name}": value for name, value in score_state_metrics(resp).items()}
        score_values.update({f"card_score_{name}": value for name, value in score_state_metrics(card).items()})
        score_values["resp_score_max_abs_gt_correlation"] = _safe_correlation(resp, motion.respiratory_score(timestamps.to(torch.float32)))
        selected = deterministic_indices(len(sequence), dvf_frames)
        for support, points_by_frame in {
            "full_fov_grid": [grid for _ in selected],
            "pixel_centers": [SourceFirstDynamicModel._pixel_world(sequence[index], full_image_uv(sequence[index]).to(sequence[index].image)) for index in selected],
        }.items():
            frames = resp[selected]
            points = torch.stack(points_by_frame)
            predicted = torch.stack([_respiratory_field(model, score[None], item[None])[0] for score, item in zip(frames, points)])
            target = torch.stack([motion.respiratory_dvf_mm(item[None], item.new_tensor([sequence[index].timestamp_s]))[0] for index, item in zip(selected, points)])
            values = {name: float(value) for name, value in _dvf_dynamic_tensors(predicted, target).items()}
            rows.append({"support": support, **score_values, **{f"resp_dvf_{name}": value for name, value in values.items()}})
    return rows


def _causal_metrics(model: SourceFirstDynamicModel, sequence: list, motion, *, dvf_frames: int, grid: torch.Tensor, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    resp = encoded["resp_scores"]
    selected = deterministic_indices(len(sequence), dvf_frames)
    selected_scores = resp[selected]
    predicted = torch.stack([_respiratory_field(model, score[None], grid[None])[0] for score in selected_scores])
    target = torch.stack([motion.respiratory_dvf_mm(grid[None], grid.new_tensor([sequence[index].timestamp_s]))[0] for index in selected])
    values = _score_state_tensors(resp)
    values.update({f"resp_dvf_{name}": value for name, value in _dvf_dynamic_tensors(predicted, target).items() if name in {"dynamic_rms", "dynamic_cosine"}})
    return {"resp_score_static_rms": values["static_rms"], "resp_score_temporal_rms": values["temporal_rms"], "resp_dvf_dynamic_rms": values["resp_dvf_dynamic_rms"], "resp_dvf_dynamic_cosine": values["resp_dvf_dynamic_cosine"]}


def _assert_finite(label: str, value: float) -> None:
    if not math.isfinite(value):
        raise RuntimeError(f"C3 found non-finite {label}")


def _assert_nonzero(label: str, vector: torch.Tensor) -> None:
    value = float(vector.norm())
    _assert_finite(label, value)
    if value <= EPSILON:
        raise RuntimeError(f"C3 expected nonzero gradient for {label}")


def _assert_zero(label: str, vector: torch.Tensor) -> None:
    value = float(vector.norm())
    _assert_finite(label, value)
    if value > EPSILON:
        raise RuntimeError(f"C3 expected structural zero gradient for {label}: {value}")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]], *, source: str, by: tuple[str, ...]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    numeric = sorted({name for row in rows for name, value in row.items() if isinstance(value, (int, float)) and name not in {"parameter_count", "finite_nonzero_gradient_elements"}})
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(name, "") for name in by), []).append(row)
    for key, selected in grouped.items():
        prefix = {"source": source, **dict(zip(by, key))}
        for name in numeric:
            values = [float(row[name]) for row in selected if name in row and math.isfinite(float(row[name]))]
            if values:
                result.append({**prefix, "metric": name, "n": len(values), **{aggregate: float(getattr(np, aggregate)(values)) if aggregate in {"mean", "median"} else float(np.percentile(values, int(aggregate[1:]))) for aggregate in AGGREGATES}})
    return result


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    _prepare_output(args.output_dir, args.overwrite)
    audit = prepare_frozen_motion_audit(args)
    model, before = audit.model, parameter_snapshot(audit.model)
    groups = parameter_groups(model)
    weights = effective_loss_weights(audit.config["training"].get("loss_weights"))
    prior = build_synthetic_frequency_prior(audit.motion.respiratory_frequency_hz, audit.motion.cardiac_frequency_hz, {(item.view, item.slice_id) for item in audit.observations})
    grid, _ = model._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, tuple(args.dvf_grid_shape))
    grid = grid[0]
    state_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    directional_rows: list[dict[str, object]] = []
    selected_frames: dict[str, list[int]] = {}
    selected_timestamps: dict[str, list[float]] = {}
    psf_seeds: dict[str, int] = {}
    active_terms = ("weighted_data_mse", "weighted_mbc_normalization", "weighted_zero_mean_score", "weighted_cardiac_leakage_in_resp", "weighted_respiratory_leakage_in_card", "weighted_regularizer_sum")
    comparisons = (("weighted_data_mse", "weighted_mbc_normalization"), ("weighted_data_mse", "weighted_zero_mean_score"), ("weighted_data_mse", "weighted_cardiac_leakage_in_resp"), ("weighted_data_mse", "weighted_respiratory_leakage_in_card"), ("weighted_zero_mean_score", "weighted_cardiac_leakage_in_resp"), ("weighted_regularizer_sum", "weighted_data_mse"))
    for location_index, (location, sequence) in enumerate(sorted(audit.groups.items())):
        view = sequence[0].view
        seed = int(args.data_psf_seed_base + location_index)
        psf_seeds[location] = seed
        selected_frames[location] = deterministic_indices(len(sequence), args.data_frames_per_location)
        selected_timestamps[location] = [float(sequence[index].timestamp_s) for index in selected_frames[location]]
        for row in _state_rows(model, sequence, audit.motion, dvf_frames=args.dvf_frames_per_location, grid=grid):
            state_rows.append({"location": location, "view": view, "frame_count": len(sequence), **row})
        for objective_view in ("stage2c_counterfactual", "stage3b_joint"):
            terms, _ = _loss_terms(model, sequence, objective_view=objective_view, data_frame_count=args.data_frames_per_location, pixels_per_frame=args.pixels_per_frame, prior=prior, weights=weights, data_psf_seed=seed)
            values = {name: float(value.detach()) for name, value in terms.items()}
            if objective_view == "stage3b_joint":
                values["stage3b_zero_mean_score"] = values["zero_mean_score"]
                values["weighted_stage3b_zero_mean_score"] = values["weighted_zero_mean_score"]
            else:
                values["stage2c_respiratory_only_zero_mean_score"] = values["zero_mean_score"]
                values["weighted_stage2c_respiratory_only_zero_mean_score"] = values["weighted_zero_mean_score"]
            values.update({f"formal_weight_{name}": float(weight) for name, weight in weights.items()})
            for name, value in values.items():
                _assert_finite(f"{location} {objective_view} {name}", value)
            if abs(values["weighted_image_regularization"]) > EPSILON:
                raise RuntimeError("motion-only analytic canonical has nonzero weighted image regularization")
            loss_rows.append({"location": location, "view": view, "objective_view": objective_view, **values})
            vectors_by_term: dict[str, dict[str, torch.Tensor]] = {}
            for term_name in active_terms:
                fresh_terms, _ = _loss_terms(model, sequence, objective_view=objective_view, data_frame_count=args.data_frames_per_location, pixels_per_frame=args.pixels_per_frame, prior=prior, weights=weights, data_psf_seed=seed)
                vectors_by_term[term_name] = _gradient_vectors(fresh_terms[term_name], groups)
                for group_name, vector in vectors_by_term[term_name].items():
                    statistics = vector_statistics(vector)
                    if not math.isfinite(statistics["gradient_l2_norm"]):
                        raise RuntimeError(f"non-finite gradient norm for {term_name}/{group_name}")
                    gradient_rows.append({"location": location, "view": view, "objective_view": objective_view, "loss_term": term_name, "parameter_group": group_name, **statistics})
            if objective_view == "stage3b_joint":
                _assert_nonzero(f"{location} stage3b data encoder", vectors_by_term["weighted_data_mse"]["encoder_all"])
                _assert_nonzero(f"{location} stage3b data respiratory MBC", vectors_by_term["weighted_data_mse"]["respiratory_mbc_all"])
                _assert_nonzero(f"{location} stage3b data cardiac MBC", vectors_by_term["weighted_data_mse"]["cardiac_mbc"])
                _assert_zero(f"{location} MBC normalization encoder", vectors_by_term["weighted_mbc_normalization"]["encoder_all"])
                _assert_zero(f"{location} zero mean respiratory MBC", vectors_by_term["weighted_zero_mean_score"]["respiratory_mbc_all"])
                _assert_zero(f"{location} card leakage respiratory MBC", vectors_by_term["weighted_cardiac_leakage_in_resp"]["respiratory_mbc_all"])
                _assert_zero(f"{location} respiratory leakage cardiac MBC", vectors_by_term["weighted_respiratory_leakage_in_card"]["cardiac_mbc"])
            for group_name in groups:
                for first, second in comparisons:
                    cosine = gradient_cosine(vectors_by_term[first][group_name], vectors_by_term[second][group_name])
                    cosine_rows.append({"location": location, "view": view, "objective_view": objective_view, "parameter_group": group_name, "first_loss_term": first, "second_loss_term": second, "gradient_cosine": cosine, "status": "defined" if math.isfinite(cosine) else "zero_or_nonfinite_vector"})
            for term_name in active_terms:
                fresh_terms, encoded = _loss_terms(model, sequence, objective_view=objective_view, data_frame_count=args.data_frames_per_location, pixels_per_frame=args.pixels_per_frame, prior=prior, weights=weights, data_psf_seed=seed)
                loss_vectors = _gradient_vectors(fresh_terms[term_name], groups, retain_graph=True)
                metrics = _causal_metrics(model, sequence, audit.motion, dvf_frames=args.dvf_frames_per_location, grid=grid, encoded=encoded)
                for metric_name, metric in metrics.items():
                    metric_vectors = _gradient_vectors(metric, groups, retain_graph=True)
                    current = float(metric.detach())
                    _assert_finite(f"{location} {metric_name}", current)
                    for group_name in groups:
                        directional_rows.append({"location": location, "view": view, "objective_view": objective_view, "loss_term": term_name, "parameter_group": group_name, "metric": metric_name, "metric_current_value": current, **directional_effect(metric_vectors[group_name], loss_vectors[group_name], current_metric=current)})
    assert_parameters_unchanged(model, before)
    _write_rows(args.output_dir / "state_metrics_per_location.csv", state_rows)
    _write_rows(args.output_dir / "loss_terms_per_location.csv", loss_rows)
    _write_rows(args.output_dir / "gradient_norms_per_location.csv", gradient_rows)
    _write_rows(args.output_dir / "gradient_cosines_per_location.csv", cosine_rows)
    _write_rows(args.output_dir / "directional_effects_per_location.csv", directional_rows)
    aggregate = _aggregate(state_rows, source="state", by=("support",)) + _aggregate(loss_rows, source="loss", by=("objective_view",)) + _aggregate(gradient_rows, source="gradient_norm", by=("objective_view", "loss_term", "parameter_group")) + _aggregate(cosine_rows, source="gradient_cosine", by=("objective_view", "first_loss_term", "second_loss_term", "parameter_group")) + _aggregate(directional_rows, source="directional_effect", by=("objective_view", "loss_term", "parameter_group", "metric"))
    _write_rows(args.output_dir / "aggregate.csv", aggregate)
    summary = {
        "experiment": "Exp2-C3 frozen causal gradient attribution",
        "checkpoint": str(args.checkpoint),
        "source_checkpoint_git_sha": audit.payload["summary"].get("provenance", {}).get("git_sha"),
        "diagnostic_git_sha": _git_sha(),
        "source_config": str(args.source_config),
        "encoder_implementation": model.film_implementation,
        "experiment_config": str(args.experiment_config),
        "manifest": str(args.manifest),
        "qc_table": str(args.qc_table),
        "canonical_domain": str(args.canonical_domain),
        "source_recovery_arrays": str(audit.recovery_arrays),
        "timestamps_validated": audit.timestamps_validated,
        "n_locations": len(audit.groups),
        "n_frames": len(audit.observations),
        "formal_loss_weights": weights,
        "stage3b_objective_definition": "data_mse + weighted image/MBC/smoothness/temporal terms on deterministic matched support",
        "stage2c_counterfactual_definition": "final frozen checkpoint evaluated under respiratory-only Stage2c forward and objective semantics; not a saved Stage2c state",
        "deterministic_support_policy": {"data_frames_per_location": args.data_frames_per_location, "pixels_per_frame": args.pixels_per_frame, "dvf_frames_per_location": args.dvf_frames_per_location, "dvf_grid_shape": list(args.dvf_grid_shape), "selected_frame_indices_by_location": selected_frames, "selected_timestamps_by_location": selected_timestamps, "pixel_sampling": "evenly spaced row-major pixels", "psf_rng": "fork_rng reset before every loss graph", "psf_seed_by_location": psf_seeds},
        "parameter_groups": {name: [parameter_name for parameter_name, parameter in model.named_parameters() if any(id(parameter) == id(member) for member in members)] for name, members in groups.items()},
        "optimizer_created": False,
        "parameters_unchanged": True,
        "directional_effects_note": "directional effects are first-order frozen-checkpoint derivatives, not optimizer steps and not retraining.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C3 frozen causal gradient attribution")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--data-frames-per-location", type=int, default=8); parser.add_argument("--pixels-per-frame", type=int, default=128); parser.add_argument("--dvf-frames-per-location", type=int, default=8); parser.add_argument("--dvf-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(8, 8, 8)); parser.add_argument("--data-psf-seed-base", type=int, default=20261017)
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("Exp2-C3 is an offline CPU diagnostic; use --device cpu")
    if len(args.dvf_grid_shape) != 3 or min(args.dvf_grid_shape) < 2:
        parser.error("--dvf-grid-shape must be three comma-separated values >=2")
    if min(args.data_frames_per_location, args.pixels_per_frame, args.dvf_frames_per_location) <= 0:
        parser.error("deterministic support counts must be positive")
    summary = run(args)
    print(f"Exp2-C3: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
