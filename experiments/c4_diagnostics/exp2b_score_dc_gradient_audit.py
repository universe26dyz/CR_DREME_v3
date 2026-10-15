#!/usr/bin/env python3
"""Exp2-B: frozen-checkpoint score-DC loss/gradient competition audit."""
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import torch
import yaml

from cardioresp4d.losses.frequency_loss import dreme_cardiac_leakage_in_resp
from cardioresp4d.losses.motion_loss import dreme_mbc_normalization, dreme_zero_mean_scores
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.trainer import effective_loss_weights
from experiments.c4_diagnostics.exp2_checkpoint_common import (
    assert_parameters_unchanged,
    deterministic_indices,
    parameter_snapshot,
    prepare_frozen_motion_audit,
)
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior


EPSILON = 1e-12


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _geometry_kwargs(observation) -> dict[str, torch.Tensor]:
    image = observation.image.unsqueeze(0)
    return {
        "center_mm": observation.center_mm.to(image)[None],
        "row_direction": observation.row_direction.to(image)[None],
        "column_direction": observation.column_direction.to(image)[None],
        "normal": observation.normal.to(image)[None],
        "pixel_spacing_mm": observation.pixel_spacing_mm.to(image)[None],
        "slice_thickness_mm": image.new_tensor([[observation.slice_thickness_mm]]),
    }


def gradient_statistics(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> dict[str, float]:
    """Gradient scale for one already-weighted scalar loss, without mutation."""
    count = sum(parameter.numel() for parameter in parameters)
    if not parameters or not loss.requires_grad:
        return {"l2_norm": 0.0, "rms_per_parameter": 0.0, "max_abs": 0.0, "nonzero_finite_parameters": 0.0}
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    vectors = [torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1) for parameter, gradient in zip(parameters, gradients)]
    vector = torch.cat(vectors) if vectors else loss.new_zeros(0)
    finite = torch.isfinite(vector)
    finite_values = vector[finite]
    return {
        "l2_norm": float(finite_values.norm()) if finite_values.numel() else 0.0,
        "rms_per_parameter": float(finite_values.square().mean().sqrt()) if finite_values.numel() else 0.0,
        "max_abs": float(finite_values.abs().max()) if finite_values.numel() else 0.0,
        "nonzero_finite_parameters": float(((finite_values != 0).sum()).item()),
    }


def _gradient_vectors(loss: torch.Tensor, groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for values in groups.values():
        for parameter in values:
            if id(parameter) not in seen:
                parameters.append(parameter); seen.add(id(parameter))
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True) if loss.requires_grad else (None,) * len(parameters)
    by_id = {id(parameter): torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1) for parameter, gradient in zip(parameters, gradients)}
    return {name: torch.cat([by_id[id(parameter)] for parameter in values]) for name, values in groups.items()}


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = first.norm() * second.norm()
    return float((first * second).sum() / denominator) if float(denominator) > EPSILON else float("nan")


def _stats_from_vector(vector: torch.Tensor) -> dict[str, float]:
    finite = vector[torch.isfinite(vector)]
    return {"l2_norm": float(finite.norm()) if finite.numel() else 0.0, "rms_per_parameter": float(finite.square().mean().sqrt()) if finite.numel() else 0.0, "max_abs": float(finite.abs().max()) if finite.numel() else 0.0, "nonzero_finite_parameters": float((finite != 0).sum())}


def _gradient_groups(model: SourceFirstDynamicModel) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "film_encoder.image": list(model.film_encoder.image.parameters()),
        "film_encoder.geometry_mlp": list(model.film_encoder.geometry_mlp.parameters()),
        "film_encoder.head": list(model.film_encoder.head.parameters()),
        "film_encoder.resp": list(model.film_encoder.resp.parameters()),
        "film_encoder": list(model.film_encoder.parameters()),
        "respiratory_mbc": list(model.respiratory_mbc.parameters()),
    }


@contextmanager
def _diagnostic_psf_rng(seed: int):
    """Replay production PSF ``torch.randn`` draws without changing production code."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _loss_terms(model: SourceFirstDynamicModel, sequence: list, *, data_frame_count: int, pixels_per_frame: int, prior, weights: dict[str, float], data_psf_seed: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Matched deterministic support for every scalar loss in this audit."""
    encoded = [model.film_encoder(observation.image.unsqueeze(0), **_geometry_kwargs(observation)) for observation in sequence]
    scores = torch.cat([value["resp_scores"] for value in encoded])
    card_scores = torch.cat([value["card_scores"] for value in encoded])
    timestamps = scores.new_tensor([observation.timestamp_s for observation in sequence], dtype=torch.float64)
    selected = deterministic_indices(len(sequence), data_frame_count)
    data_losses = []
    with _diagnostic_psf_rng(data_psf_seed):
        for index in selected:
            observation = sequence[index]
            height, width = observation.image.shape[-2:]
            linear = torch.linspace(0, height * width - 1, min(pixels_per_frame, height * width), device=observation.image.device).round().long()
            pixel_uv = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)
            target = observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]
            data_losses.append((model.predict(observation, pixel_uv, "stage3b")["predicted_intensity"] - target).square().mean())
    grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, model.respiratory_smoothness_shape)
    mbc_norm = dreme_mbc_normalization(model.respiratory_mbc.raw_active(grid, active_levels=3))
    local_prior = prior.for_location(sequence[0].view, sequence[0].slice_id)
    resp_only_zero_mean = dreme_zero_mean_scores(scores)
    stage3b_zero_mean = dreme_zero_mean_scores(torch.cat((scores, card_scores), dim=1))
    card_leak = dreme_cardiac_leakage_in_resp(scores, timestamps, local_prior.cardiac_baseline_pairs)
    return {
        "data": torch.stack(data_losses).mean(),
        "resp_only_zero_mean_score": resp_only_zero_mean,
        "weighted_resp_only_zero_mean_score": weights["zero_mean_score"] * resp_only_zero_mean,
        "stage3b_zero_mean_score": stage3b_zero_mean,
        "weighted_stage3b_zero_mean_score": weights["zero_mean_score"] * stage3b_zero_mean,
        "cardiac_leakage_in_resp": card_leak,
        "weighted_cardiac_leakage_in_resp": weights["cardiac_leakage_in_resp"] * card_leak,
        "mbc_normalization": mbc_norm,
        "weighted_mbc_normalization": weights["mbc_normalization"] * mbc_norm,
    }, scores


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _ratio(numerator: float, denominator: float) -> float:
    return float("nan") if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= EPSILON else numerator / denominator


def run(args: argparse.Namespace) -> dict[str, object]:
    _prepare_output(args.output_dir, args.overwrite)
    audit = prepare_frozen_motion_audit(args)
    model, before = audit.model, parameter_snapshot(audit.model)
    weights = effective_loss_weights(audit.config["training"].get("loss_weights"))
    prior = build_synthetic_frequency_prior(audit.motion.respiratory_frequency_hz, audit.motion.cardiac_frequency_hz, {(item.view, item.slice_id) for item in audit.observations})
    groups = _gradient_groups(model)
    loss_rows: list[dict[str, object]] = []; gradient_rows: list[dict[str, object]] = []; ratio_rows: list[dict[str, object]] = []; score_rows: list[dict[str, object]] = []
    selected_frame_ids: dict[str, list[int]] = {}
    selected_timestamps_s: dict[str, list[float]] = {}
    data_psf_seeds: dict[str, int] = {}
    for location_index, (location, sequence) in enumerate(sorted(audit.groups.items())):
        view = sequence[0].view
        data_psf_seeds[location] = int(args.data_psf_seed_base + location_index)
        selected_frame_ids[location] = deterministic_indices(len(sequence), args.data_frames_per_location)
        selected_timestamps_s[location] = [float(sequence[index].timestamp_s) for index in selected_frame_ids[location]]
        terms, scores = _loss_terms(model, sequence, data_frame_count=args.data_frames_per_location, pixels_per_frame=args.pixels_per_frame, prior=prior, weights=weights, data_psf_seed=data_psf_seeds[location])
        loss_rows.append({"location": location, "view": view, "frame_count": len(sequence), "data_objective": "stage3b_joint", "stage3b_data_loss": float(terms["data"].detach()), **{f"loss_{name}": float(value.detach()) for name, value in terms.items()}})
        dynamic = scores - scores.mean(dim=0, keepdim=True)
        for level in range(3):
            for axis in range(3):
                score_rows.append({"location": location, "view": view, "level": level, "axis": axis, "temporal_mean": float(scores[:, level, axis].mean().detach()), "temporal_rms": float(dynamic[:, level, axis].square().mean().sqrt().detach()), "location_mean_score_norm": float(scores.mean(dim=0).norm().detach()), "overall_temporal_rms": float(dynamic.square().mean().sqrt().detach()), "static_to_temporal_ratio": _ratio(float(scores.mean(dim=0).norm().detach()), float(dynamic.square().mean().sqrt().detach())), "resp_only_zero_mean_score_loss": float(terms["resp_only_zero_mean_score"].detach()), "weighted_resp_only_zero_mean_score_loss": float(terms["weighted_resp_only_zero_mean_score"].detach()), "stage3b_zero_mean_score_loss": float(terms["stage3b_zero_mean_score"].detach()), "weighted_stage3b_zero_mean_score_loss": float(terms["weighted_stage3b_zero_mean_score"].detach())})
        vectors: dict[str, dict[str, torch.Tensor]] = {}
        for name in ("data", "resp_only_zero_mean_score", "weighted_resp_only_zero_mean_score", "stage3b_zero_mean_score", "weighted_stage3b_zero_mean_score", "cardiac_leakage_in_resp", "weighted_cardiac_leakage_in_resp", "mbc_normalization", "weighted_mbc_normalization"):
            fresh_terms, _ = _loss_terms(model, sequence, data_frame_count=args.data_frames_per_location, pixels_per_frame=args.pixels_per_frame, prior=prior, weights=weights, data_psf_seed=data_psf_seeds[location])
            vectors[name] = _gradient_vectors(fresh_terms[name], groups)
            for module, vector in vectors[name].items():
                gradient_rows.append({"location": location, "view": view, "data_objective": "stage3b_joint", "loss_term": name, "module": module, **_stats_from_vector(vector)})
        for module in groups:
            data_norm = _stats_from_vector(vectors["data"][module])["l2_norm"]
            stage3b_zero_norm = _stats_from_vector(vectors["weighted_stage3b_zero_mean_score"][module])["l2_norm"]
            resp_zero_norm = _stats_from_vector(vectors["weighted_resp_only_zero_mean_score"][module])["l2_norm"]
            leak_norm = _stats_from_vector(vectors["weighted_cardiac_leakage_in_resp"][module])["l2_norm"]
            ratio_rows.append({"scope": "location", "location": location, "view": view, "data_objective": "stage3b_joint", "module": module, "weighted_stage3b_zero_mean_grad_norm_over_data_grad_norm": _ratio(stage3b_zero_norm, data_norm), "weighted_resp_only_zero_mean_grad_norm_over_data_grad_norm": _ratio(resp_zero_norm, data_norm), "weighted_card_leak_grad_norm_over_data_grad_norm": _ratio(leak_norm, data_norm), "weighted_stage3b_zero_mean_loss_over_data_loss": _ratio(float(terms["weighted_stage3b_zero_mean_score"].detach()), float(terms["data"].detach())), "weighted_resp_only_zero_mean_loss_over_data_loss": _ratio(float(terms["weighted_resp_only_zero_mean_score"].detach()), float(terms["data"].detach())), "data_vs_weighted_stage3b_zero_mean_cosine": gradient_cosine(vectors["data"][module], vectors["weighted_stage3b_zero_mean_score"][module]), "data_vs_weighted_resp_only_zero_mean_cosine": gradient_cosine(vectors["data"][module], vectors["weighted_resp_only_zero_mean_score"][module]), "data_vs_weighted_card_leakage_cosine": gradient_cosine(vectors["data"][module], vectors["weighted_cardiac_leakage_in_resp"][module]), "weighted_stage3b_zero_mean_vs_weighted_card_leakage_cosine": gradient_cosine(vectors["weighted_stage3b_zero_mean_score"][module], vectors["weighted_cardiac_leakage_in_resp"][module])})
    for module in groups:
        selected = [row for row in ratio_rows if row["scope"] == "location" and row["module"] == module]
        aggregate = {key: float(torch.tensor([float(row[key]) for row in selected if math.isfinite(float(row[key]))]).mean()) if any(math.isfinite(float(row[key])) for row in selected) else float("nan") for key in ("weighted_stage3b_zero_mean_grad_norm_over_data_grad_norm", "weighted_resp_only_zero_mean_grad_norm_over_data_grad_norm", "weighted_card_leak_grad_norm_over_data_grad_norm", "weighted_stage3b_zero_mean_loss_over_data_loss", "weighted_resp_only_zero_mean_loss_over_data_loss", "data_vs_weighted_stage3b_zero_mean_cosine", "data_vs_weighted_resp_only_zero_mean_cosine", "data_vs_weighted_card_leakage_cosine", "weighted_stage3b_zero_mean_vs_weighted_card_leakage_cosine")}
        ratio_rows.append({"scope": "overall", "location": "", "view": "overall", "data_objective": "stage3b_joint", "module": module, **aggregate})
    assert_parameters_unchanged(model, before)
    _write_rows(args.output_dir / "loss_terms_per_location.csv", loss_rows)
    _write_rows(args.output_dir / "gradient_norms_per_location.csv", gradient_rows)
    _write_rows(args.output_dir / "gradient_ratios_by_module.csv", ratio_rows)
    _write_rows(args.output_dir / "score_dc_per_location.csv", score_rows)
    summary = {"experiment": "exp2b_score_dc_gradient_audit", "checkpoint": str(args.checkpoint), "source_checkpoint_git_sha": audit.payload["summary"].get("provenance", {}).get("git_sha"), "diagnostic_git_sha": _git_sha(), "source_config": str(args.source_config), "experiment_config": str(args.experiment_config), "manifest": str(args.manifest), "qc_table": str(args.qc_table), "canonical_domain": str(args.canonical_domain), "source_recovery_arrays": str(audit.recovery_arrays), "timestamps_validated": audit.timestamps_validated, "n_locations": len(audit.groups), "n_frames": len(audit.observations), "formal_loss_weights": weights, "primary_comparison": "weighted_stage3b_zero_mean_score versus stage3b data", "deterministic_policy": {"data_frame_indices_by_location": selected_frame_ids, "data_timestamps_s_by_location": selected_timestamps_s, "data_frames_per_location": args.data_frames_per_location, "pixels_per_frame": args.pixels_per_frame, "pixel_sampling": "evenly spaced row-major fixed pixels; same support for every loss term", "psf_rng": "torch.random.fork_rng with manual reset before every data-loss graph reconstruction", "data_psf_seed_base": args.data_psf_seed_base, "data_psf_seed_by_location": data_psf_seeds}, "optimizer_created": False, "parameters_unchanged": True}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-B frozen score-DC loss/gradient competition audit")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--data-frames-per-location", type=int, default=8); parser.add_argument("--pixels-per-frame", type=int, default=128); parser.add_argument("--data-psf-seed-base", type=int, default=20261015)
    args = parser.parse_args(); summary = run(args)
    print(f"Exp2-B: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
