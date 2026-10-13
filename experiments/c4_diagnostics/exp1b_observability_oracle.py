"""C4 Exp1-B: synthetic observability, direct MBC, and timestamp-oracle tests.

This module intentionally owns all orchestration; production C4 modules remain
read-only dependencies.  Temporal results are formed per fixed view/slice
before their robust aggregate is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.model import SourceFirstDynamicModel

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import (
    _build_observations, _load_domain, fallback_geometry, geometry_from_manifest,
)
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics, dvf_metrics, location_spectral_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.observability import image_sequence_metrics, normalized_cross_correlation
from experiments.c4_diagnostics.synthetic.oracle import DirectMBCFit, OracleScoreGenerator, build_oracle_motion_model
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, pixel_centers, rescale_frame_for_rendering, render_synthetic_observation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)


def _group(observations: list, frames: list[GeometryFrame]) -> dict[str, list[tuple[int, object, GeometryFrame]]]:
    grouped: dict[str, list[tuple[int, object, GeometryFrame]]] = {}
    for index, (observation, frame) in enumerate(zip(observations, frames)):
        grouped.setdefault(f"{observation.view}/{observation.slice_id}", []).append((index, observation, frame))
    return {key: sorted(value, key=lambda item: item[1].timestamp_s) for key, value in grouped.items()}


def _motion_for_branch(motion: GroundTruthMotion, branch: str) -> GroundTruthMotion:
    if branch == "resp":
        return replace(motion, cardiac_amplitude_mm=0.0)
    if branch == "card":
        return replace(motion, respiratory_amplitude_mm=0.0)
    if branch == "zero":
        return replace(motion, respiratory_amplitude_mm=0.0, cardiac_amplitude_mm=0.0)
    return motion


def _render_sequence(sequence: list[tuple[int, object, GeometryFrame]], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, psf_samples: int, seed: int) -> torch.Tensor:
    images = []
    for index, _, frame in sequence:
        rendered = render_synthetic_observation(frame, height=image_shape[0], width=image_shape[1], phantom=phantom, motion=motion, psf_samples=psf_samples, seed=seed + index, dynamic_frame_id=index)
        images.append(rendered.image)
    return torch.stack(images)


def _dynamic_context(sequence: list[tuple[int, object, GeometryFrame]], motion: GroundTruthMotion, image_shape: tuple[int, int]) -> dict[str, float]:
    frame = rescale_frame_for_rendering(sequence[0][2], height=image_shape[0], width=image_shape[1])
    points = pixel_centers(frame, *image_shape)[None]
    timestamps = points.new_tensor([item[1].timestamp_s for item in sequence])
    tiled = points.expand(len(sequence), -1, -1)
    resp = motion.respiratory_dvf_mm(tiled, timestamps)
    card = motion.cardiac_dvf_mm(tiled, timestamps)
    resp_dynamic = resp - resp.mean(0, keepdim=True)
    card_dynamic = card - card.mean(0, keepdim=True)
    inplane = float(frame.pixel_spacing_mm.mean())
    resp_rms, card_rms = float(resp_dynamic.square().mean().sqrt()), float(card_dynamic.square().mean().sqrt())
    return {"rendered_row_spacing_mm": float(frame.pixel_spacing_mm[0]), "rendered_column_spacing_mm": float(frame.pixel_spacing_mm[1]), "inplane_spacing_mm": inplane, "gt_resp_dynamic_dvf_rms_mm": resp_rms, "gt_card_dynamic_dvf_rms_mm": card_rms, "resp_dvf_to_pixel_ratio": resp_rms / max(inplane, 1e-12), "card_dvf_to_pixel_ratio": card_rms / max(inplane, 1e-12)}


def observability_audit(observations: list, frames: list[GeometryFrame], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, psf_samples: int, seed: int, amplitude_scales: list[float]) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    rows: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}
    for scale in amplitude_scales:
        scaled = replace(motion, respiratory_amplitude_mm=motion.respiratory_amplitude_mm * scale, cardiac_amplitude_mm=motion.cardiac_amplitude_mm * scale)
        for location, sequence in _group(observations, frames).items():
            target = _render_sequence(sequence, image_shape=image_shape, phantom=phantom, motion=scaled, psf_samples=psf_samples, seed=seed)
            variants = {"gt_motion": scaled, "zero_motion": _motion_for_branch(scaled, "zero"), "resp_only_gt": _motion_for_branch(scaled, "resp"), "card_only_gt": _motion_for_branch(scaled, "card")}
            values: dict[str, float] = {"amplitude_scale": float(scale), **_dynamic_context(sequence, scaled, image_shape)}
            for name, variant in variants.items():
                prediction = _render_sequence(sequence, image_shape=image_shape, phantom=phantom, motion=variant, psf_samples=psf_samples, seed=seed)
                metrics = image_sequence_metrics(prediction, target)
                values.update({f"{name}_{metric}": value for metric, value in metrics.items()})
                values[f"{name}_ncc"] = normalized_cross_correlation(prediction, target)
                arrays[f"observability_{scale:g}_{location.replace('/', '_')}_{name}"] = prediction.detach().cpu().numpy()
            gt_mse = values["gt_motion_per_frame_mse"]
            zero_mse = values["zero_motion_per_frame_mse"]
            values["zero_vs_gt_mse_gap"] = zero_mse - gt_mse
            values["resp_only_vs_gt_mse_gap"] = values["resp_only_gt_per_frame_mse"] - gt_mse
            values["card_only_vs_gt_mse_gap"] = values["card_only_gt_per_frame_mse"] - gt_mse
            values["zero_motion_observability_fraction"] = (zero_mse - gt_mse) / max(zero_mse, 1e-12)
            rows[f"{location}|scale={scale:g}"] = values
    return rows, arrays


def _source_mbc_shapes(config: dict[str, object], smoke: bool) -> tuple[int, tuple[tuple[int, int, int], ...], tuple[int, int, int], int]:
    if smoke:
        return 4, ((2, 2, 2),) * 3, (2, 2, 2), 1
    model = config["model"]
    return int(model["respiratory_mbc"]["hidden_dim"]), tuple(tuple(int(x) for x in shape) for shape in model["respiratory_mbc"]["logical_control_shapes"]), tuple(int(x) for x in model["cardiac_mbc"]["logical_control_shape"]), int(model["psf"]["n_samples"])


def _direct_fit(domain: dict[str, object], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, config: dict[str, object], *, device: torch.device, steps: int, smoke: bool) -> dict[str, dict[str, float]]:
    hidden, resp_shapes, card_shape, psf_samples = _source_mbc_shapes(config, smoke)
    lr = float(config["training"]["optimizer"]["stage2"]["respiratory_mbc_lr"])
    output: dict[str, dict[str, float]] = {}
    for branch in ("resp", "card"):
        model = build_oracle_motion_model(domain, phantom, device=device, active_respiratory_levels=3, hidden_dim=hidden, respiratory_shapes=resp_shapes, cardiac_shape=card_shape, psf_samples=psf_samples)
        fitter = DirectMBCFit(model, motion, branch=branch, grid_shape=(3, 3, 3) if smoke else (12, 12, 12), lr=lr)
        for _ in range(steps):
            fitter.step()
        output[branch] = fitter.metrics()
    return output


def _uv(observation, count: int, generator: torch.Generator) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    return torch.stack((torch.randint(width, (count,), generator=generator, device=observation.image.device), torch.randint(height, (count,), generator=generator, device=observation.image.device)), dim=-1).float()


def _oracle_target_observations(frames: list[GeometryFrame], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, branch: str, psf_samples: int, seed: int) -> list:
    return _build_observations(frames, image_shape=image_shape, phantom=phantom, motion=_motion_for_branch(motion, branch), psf_samples=psf_samples, seed=seed)


def _oracle_evaluation(model, observations: list, motion: GroundTruthMotion, generator: OracleScoreGenerator, branch: str, domain: dict[str, object], pixel_samples: int) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    grouped = _group(observations, [GeometryFrame(item.view, item.slice_id, item.center_mm, item.row_direction, item.column_direction, item.normal, item.pixel_spacing_mm, item.slice_thickness_mm, item.timestamp_s, item.image.shape[-2], item.image.shape[-1]) for item in observations])
    result: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}
    lower, upper = model.lower_world_mm, model.upper_world_mm
    grid, _ = SourceFirstDynamicModel._physical_grid(lower, upper, (4, 4, 4))
    for location, sequence in grouped.items():
        predicted_images: list[torch.Tensor] = []; target_images: list[torch.Tensor] = []; pred_dvf: list[torch.Tensor] = []; gt_dvf: list[torch.Tensor] = []; ts: list[float] = []
        for _, observation, _ in sequence:
            timestamp = grid.new_tensor([observation.timestamp_s]); scores = generator(timestamp)
            uv = torch.stack(torch.meshgrid(torch.arange(observation.image.shape[-1], device=grid.device), torch.arange(observation.image.shape[-2], device=grid.device), indexing="xy"), dim=-1).reshape(-1, 2).float()
            with torch.random.fork_rng(devices=[grid.device.index or 0] if grid.is_cuda else []):
                torch.manual_seed(1000 + observation.dynamic_frame_id)
                prediction = model.render(observation, scores, branch, uv)
            predicted_images.append(prediction.reshape(1, *observation.image.shape[-2:]))
            target_images.append(observation.image)
            pred_dvf.append(model.motion(scores, branch)(grid)["reference_points_mm"] - grid)
            gt_dvf.append(_motion_for_branch(motion, branch).pullback_reference_mm(grid, timestamp) - grid)
            ts.append(observation.timestamp_s)
        values = {**{f"dvf_{key}": value for key, value in dvf_metrics(torch.cat(pred_dvf), torch.cat(gt_dvf)).items()}, **image_sequence_metrics(torch.stack(predicted_images), torch.stack(target_images))}
        all_scores = generator(torch.as_tensor(ts, device=grid.device))
        for score_name, score_values, frequency in (("resp", all_scores["resp_scores"], motion.respiratory_frequency_hz), ("card", all_scores["card_scores"], motion.cardiac_frequency_hz)):
            score_metrics = location_spectral_metrics(score_values, torch.as_tensor(ts, device=grid.device), [(frequency - .05, frequency + .05)])
            values.update({f"oracle_{score_name}_score_{key}": value for key, value in score_metrics.items()})
        result[location] = values
        stem = location.replace("/", "_"); arrays[f"oracle_{stem}_timestamps_s"] = np.asarray(ts); arrays[f"oracle_{stem}_predicted_images"] = torch.stack(predicted_images).detach().cpu().numpy()
    return result, arrays


def _oracle_run(domain: dict[str, object], frames: list[GeometryFrame], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, config: dict[str, object], branch: str, device: torch.device, steps: int, pixel_samples: int, observations_per_update: int, seed: int, smoke: bool) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    hidden, resp_shapes, card_shape, psf_samples = _source_mbc_shapes(config, smoke)
    observations = _oracle_target_observations(frames, image_shape=image_shape, phantom=phantom, motion=motion, branch=branch, psf_samples=psf_samples, seed=seed)
    model = build_oracle_motion_model(domain, phantom, device=device, active_respiratory_levels=3, hidden_dim=hidden, respiratory_shapes=resp_shapes, cardiac_shape=card_shape, psf_samples=psf_samples)
    train_branch = {"resp": "resp", "card": "card", "combined": "combined"}[branch]
    model.configure_trainable(train_branch)
    lr_key = "stage2" if branch == "resp" else "stage3a"
    lr_name = "respiratory_mbc_lr" if branch == "resp" else "cardiac_mbc_lr"
    optimizer = torch.optim.Adam([item for item in model.parameters() if item.requires_grad], lr=float(config["training"]["optimizer"][lr_key][lr_name]))
    score_generator = OracleScoreGenerator(motion, active_respiratory_levels=3)
    random = torch.Generator(device=device).manual_seed(seed)
    norm_weight = float(config["training"]["loss_weights"]["mbc_normalization"])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True); losses = []
        offset = (step * observations_per_update) % len(observations)
        for position in range(observations_per_update):
            observation = observations[(offset + position) % len(observations)]
            scores = score_generator(torch.as_tensor([observation.timestamp_s], device=device))
            uv = _uv(observation, min(pixel_samples, observation.image.shape[-1] * observation.image.shape[-2]), random)
            prediction = model.render(observation, scores, train_branch, uv)
            target = observation.image[0, uv[:, 1].long(), uv[:, 0].long()]
            losses.append((prediction - target).square().mean())
        loss = torch.stack(losses).mean() + norm_weight * model.normalization(train_branch, grid_shape=(3, 3, 3))
        loss.backward(); optimizer.step()
    return _oracle_evaluation(model, observations, motion, score_generator, train_branch, domain, pixel_samples)


def _write_rows(path: Path, rows: dict[str, dict[str, float]], *, include_scale: bool = False) -> None:
    names = sorted({name for values in rows.values() for name in values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["location", *names]); writer.writeheader()
        writer.writerows({"location": location, **values} for location, values in rows.items())


def _write_aggregate(path: Path, aggregate: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "median", "p10", "p90"]); writer.writeheader()
        writer.writerows({"metric": name, **values} for name, values in aggregate.items())


def _recovery_label(aggregate: dict[str, dict[str, float]]) -> str:
    if not aggregate:
        return "indeterminate"
    value = lambda key: aggregate.get(key, {}).get("median", float("nan"))
    ok = .15 <= value("dvf_dynamic_rms_ratio") <= 4.0 and value("dvf_dynamic_cosine") >= .2 and value("dvf_endpoint_error_mm") <= 10.0
    return "pass" if ok else "fail"


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    _prepare_output(args.output_dir, args.overwrite)
    with args.source_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.experiment_config.open(encoding="utf-8") as handle:
        synthetic = yaml.safe_load(handle)["synthetic"]
    domain = _load_domain(args.canonical_domain)
    motion = GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"]))
    image_shape = tuple(int(value) for value in synthetic["image_shape"])
    if args.smoke:
        image_shape = (8, 8)
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    phantom = AnalyticCanonicalPhantom(domain).to(device)
    psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    observations = _build_observations(frames, image_shape=image_shape, phantom=phantom, motion=motion, psf_samples=psf_samples, seed=args.seed)
    obs_rows: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}; direct: dict[str, dict[str, float]] = {}; oracle_rows: dict[str, dict[str, float]] = {}
    if args.mode in {"observability", "all"}:
        obs_rows, values = observability_audit(observations, frames, image_shape=image_shape, phantom=phantom, motion=motion, psf_samples=psf_samples, seed=args.seed, amplitude_scales=args.amplitude_scales)
        arrays.update(values)
    if args.mode in {"direct_mbc", "all"}:
        direct = _direct_fit(domain, phantom, motion, config, device=device, steps=args.steps, smoke=args.smoke)
    branch_map = {"oracle_resp_only": "resp", "oracle_card_only": "card", "oracle_combined": "combined"}
    if args.mode in branch_map:
        oracle_rows, values = _oracle_run(domain, frames, image_shape=image_shape, phantom=phantom, motion=motion, config=config, branch=branch_map[args.mode], device=device, steps=args.steps, pixel_samples=args.pixel_samples, observations_per_update=args.observations_per_update, seed=args.seed, smoke=args.smoke)
        arrays.update(values)
    obs_aggregate = aggregate_location_metrics(obs_rows) if obs_rows else {}
    oracle_aggregate = aggregate_location_metrics(oracle_rows) if oracle_rows else {}
    _write_rows(args.output_dir / "observability_per_location.csv", obs_rows)
    _write_aggregate(args.output_dir / "observability_aggregate.csv", obs_aggregate)
    with (args.output_dir / "direct_mbc_fit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["branch", "epe_mm", "spatial_cosine", "rms_ratio", "max_error_mm", "mbc_normalization"]); writer.writeheader(); writer.writerows({"branch": key, **value} for key, value in direct.items())
    _write_rows(args.output_dir / "oracle_per_location.csv", oracle_rows)
    _write_aggregate(args.output_dir / "oracle_aggregate.csv", oracle_aggregate)
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **arrays)
    if obs_rows:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(5, 3)); axis.bar(range(len(obs_rows)), [value["zero_vs_gt_mse_gap"] for value in obs_rows.values()]); axis.set_ylabel("zero-vs-GT MSE gap"); axis.set_xlabel("fixed location / amplitude"); figure.tight_layout(); figure.savefig(args.output_dir / "figures" / "observability_mse_gaps.png", dpi=140); plt.close(figure)
    observability = "indeterminate" if not obs_aggregate else ("strong" if obs_aggregate["zero_motion_observability_fraction"]["median"] >= .01 else "weak")
    direct_label = lambda branch: "indeterminate" if branch not in direct else ("pass" if direct[branch]["epe_mm"] <= 2.0 and direct[branch]["spatial_cosine"] >= .2 else "fail")
    recovery = _recovery_label(oracle_aggregate)
    interpretation = "observability not evaluated"
    if observability == "weak": interpretation = "benchmark resolution/amplitude gives weak image-domain motion observability"
    elif observability == "strong" and any(direct_label(branch) == "fail" for branch in direct): interpretation = "production MBC direct representability is the limiting factor"
    elif observability == "strong" and oracle_rows and recovery == "fail": interpretation = "MBC is tested directly; oracle image-domain recovery points to reconstruction-gradient/optimization"
    elif observability == "strong" and oracle_rows and recovery == "pass": interpretation = "oracle recovery supports an encoder/state-inference explanation for original Exp1 failure"
    summary = {"experiment": "exp1b_observability_oracle", "mode": args.mode, "observability": observability, "resp_direct_mbc_fit": direct_label("resp"), "card_direct_mbc_fit": direct_label("card"), "oracle_resp_recovery": recovery if args.mode == "oracle_resp_only" else "indeterminate", "oracle_card_recovery": recovery if args.mode == "oracle_card_only" else "indeterminate", "oracle_combined_recovery": recovery if args.mode == "oracle_combined" else "indeterminate", "mechanistic_interpretation": interpretation, "observability_aggregate": obs_aggregate, "direct_mbc_metrics": direct, "oracle_aggregate": oracle_aggregate, "training": {"steps": args.steps, "pixel_samples": args.pixel_samples, "observations_per_update": args.observations_per_update, "canonical_trainable": False, "encoder_trainable": False}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-B synthetic observability and timestamp-oracle diagnostic")
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path)
    parser.add_argument("--mode", choices=("observability", "direct_mbc", "oracle_resp_only", "oracle_card_only", "oracle_combined", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=20); parser.add_argument("--steps", type=int, default=400); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--observations-per-update", type=int, default=32); parser.add_argument("--amplitude-scales", type=lambda value: [float(item) for item in value.split(",")], default=[1.0]); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.manifest is None) != (args.qc_table is None): parser.error("--manifest and --qc-table must be provided together")
    if args.smoke:
        args.steps = min(args.steps, 2); args.pixel_samples = min(args.pixel_samples, 8); args.observations_per_update = min(args.observations_per_update, 2); args.locations_per_view = min(args.locations_per_view, 1); args.frames_per_location = min(args.frames_per_location, 4)
    summary = run(args)
    print(f"Exp1-B {summary['mode']}: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
