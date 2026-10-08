#!/usr/bin/env python3
"""Read-only C4 stage visual audit: paired reprojections, temporal maps, and DVF decomposition."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.diagnostics.change5c import choose_representative_indices, diagnostic_rng, quiver_subsample_indices, stable_diagnostic_seed
from cardioresp4d.diagnostics.foundation import cardiac_intersection, foundation_predictions, slice_pixels
from cardioresp4d.diagnostics.stage_visual import dvf_static_dynamic_metrics, finite_array_stats, select_stack_locations, temporal_maps_and_metrics
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import checkpoint_compatible_dynamic_frame_count, is_hard_invalid_reason
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.stage_contract import stage_contract
from cardioresp4d.visualization.dynamics import chunked_canonical_query, jacobian_determinant, pullback_displacement, world_grid
from train_source_first import observations_from_manifest


class _Zero(torch.nn.Module):
    def forward(self, points): return torch.zeros_like(points)


def _scores(model, observation):
    image = observation.image.unsqueeze(0); device, dtype = image.device, image.dtype
    return model.film_encoder(image, center_mm=observation.center_mm.to(device=device, dtype=dtype)[None], row_direction=observation.row_direction.to(device=device, dtype=dtype)[None], column_direction=observation.column_direction.to(device=device, dtype=dtype)[None], normal=observation.normal.to(device=device, dtype=dtype)[None], pixel_spacing_mm=observation.pixel_spacing_mm.to(device=device, dtype=dtype)[None], slice_thickness_mm=torch.tensor([[observation.slice_thickness_mm]], device=device, dtype=dtype))


def _motion_for(model, observation, stage: str):
    """Build the checkpoint stage's observation-to-reference pullback motion."""
    contract = stage_contract(stage)
    if not contract.enable_motion:
        return None, None
    encoded = _scores(model, observation)
    resp = ScoreWeightedMBCField(model.respiratory_mbc, encoded["resp_scores"][:, :contract.active_respiratory_levels])
    card = ScoreWeightedMBCField(model.cardiac_mbc, encoded["card_scores"]) if contract.enable_cardiac else _Zero()
    return SequentialPullbackMotion(resp, card), SequentialPullbackMotion(resp, _Zero())


def _render_scores(model, observation, *, resp_scores: torch.Tensor, card_scores: torch.Tensor | None, seed: int, chunk_size: int) -> torch.Tensor:
    pixels = slice_pixels(observation); resolution = torch.tensor([observation.pixel_spacing_mm[1], observation.pixel_spacing_mm[0], observation.slice_thickness_mm], device=pixels.device, dtype=pixels.dtype)
    resp_scores = resp_scores.to(device=pixels.device, dtype=pixels.dtype)
    card_scores = None if card_scores is None else card_scores.to(device=pixels.device, dtype=pixels.dtype)
    resp = ScoreWeightedMBCField(model.respiratory_mbc, resp_scores); card = ScoreWeightedMBCField(model.cardiac_mbc, card_scores) if card_scores is not None else _Zero(); motion = SequentialPullbackMotion(resp, card)
    output = []
    with diagnostic_rng(seed, observation.image.device):
        for start in range(0, len(pixels), chunk_size):
            chunk = pixels[start:start + chunk_size]; points = model._pixel_world(observation, chunk); count = len(chunk)
            output.append(model.psf(model.canonical, points, resolution.expand(count, -1), row_direction=observation.row_direction.to(points).expand(count, -1), column_direction=observation.column_direction.to(points).expand(count, -1), normal=observation.normal.to(points).expand(count, -1), motion=motion)["predicted_intensity"])
    return torch.cat(output).reshape(observation.image.shape[-2:])


def _plot_maps(path: Path, maps: dict[str, np.ndarray], *, shared: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = tuple(maps); vmax = max(float(maps[name].max()) for name in names) if shared else None
    figure, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.1))
    for axis, name in zip(axes, names):
        values = maps[name]; local = vmax if shared else float(values.max())
        axis.imshow(values, cmap="magma", vmin=0., vmax=local); axis.set_title(f"{name}\nmean={values.mean():.4g} p95={np.quantile(values,.95):.4g} max={values.max():.4g}" if not shared else name); axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _plot_reprojection(path: Path, frames: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = ("acquired", "canonical_direct", "resp_only", "resp_plus_card", "abs_acquired_minus_joint", "abs_joint_minus_resp")
    intensity = np.concatenate([frame[name].reshape(-1) for frame in frames for name in names[:4]]); window = tuple(np.quantile(intensity, (.01, .99)))
    figure, axes = plt.subplots(len(names), len(frames), figsize=(2.5 * len(frames), 2.1 * len(names)), squeeze=False)
    for column, frame in enumerate(frames):
        for row, name in enumerate(names):
            axis = axes[row, column]; axis.imshow(frame[name], cmap="gray", vmin=window[0] if row < 4 else 0., vmax=window[1] if row < 4 else None); axis.axis("off")
            if column == 0: axis.set_ylabel(name.replace("_", " "))
            if row == 0: axis.set_title(f"t={frame['timestamp_s']:.3f}")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _plot_volume_orthogonal(path: Path, arrays: dict[str, np.ndarray], *, title: str) -> None:
    """Plot central world-grid planes; arrays are direct canonical INR samples, no PSF."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = tuple(arrays)
    figure, axes = plt.subplots(len(names), 3, figsize=(8, 2.6 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        values = arrays[name]
        planes = (values[values.shape[0] // 2], values[:, values.shape[1] // 2], values[:, :, values.shape[2] // 2])
        for column, (axis, plane) in enumerate(zip(axes[row], planes)):
            axis.imshow(plane.T, cmap="gray", origin="lower")
            axis.set_title(f"{name}: {'xyz'[column]}-normal")
            axis.axis("off")
    figure.suptitle(title); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _plot_dvf_quiver(path: Path, world_grid_mm: np.ndarray, dvf_mm: np.ndarray, jacobian: np.ndarray, *, title: str) -> None:
    """Show an observation-to-reference DVF magnitude/quiver in world-mm coordinates."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    center = dvf_mm.shape[0] // 2
    vectors = dvf_mm[center]
    coordinates = world_grid_mm[center]
    magnitude = np.linalg.norm(vectors, axis=-1)
    extent = (float(coordinates[..., 2].min()), float(coordinates[..., 2].max()), float(coordinates[..., 1].min()), float(coordinates[..., 1].max()))
    y = quiver_subsample_indices(vectors.shape[0], min(vectors.shape[0], 16))
    x = quiver_subsample_indices(vectors.shape[1], min(vectors.shape[1], 16))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    image = axes[0].imshow(magnitude.T, origin="lower", cmap="magma", extent=extent, aspect="auto"); figure.colorbar(image, ax=axes[0], label="pullback magnitude (mm)")
    axes[0].set_xlabel("world z (mm)"); axes[0].set_ylabel("world y (mm)"); axes[0].set_title("DVF magnitude")
    axes[1].quiver(coordinates[yy, xx, 2], coordinates[yy, xx, 1], vectors[yy, xx, 2], vectors[yy, xx, 1], color="cyan", angles="xy", scale_units="xy")
    axes[1].imshow(jacobian[center].T, origin="lower", cmap="coolwarm", alpha=.6, extent=extent, aspect="auto")
    axes[1].set_xlabel("world z (mm)"); axes[1].set_ylabel("world y (mm)"); axes[1].set_title("world-mm quiver; Jacobian background")
    figure.suptitle(title); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _ncc(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left, right = left.float().reshape(-1), right.float().reshape(-1); left, right = left - left.mean(), right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return None if float(denominator) == 0. else float((left @ right / denominator).cpu())


def _checkpoint_specs(run_dir: Path | None, supplied: list[str]) -> list[tuple[str, Path]]:
    result = []
    for item in supplied:
        label, value = item.split("=", 1); result.append((label, Path(value)))
    if run_dir:
        for label, directory in (("s1a", "s1a_500"), ("s1b", "s1b_1800"), ("s2a_joint", "s2a_joint_2050"), ("s2b_joint", "s2b_joint_2300"), ("s2c_joint", "s2c_joint_2550"), ("s3a", "s3a_2600"), ("s3b_full", "s3b_full_6250")):
            result.append((label, run_dir / directory / "source_first_last.pt"))
    return result


def _audit_observation_conditioned_volume(model, stage: str, observation, output: Path, *, grid_shape: tuple[int, int, int], volume_chunk_size: int) -> dict[str, str]:
    """Export direct and conditioned 3-D pullback arrays for one real observation."""
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        grid, spacing = world_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, grid_shape)
        canonical, _ = chunked_canonical_query(model.canonical, None, grid, chunk_size=volume_chunk_size)
        motion, resp_motion = _motion_for(model, observation, stage)
        joint, reference = chunked_canonical_query(model.canonical, motion, grid, chunk_size=volume_chunk_size)
        resp, _ = chunked_canonical_query(model.canonical, resp_motion, grid, chunk_size=volume_chunk_size)
        pullback = pullback_displacement(grid, reference)
        determinant = jacobian_determinant(reference, spacing)
        motion_data = motion(grid.reshape(1, -1, 3)) if motion is not None else {"respiratory_dvf_mm": torch.zeros_like(grid.reshape(1, -1, 3)), "cardiac_dvf_mm": torch.zeros_like(grid.reshape(1, -1, 3))}
        np.savez_compressed(output / "observation_conditioned_3d.npz", observation_world_grid_mm=grid.cpu().numpy(), reference_world_grid_mm=reference.cpu().numpy(), spacing_mm=spacing.cpu().numpy(), canonical_direct_3d=canonical.cpu().numpy(), resp_only_3d=resp.cpu().numpy(), resp_plus_card_3d=joint.cpu().numpy(), cardiac_effect_3d=(joint - resp).cpu().numpy(), total_pullback_dvf_mm=pullback.cpu().numpy(), respiratory_pullback_dvf_mm=motion_data["respiratory_dvf_mm"].reshape(*grid_shape, 3).cpu().numpy(), cardiac_pullback_dvf_mm=motion_data["cardiac_dvf_mm"].reshape(*grid_shape, 3).cpu().numpy(), jacobian_observation_to_reference=determinant.cpu().numpy())
        _plot_volume_orthogonal(output / "canonical_orthogonal.png", {"canonical": canonical.cpu().numpy()}, title="Direct canonical INR query on world-mm grid (no PSF)")
        _plot_volume_orthogonal(output / "conditioned_orthogonal.png", {"resp_only": resp.cpu().numpy(), "resp_plus_card": joint.cpu().numpy(), "cardiac_effect": (joint - resp).cpu().numpy()}, title=f"Observation-conditioned pullback volume: {observation.view}/{observation.slice_id}, t={observation.timestamp_s:.6g}s")
        _plot_dvf_quiver(output / "pullback_dvf_jacobian.png", grid.cpu().numpy(), pullback.cpu().numpy(), determinant.cpu().numpy(), title="Observation-to-reference pullback DVF and det(d phi / d y)")
    contract = stage_contract(stage)
    return {"status": "available", "view": observation.view, "slice_id": observation.slice_id, "timestamp_s": float(observation.timestamp_s), "dynamic_frame_id": int(observation.dynamic_frame_id), "world_coordinate_unit": "mm", "dvf_semantics": "observation_to_reference_pullback", "jacobian_semantics": "det(d phi / d y) on direct world-mm grid", "respiratory_branch": "available" if contract.enable_motion else "unavailable", "cardiac_branch": "available" if contract.enable_cardiac else "unavailable"}


def _audit_all_qc_location_dvf(model, stage: str, observations, output: Path) -> list[dict]:
    """Numerically audit every QC-valid physical location without implying a cross-slice cine."""
    contract = stage_contract(stage)
    grouped: dict[tuple[str, str], list] = {}
    for observation in observations:
        if observation.qc_valid and not is_hard_invalid_reason(observation.qc_reason):
            grouped.setdefault((observation.view, observation.slice_id), []).append(observation)
    rows = []
    with torch.no_grad():
        for (view, slice_id), valid in sorted(grouped.items()):
            valid.sort(key=lambda item: item.timestamp_s)
            if len(valid) < 2 or not contract.enable_motion:
                rows.append({"view": view, "slice_id": slice_id, "stage": stage, "n_valid_frames": len(valid), "respiratory_status": "unavailable", "cardiac_status": "unavailable" if not contract.enable_cardiac else "not_computed", "reason": "motion is not enabled by this checkpoint stage"})
                continue
            resp_dvf, card_dvf, resp_scores, card_scores = [], [], [], []
            for observation in valid:
                encoded = _scores(model, observation)
                resp_score = encoded["resp_scores"][:, :contract.active_respiratory_levels]
                world = model._pixel_world(observation, slice_pixels(observation))
                resp_dvf.append(ScoreWeightedMBCField(model.respiratory_mbc, resp_score)(world[None]).reshape(*observation.image.shape[-2:], 3).cpu())
                resp_scores.append(resp_score.cpu())
                if contract.enable_cardiac:
                    card_score = encoded["card_scores"]
                    card_dvf.append(ScoreWeightedMBCField(model.cardiac_mbc, card_score)(world[None]).reshape(*observation.image.shape[-2:], 3).cpu())
                    card_scores.append(card_score.cpu())
            _, _, resp_metrics = dvf_static_dynamic_metrics(torch.stack(resp_dvf))
            row = {"view": view, "slice_id": slice_id, "stage": stage, "n_valid_frames": len(valid), "respiratory_status": "available", "cardiac_status": "available" if card_dvf else "unavailable", "resp_score_temporal_std": float(torch.cat(resp_scores).std(unbiased=False)), **{f"resp_{name}": value for name, value in resp_metrics.items()}}
            if card_dvf:
                _, _, card_metrics = dvf_static_dynamic_metrics(torch.stack(card_dvf))
                row.update({"card_score_temporal_std": float(torch.cat(card_scores).std(unbiased=False)), **{f"card_{name}": value for name, value in card_metrics.items()}})
            rows.append(row)
    if rows:
        fields = sorted({name for row in rows for name in row})
        with (output / "all_qc_location_dvf_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows


def _audit_checkpoint(model, stage: str, locations: dict[str, list[str]], observations, output: Path, *, seed: int, chunk_size: int, grid_shape: tuple[int, int, int], volume_chunk_size: int) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True); rows = []
    with torch.no_grad():
        for view, slice_ids in locations.items():
            for slice_id in slice_ids:
                valid = sorted([item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason) and item.view == view and item.slice_id == slice_id], key=lambda item: item.timestamp_s)
                if len(valid) < 2: continue
                location = output / f"{view}_{slice_id}"; (location / "reprojection").mkdir(parents=True, exist_ok=True); (location / "temporal_std").mkdir(exist_ok=True); (location / "temporal_delta").mkdir(exist_ok=True); (location / "mean_vs_dynamic_dvf").mkdir(exist_ok=True)
                shared_seed = stable_diagnostic_seed(seed, view, slice_id, 0, purpose="stage_visual_shared_psf")
                frame_data, score_rows, resp_dvf, card_dvf = [], [], [], []
                for observation in valid:
                    contract = stage_contract(stage)
                    prediction = foundation_predictions(model, observation, joint_stage=stage if contract.enable_cardiac else "stage2c", seed=seed, psf_seed=shared_seed, slice_chunk_size=chunk_size)
                    if not contract.enable_motion:
                        prediction["resp_only"] = prediction["canonical_psf"]
                        prediction["resp_plus_card"] = prediction["canonical_psf"]
                    encoded = _scores(model, observation); contract = stage_contract(stage); resp_score = encoded["resp_scores"][:, :contract.active_respiratory_levels] if contract.enable_motion else None; card_score = encoded["card_scores"] if contract.enable_cardiac else None
                    if resp_score is not None:
                        world = model._pixel_world(observation, slice_pixels(observation)).reshape(*observation.image.shape[-2:], 3); resp_field = ScoreWeightedMBCField(model.respiratory_mbc, resp_score); resp_dvf.append(resp_field(world.reshape(1, -1, 3)).reshape(*world.shape).cpu());
                        if card_score is not None: card_dvf.append(ScoreWeightedMBCField(model.cardiac_mbc, card_score)(world.reshape(1, -1, 3)).reshape(*world.shape).cpu())
                    score_rows.append({"resp": None if resp_score is None else resp_score.cpu(), "card": None if card_score is None else card_score.cpu()})
                    acquired = observation.image[0].detach().cpu(); values = {name: value.detach().cpu() for name, value in prediction.items()}; frame_data.append({"timestamp_s": float(observation.timestamp_s), "dynamic_frame_id": int(observation.dynamic_frame_id), "acquired": acquired, **values})
                stacks = {name: torch.stack([frame[name] for frame in frame_data]) for name in ("acquired", "canonical_direct", "resp_only", "resp_plus_card")}; mask = cardiac_intersection(model, valid[0]).cpu()
                temporal = {}; temporal_arrays = {}
                for name, values in stacks.items():
                    maps, metrics = temporal_maps_and_metrics(stacks["acquired"], values, mask); temporal[name] = metrics; temporal_arrays[f"{name}_temporal_std"] = maps["predicted_temporal_std"].numpy(); temporal_arrays[f"{name}_delta"] = maps["predicted_delta"].numpy()
                stats = {name: finite_array_stats(values) for name, values in stacks.items()}
                np.savez_compressed(location / "temporal_std" / "temporal_std_arrays.npz", cardiac_roi=mask.numpy(), **{name: values.numpy() for name, values in stacks.items()}, **temporal_arrays)
                (location / "temporal_std" / "temporal_std_stats.json").write_text(json.dumps({"arrays": stats, "temporal": temporal}, indent=2) + "\n", encoding="utf-8")
                maps = {name: temporal_arrays[f"{name}_temporal_std"] for name in stacks}; _plot_maps(location / "temporal_std" / "temporal_std_shared_scale.png", maps, shared=True); _plot_maps(location / "temporal_std" / "temporal_std_individual_scale.png", maps, shared=False)
                selected = [frame_data[index] for index in choose_representative_indices(len(frame_data), min(8, len(frame_data)))]; montage = [{**{name: frame[name].numpy() for name in ("acquired", "canonical_direct", "resp_only", "resp_plus_card")}, "timestamp_s": frame["timestamp_s"], "abs_acquired_minus_joint": (frame["acquired"] - frame["resp_plus_card"]).abs().numpy(), "abs_joint_minus_resp": (frame["resp_plus_card"] - frame["resp_only"]).abs().numpy()} for frame in selected]; _plot_reprojection(location / "reprojection" / "montage.png", montage)
                demeaned = {}
                if stage_contract(stage).enable_cardiac and all(item["resp"] is not None and item["card"] is not None for item in score_rows):
                    resp_scores = torch.cat([item["resp"] for item in score_rows]); card_scores = torch.cat([item["card"] for item in score_rows]); mean_resp, mean_card = resp_scores.mean(0, keepdim=True), card_scores.mean(0, keepdim=True)
                    variants = {"resp_only_demeaned_resp": [], "joint_demeaned_card": [], "joint_demeaned_both": []}
                    for index, observation in enumerate(valid):
                        variants["resp_only_demeaned_resp"].append(_render_scores(model, observation, resp_scores=resp_scores[index:index + 1] - mean_resp, card_scores=None, seed=shared_seed, chunk_size=chunk_size).cpu())
                        variants["joint_demeaned_card"].append(_render_scores(model, observation, resp_scores=resp_scores[index:index + 1], card_scores=card_scores[index:index + 1] - mean_card, seed=shared_seed, chunk_size=chunk_size).cpu())
                        variants["joint_demeaned_both"].append(_render_scores(model, observation, resp_scores=resp_scores[index:index + 1] - mean_resp, card_scores=card_scores[index:index + 1] - mean_card, seed=shared_seed, chunk_size=chunk_size).cpu())
                    for name, values in variants.items():
                        values = torch.stack(values); variants[name] = values; _, metrics = temporal_maps_and_metrics(stacks["acquired"], values, mask); demeaned[name] = {"mse": float((stacks["acquired"] - values).square().mean()), "ncc": _ncc(stacks["acquired"], values), "mean_temporal_std": metrics["whole_fov_temporal_std"], "temporal_std_recovery_ratio": metrics["temporal_std_recovery_ratio"], "mean_abs_consecutive_change": metrics["mean_abs_consecutive_change"], "delta_correlation": metrics["delta_correlation"], "mean_abs_joint_minus_resp": float((values - stacks["resp_only"]).abs().mean())}; temporal_arrays[name] = values.numpy()
                    np.savez_compressed(location / "temporal_std" / "demeaned_score_arrays.npz", raw_resp_scores=resp_scores.numpy(), raw_card_scores=card_scores.numpy(), mean_resp_scores=mean_resp.numpy(), mean_card_scores=mean_card.numpy(), **{name: values.numpy() for name, values in variants.items()})
                dvf = {}
                for name, values in (("respiratory", resp_dvf), ("cardiac", card_dvf)):
                    if values:
                        mean, dynamic, metrics = dvf_static_dynamic_metrics(torch.stack(values)); np.savez_compressed(location / "mean_vs_dynamic_dvf" / f"{name}_dvf.npz", raw_dvf_mm=torch.stack(values).numpy(), mean_dvf_mm=mean.numpy(), dynamic_dvf_mm=dynamic.numpy()); score_name = "resp" if name == "respiratory" else "card"; score_stack = torch.cat([item[score_name] for item in score_rows if item[score_name] is not None]); metrics.update({"score_temporal_mean": float(score_stack.mean()), "score_temporal_std": float(score_stack.std(unbiased=False))}); dvf[name] = metrics
                    else:
                        reason = "motion is not enabled by this checkpoint stage" if name == "respiratory" else "cardiac motion is not enabled by this checkpoint stage"
                        dvf[name] = {"status": "unavailable", "reason": reason}
                (location / "metrics.json").write_text(json.dumps({"view": view, "slice_id": slice_id, "n_valid_frames": len(valid), "temporal": temporal, "arrays": stats, "dvf": dvf, "demeaned_score_rendering": demeaned if demeaned else {"status": "unavailable", "reason": "cardiac branch is not enabled by this checkpoint stage"}, "psf_seed": shared_seed, "psf_contract": "one shared isolated PSF RNG realization for every valid frame at this fixed location"}, indent=2) + "\n", encoding="utf-8")
                rows.append({"view": view, "slice_id": slice_id, "stage": stage, "n_valid_frames": len(valid), "acquired_temporal_std": temporal["acquired"]["whole_fov_temporal_std"], "joint_temporal_std": temporal["resp_plus_card"]["whole_fov_temporal_std"], "joint_recovery_ratio": temporal["resp_plus_card"]["temporal_std_recovery_ratio"], "resp_static_dynamic_ratio": dvf.get("respiratory", {}).get("static_to_dynamic_RMS_ratio"), "card_static_dynamic_ratio": dvf.get("cardiac", {}).get("static_to_dynamic_RMS_ratio")})
    representative = None
    for view, slice_ids in locations.items():
        for slice_id in slice_ids:
            valid = sorted((row for row in observations if row.qc_valid and not is_hard_invalid_reason(row.qc_reason) and row.view == view and row.slice_id == slice_id), key=lambda row: row.timestamp_s)
            if valid:
                representative = valid[len(valid) // 2]
                break
        if representative is not None:
            break
    if representative is not None:
        volume_info = _audit_observation_conditioned_volume(model, stage, representative, output / "observation_conditioned_3d", grid_shape=grid_shape, volume_chunk_size=volume_chunk_size)
    else:
        volume_info = {"status": "unavailable", "reason": "no selected location has a QC-valid observation"}
    (output / "observation_conditioned_3d.json").write_text(json.dumps(volume_info, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first_change4_paperaligned.yaml"); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH"); parser.add_argument("--run-dir", type=Path); parser.add_argument("--location", action="append", default=[], metavar="VIEW/SLICE_ID"); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--slice-chunk-size", type=int, default=1024); parser.add_argument("--volume-chunk-size", type=int, default=65536); parser.add_argument("--grid-shape", type=int, nargs=3, default=(64, 64, 64))
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    specs = _checkpoint_specs(args.run_dir, args.checkpoint)
    if not specs: parser.error("supply --checkpoint LABEL=PATH or --run-dir")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT); domain = json.loads(args.canonical_domain.read_text(encoding="utf-8")); observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    args.output_dir.mkdir(parents=True, exist_ok=True); selected_path = args.output_dir / "selected_locations.json"; existing = json.loads(selected_path.read_text()).get("locations") if selected_path.is_file() else None; locations = select_stack_locations(observations, existing=existing)
    if args.location:
        locations = {"SAX": [], "2CH": [], "4CH": []}
        for value in args.location:
            if "/" not in value: parser.error("--location must be VIEW/SLICE_ID")
            view, slice_id = value.split("/", 1)
            if view not in locations: parser.error("--location view must be SAX, 2CH, or 4CH")
            locations[view].append(slice_id)
    selected_path.write_text(json.dumps({"locations": locations, "policy": "explicit CLI locations" if args.location else "physical slice-position lower/middle/upper; existing selections retained when available"}, indent=2) + "\n")
    cross, available = [], {}
    for label, checkpoint_path in specs:
        if not checkpoint_path.is_file(): available[label] = {"status": "unavailable", "checkpoint": str(checkpoint_path)}; continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); stage = str(checkpoint.get("training_state", {}).get("current_stage", "stage1")); model = build_source_first_model(config, domain, n_dynamic_frames=checkpoint_compatible_dynamic_frame_count(checkpoint, observations), device=device, canonical_encoding_backend=detect_checkpoint_encoding_backend(checkpoint["model"])).to(device); model.load_state_dict(checkpoint["model"]); model.eval(); model.canonical.inr.train(); checkpoint_output = args.output_dir / label; cross.extend(_audit_checkpoint(model, stage, locations, observations, checkpoint_output, seed=args.seed, chunk_size=args.slice_chunk_size, grid_shape=tuple(args.grid_shape), volume_chunk_size=args.volume_chunk_size)); all_location_rows = _audit_all_qc_location_dvf(model, stage, observations, checkpoint_output); available[label] = {"status": "audited", "checkpoint": str(checkpoint_path), "stage": stage, "all_qc_location_rows": len(all_location_rows)}
    if cross:
        with (args.output_dir / "cross_stage_metrics.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=cross[0].keys()); writer.writeheader(); writer.writerows(cross)
    (args.output_dir / "provenance.json").write_text(json.dumps({"status": "read_only_no_optimizer_step", "seed": args.seed, "slice_chunk_size": args.slice_chunk_size, "volume_chunk_size": args.volume_chunk_size, "grid_shape": args.grid_shape, "checkpoints": available}, indent=2) + "\n")
    (args.output_dir / "cross_stage_report.md").write_text("# Cross-stage visual audit\n\nRead `cross_stage_metrics.csv` and per-location `metrics.json`; unavailable historical checkpoints are recorded in `provenance.json`. No causal conclusion is emitted without the saved numerical metrics.\n", encoding="utf-8")
    print(json.dumps({"status": "read_only_no_optimizer_step", "output": str(args.output_dir), "audited_stages": [name for name, item in available.items() if item["status"] == "audited"]}, indent=2))


if __name__ == "__main__":
    main()
