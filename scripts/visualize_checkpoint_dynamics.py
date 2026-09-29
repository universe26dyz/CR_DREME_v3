#!/usr/bin/env python3
"""Export read-only observation-conditioned implied 3D dynamics, never global cine."""
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
from cardioresp4d.diagnostics.change5c import diagnostic_rng, stable_diagnostic_seed
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import checkpoint_compatible_dynamic_frame_count, is_hard_invalid_reason
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.stage_contract import stage_contract
from cardioresp4d.visualization.dynamics import chunked_canonical_query, jacobian_determinant, pullback_displacement, world_grid
from train_source_first import observations_from_manifest


class _Zero(torch.nn.Module):
    def forward(self, points): return torch.zeros_like(points)


def _motion_for(model, observation, stage: str):
    contract = stage_contract(stage)
    if not contract.enable_motion: return None, None, {}
    image = observation.image.unsqueeze(0); dtype, device = image.dtype, image.device
    scores = model.film_encoder(image, center_mm=observation.center_mm.to(device=device, dtype=dtype)[None], row_direction=observation.row_direction.to(device=device, dtype=dtype)[None], column_direction=observation.column_direction.to(device=device, dtype=dtype)[None], normal=observation.normal.to(device=device, dtype=dtype)[None], pixel_spacing_mm=observation.pixel_spacing_mm.to(device=device, dtype=dtype)[None], slice_thickness_mm=torch.tensor([[observation.slice_thickness_mm]], device=device, dtype=dtype))
    resp = ScoreWeightedMBCField(model.respiratory_mbc, scores["resp_scores"][:, :contract.active_respiratory_levels])
    card = ScoreWeightedMBCField(model.cardiac_mbc, scores["card_scores"]) if contract.enable_cardiac else _Zero()
    return SequentialPullbackMotion(resp, card), SequentialPullbackMotion(resp, _Zero()), scores


def _slice_prediction(model, observation, stage: str, *, seed: int, chunk: int) -> torch.Tensor:
    height, width = observation.image.shape[-2:]; linear = torch.arange(height * width, device=observation.image.device)
    pixels = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), -1).to(observation.image.dtype)
    outputs = []
    psf_seed = stable_diagnostic_seed(seed, observation.view, observation.slice_id, observation.dynamic_frame_id, purpose="psf")
    with diagnostic_rng(psf_seed, observation.image.device):
        for start in range(0, len(pixels), chunk): outputs.append(model.predict(observation, pixels[start:start + chunk], stage)["predicted_intensity"])
    return torch.cat(outputs).reshape(height, width)


def _save_png(path: Path, images: list[np.ndarray], titles: list[str], *, intensity_window: tuple[float, float] | None = None, residual_vmax: float | None = None, effect_vmax: float | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(2, 3, figsize=(12, 7)); axes = axes.ravel()
    for index, (axis, image, title) in enumerate(zip(axes, images, titles)):
        limits = intensity_window if index < 3 else (0., effect_vmax if index == 5 else residual_vmax) if (effect_vmax if index == 5 else residual_vmax) is not None else None
        axis.imshow(image, cmap="gray", vmin=None if limits is None else limits[0], vmax=None if limits is None else limits[1]); axis.set_title(title); axis.axis("off")
    for axis in axes[len(images):]: axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--view", required=True); parser.add_argument("--slice-id", required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--frames", type=int, default=12); parser.add_argument("--grid-shape", type=int, nargs=3, default=(64, 64, 64)); parser.add_argument("--chunk-size", type=int, default=65536); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--intensity-vmin", type=float); parser.add_argument("--intensity-vmax", type=float); parser.add_argument("--residual-vmax", type=float); parser.add_argument("--effect-vmax", type=float)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8")); observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    valid = sorted([item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason) and item.view == args.view and item.slice_id == args.slice_id], key=lambda item: item.timestamp_s)
    if not valid: parser.error("no QC-valid non-hard-invalid frames at requested location")
    chosen = [valid[index] for index in torch.linspace(0, len(valid) - 1, min(args.frames, len(valid))).round().long().tolist()]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); stage = str(checkpoint.get("training_state", {}).get("current_stage", "stage3a"))
    if stage not in ("stage2a", "stage2b", "stage2c", "stage3a", "stage3b", "stage3c"): parser.error(f"unsupported checkpoint stage {stage}")
    n_dynamic_frames = checkpoint_compatible_dynamic_frame_count(checkpoint, observations)
    model = build_source_first_model(config, domain, n_dynamic_frames=n_dynamic_frames, device=device, canonical_encoding_backend=detect_checkpoint_encoding_backend(checkpoint["model"])).to(device); model.load_state_dict(checkpoint["model"]); model.eval(); model.canonical.inr.train()
    args.output_dir.mkdir(parents=True, exist_ok=True); rows = []
    with torch.no_grad():
        for index, observation in enumerate(chosen):
            acquired = observation.image[0].detach().cpu(); resp = _slice_prediction(model, observation, "stage2c", seed=args.seed, chunk=args.chunk_size); joint = _slice_prediction(model, observation, stage, seed=args.seed, chunk=args.chunk_size); psf_seed = stable_diagnostic_seed(args.seed, observation.view, observation.slice_id, observation.dynamic_frame_id, purpose="psf")
            np.savez_compressed(args.output_dir / f"reprojection_{index:03d}.npz", acquired=acquired.numpy(), resp_only=resp.cpu().numpy(), resp_plus_card=joint.cpu().numpy(), timestamp_s=observation.timestamp_s, dynamic_frame_id=observation.dynamic_frame_id, psf_seed=psf_seed)
            _save_png(args.output_dir / f"reprojection_{index:03d}.png", [acquired.numpy(), resp.cpu().numpy(), joint.cpu().numpy(), (acquired - resp.cpu()).abs().numpy(), (acquired - joint.cpu()).abs().numpy(), (joint - resp.cpu()).abs().numpy()], ["Acquired 2D", "Resp-only", "Resp+Card", "|Acquired - Resp|", "|Acquired - Resp+Card|", "|Resp+Card - Resp|"], intensity_window=None if args.intensity_vmin is None or args.intensity_vmax is None else (args.intensity_vmin, args.intensity_vmax), residual_vmax=args.residual_vmax, effect_vmax=args.effect_vmax)
            rows.append({"frame": index, "timestamp_s": observation.timestamp_s, "dynamic_frame_id": int(observation.dynamic_frame_id), "psf_seed": psf_seed, "resp_only_mse": float((acquired - resp.cpu()).square().mean()), "resp_plus_card_mse": float((acquired - joint.cpu()).square().mean())})
        grid, spacing = world_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, tuple(args.grid_shape)); canonical_reference, _ = chunked_canonical_query(model.canonical, None, grid, chunk_size=args.chunk_size); dynamic, resp_dynamic, dvf, jacobians = [], [], [], []
        for index, observation in enumerate(chosen):
            motion, resp_motion, scores = _motion_for(model, observation, stage); volume, reference = chunked_canonical_query(model.canonical, motion, grid, chunk_size=args.chunk_size); resp_volume, _ = chunked_canonical_query(model.canonical, resp_motion, grid, chunk_size=args.chunk_size); total = pullback_displacement(grid, reference)
            motion_data = motion(grid.reshape(1, -1, 3)) if motion is not None else {"cardiac_dvf_mm": torch.zeros_like(grid.reshape(1, -1, 3)), "respiratory_dvf_mm": torch.zeros_like(grid.reshape(1, -1, 3)), "reference_points_mm": grid.reshape(1, -1, 3)}
            determinant = jacobian_determinant(reference, spacing); dynamic.append(volume.cpu()); jacobians.append(determinant.cpu()); dvf.append(total.cpu())
            resp_dynamic.append(resp_volume.cpu()); np.savez_compressed(args.output_dir / f"dvf_{index:03d}.npz", observation_world_mm=grid.cpu().numpy(), reference_world_mm=reference.cpu().numpy(), cardiac_center_world_mm=((model.cardiac_lower_world_mm + model.cardiac_upper_world_mm) / 2).cpu().numpy(), cardiac_dvf_mm=motion_data["cardiac_dvf_mm"].reshape(*args.grid_shape, 3).cpu().numpy(), respiratory_dvf_mm=motion_data["respiratory_dvf_mm"].reshape(*args.grid_shape, 3).cpu().numpy(), total_dvf_mm=total.cpu().numpy(), jacobian_observation_to_reference=determinant.cpu().numpy())
        joint_dynamic = torch.stack(dynamic); resp_dynamic_tensor = torch.stack(resp_dynamic)
        np.savez_compressed(args.output_dir / "conditioned_dynamic_4d.npz", canonical_reference=canonical_reference.cpu().numpy(), resp_only_implied_4d=resp_dynamic_tensor.numpy(), conditioned_dynamic_4d=joint_dynamic.numpy(), cardiac_effect_4d=(joint_dynamic - resp_dynamic_tensor).numpy(), total_dvf_mm=torch.stack(dvf).numpy(), jacobian_observation_to_reference=torch.stack(jacobians).numpy(), world_grid_mm=grid.cpu().numpy(), spacing_mm=spacing.cpu().numpy())
    with (args.output_dir / "reprojection_metrics.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "README.txt").write_text("Read-only observation-conditioned implied 3D dynamics. Resp-only and resp+card predictions use the identical isolated PSF RNG realization derived from global seed + view + slice_id + dynamic_frame_id; experiment label is excluded. Frames from different locations are not a shared physiological phase. DVFs and Jacobians are observation-to-reference pullback fields in mm; positive Jacobian values do not establish diffeomorphism. PCA sign/amplitude are arbitrary; training uses local corr² subspace agreement.\n", encoding="utf-8")
    print(json.dumps({"status": "read_only_no_training", "location": f"{args.view}/{args.slice_id}", "frames": len(chosen), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
