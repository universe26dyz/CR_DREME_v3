#!/usr/bin/env python3
"""Read-only full-location cardiac reprojection contribution audit."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.diagnostics.change5c import aggregate_ablation, aggregate_ablation_by_view, choose_representative_indices, diagnostic_rng, paired_reconstruction_seeds
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import checkpoint_compatible_dynamic_frame_count, is_hard_invalid_reason
from cardioresp4d.training.source_first_config import validate_source_first_config
from train_source_first import observations_from_manifest


def _pixels(model, observation, maximum: int) -> torch.Tensor:
    height, width = observation.image.shape[-2:]; linear = torch.arange(height * width, device=observation.image.device)
    uv = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), -1).to(observation.image.dtype)
    world = model._pixel_world(observation, uv)
    inside = ((world >= model.cardiac_lower_world_mm.to(world)) & (world <= model.cardiac_upper_world_mm.to(world))).all(-1)
    chosen = linear[inside]
    if not chosen.numel(): return torch.empty((0, 2), device=uv.device, dtype=uv.dtype)
    chosen = chosen[torch.tensor(choose_representative_indices(chosen.numel(), maximum), device=chosen.device)]
    return torch.stack((chosen.remainder(width), torch.div(chosen, width, rounding_mode="floor")), -1).to(observation.image.dtype)


def _row(model, observation, stage: str, frame: int, pixels: torch.Tensor, *, diagnostic_seed: int) -> dict:
    target = observation.image[0, pixels[:, 1].long(), pixels[:, 0].long()]
    resp_seed, joint_seed = paired_reconstruction_seeds(diagnostic_seed, "label_not_used", observation.view, observation.slice_id, observation.dynamic_frame_id)
    with diagnostic_rng(resp_seed, observation.image.device):
        resp = model.predict(observation, pixels, "stage2c")["predicted_intensity"]
    with diagnostic_rng(joint_seed, observation.image.device):
        joint = model.predict(observation, pixels, stage)["predicted_intensity"]
    resp_mse, joint_mse = float((resp - target).square().mean()), float((joint - target).square().mean())
    effect = (joint - resp).abs()
    return {"view": observation.view, "slice_id": observation.slice_id, "frame": frame, "timestamp_s": float(observation.timestamp_s), "dynamic_frame_id": int(observation.dynamic_frame_id), "psf_seed": resp_seed, "n_pixels": int(pixels.shape[0]), "resp_only_mse": resp_mse, "resp_plus_card_mse": joint_mse, "absolute_mse_improvement": resp_mse - joint_mse, "relative_mse_improvement_percent": 100. * (resp_mse - joint_mse) / max(resp_mse, torch.finfo(torch.float32).eps), "resp_only_mae": float((resp - target).abs().mean()), "resp_plus_card_mae": float((joint - target).abs().mean()), "mean_abs_joint_minus_resp": float(effect.mean()), "p95_abs_joint_minus_resp": float(torch.quantile(effect, .95))}


def _location_row(rows: list[dict]) -> dict:
    totals = {name: sum(float(row[name]) * int(row["n_pixels"]) for row in rows) / sum(int(row["n_pixels"]) for row in rows) for name in ("resp_only_mse", "resp_plus_card_mse", "resp_only_mae", "resp_plus_card_mae", "mean_abs_joint_minus_resp", "p95_abs_joint_minus_resp")}
    return {"view": rows[0]["view"], "slice_id": rows[0]["slice_id"], "status": "evaluated", "n_frames": len(rows), "n_pixels": sum(int(row["n_pixels"]) for row in rows), **totals, "absolute_mse_improvement": totals["resp_only_mse"] - totals["resp_plus_card_mse"], "relative_mse_improvement_percent": 100. * (totals["resp_only_mse"] - totals["resp_plus_card_mse"]) / max(totals["resp_only_mse"], torch.finfo(torch.float32).eps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames-per-location", type=int, default=5); parser.add_argument("--max-cardiac-pixels", type=int, default=512); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--device", default="cpu"); parser.add_argument("--output-json", type=Path, required=True); parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8")); observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); stage = str(checkpoint.get("training_state", {}).get("current_stage", "stage3a"))
    if stage != "stage3a": parser.error(f"Change5C requires Stage3a checkpoint, got {stage}")
    n_dynamic_frames = checkpoint_compatible_dynamic_frame_count(checkpoint, observations)
    model = build_source_first_model(config, domain, n_dynamic_frames=n_dynamic_frames, device=device, canonical_encoding_backend=detect_checkpoint_encoding_backend(checkpoint["model"])).to(device); model.load_state_dict(checkpoint["model"]); model.eval(); model.canonical.inr.train()
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for observation in observations:
        grouped[(observation.view, observation.slice_id)].append(observation)
    frame_rows, location_rows = [], []
    with torch.no_grad():
        for (view, slice_id), items in sorted(grouped.items()):
            valid = sorted([item for item in items if item.qc_valid and not is_hard_invalid_reason(item.qc_reason)], key=lambda item: item.timestamp_s)
            if not valid:
                location_rows.append({"view": view, "slice_id": slice_id, "status": "skipped", "skip_reason": "no_qc_valid_frames_hard_invalid_excluded"}); continue
            selected = [valid[index] for index in choose_representative_indices(len(valid), args.frames_per_location)]
            rows = []
            for frame, observation in enumerate(selected):
                pixels = _pixels(model, observation, args.max_cardiac_pixels)
                if not pixels.numel(): continue
                rows.append(_row(model, observation, stage, frame, pixels, diagnostic_seed=args.seed))
            if not rows:
                location_rows.append({"view": view, "slice_id": slice_id, "status": "skipped", "skip_reason": "cardiac_box_does_not_intersect_acquired_pixels", "selected_frames": [{"dynamic_frame_id": int(item.dynamic_frame_id), "timestamp_s": float(item.timestamp_s)} for item in selected]}); continue
            frame_rows.extend(rows); location_rows.append(_location_row(rows) | {"selected_frames": [{"dynamic_frame_id": int(item.dynamic_frame_id), "timestamp_s": float(item.timestamp_s)} for item in selected]})
    evaluated = [row for row in location_rows if row["status"] == "evaluated"]
    payload = {"status": "read_only_no_training", "checkpoint": str(args.checkpoint), "stage": stage, "selection_policy": {"frames": "deterministic full-span representative frames", "frames_per_location": args.frames_per_location, "pixels": "cardiac-box-intersecting acquired pixels, deterministic full-span subsample", "max_cardiac_pixels": args.max_cardiac_pixels}, "psf_pairing_policy": {"global_diagnostic_seed": args.seed, "seed_identity": "global seed + view + slice_id + dynamic_frame_id", "experiment_label_in_seed": False, "resp_only_and_resp_plus_card_share_seed": True, "rng_isolation": "torch.random.fork_rng CPU/CUDA"}, "records": location_rows, "frame_records": frame_rows, "overall": aggregate_ablation(evaluated), "per_view": aggregate_ablation_by_view(evaluated), "note": "Primary metrics use cardiac-box-intersecting acquired pixels. Positive MSE gain is descriptive, not statistical significance."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = sorted({field for row in location_rows for field in row if not isinstance(row[field], (list, dict))})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{field: row.get(field) for field in fields} for row in location_rows])
    print(json.dumps({"evaluated_locations": len(evaluated), "skipped_locations": len(location_rows) - len(evaluated), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
