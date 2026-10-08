#!/usr/bin/env python3
"""Read-only paired audit of canonical, PSF, respiratory, and cardiac rendering stages."""
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
from cardioresp4d.diagnostics.change5c import choose_representative_indices, stable_diagnostic_seed
from cardioresp4d.diagnostics.foundation import cardiac_intersection, foundation_predictions, image_metrics, temporal_metrics, temporal_std_map
from cardioresp4d.diagnostics.stage_visual import finite_array_stats, temporal_maps_and_metrics
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import checkpoint_compatible_dynamic_frame_count, is_hard_invalid_reason
from cardioresp4d.training.source_first_config import validate_source_first_config
from train_source_first import observations_from_manifest


def _location(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError("--location must be VIEW/SLICE_ID")
    view, slice_id = value.split("/", 1)
    if not view or not slice_id:
        raise argparse.ArgumentTypeError("--location must be VIEW/SLICE_ID")
    return view, slice_id


def _plot_comparison(path: Path, frames: list[dict], *, view: str, slice_id: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = ("acquired", "canonical_direct", "canonical_psf", "resp_only", "resp_plus_card", "cardiac_effect")
    all_values = np.concatenate([frame[name].reshape(-1) for frame in frames for name in names[:-1]])
    window = tuple(np.quantile(all_values, (.01, .99)))
    figure, axes = plt.subplots(len(names), len(frames), figsize=(3 * len(frames), 2.6 * len(names)), squeeze=False)
    for column, frame in enumerate(frames):
        for row, name in enumerate(names):
            values = frame[name]
            axis = axes[row, column]; axis.imshow(values, cmap="gray", vmin=0 if name == "cardiac_effect" else window[0], vmax=None if name == "cardiac_effect" else window[1]); axis.axis("off")
            if column == 0: axis.set_ylabel(name.replace("_", " "))
            if row == 0: axis.set_title(f"t={frame['timestamp_s']:.3f}s")
    figure.suptitle(f"{view}/{slice_id}: fixed display window")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _plot_temporal_std(path: Path, stacks: dict[str, torch.Tensor], *, view: str, slice_id: str, individual_scale: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = ("acquired", "canonical_direct", "resp_only", "resp_plus_card")
    maps = {name: temporal_std_map(stacks[name]).numpy() for name in names}
    vmax = max(float(values.max()) for values in maps.values())
    figure, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2))
    for axis, name in zip(axes, names):
        values = maps[name]; axis.imshow(values, cmap="magma", vmin=0., vmax=float(values.max()) if individual_scale else vmax); axis.set_title(f"{name.replace('_', ' ')}\nmean={values.mean():.4g} p95={np.quantile(values,.95):.4g} max={values.max():.4g}" if individual_scale else name.replace("_", " ")); axis.axis("off")
    figure.suptitle(f"{view}/{slice_id}: temporal std, {'individual' if individual_scale else 'shared'} scale")
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _flatten_frame_rows(label: str, view: str, slice_id: str, frame: int, timestamp_s: float, metrics: dict) -> list[dict]:
    rows = []
    for stage, regions in metrics.items():
        for region, values in regions.items():
            rows.append({"kind": "frame", "checkpoint": label, "view": view, "slice_id": slice_id, "frame": frame, "timestamp_s": timestamp_s, "stage": stage, "region": region, **values})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--location", action="append", type=_location, default=[]); parser.add_argument("--view"); parser.add_argument("--slice-id")
    parser.add_argument("--frames", type=int, default=8); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--slice-chunk-size", type=int, default=1024); parser.add_argument("--device", default="cpu"); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames <= 1 or args.slice_chunk_size <= 0: parser.error("--frames must be >=2 and --slice-chunk-size must be positive")
    if bool(args.view) != bool(args.slice_id): parser.error("--view and --slice-id must be supplied together")
    locations = list(args.location) + ([(args.view, args.slice_id)] if args.view else [])
    if not locations: parser.error("supply --location VIEW/SLICE_ID or --view with --slice-id")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8")); observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); state = checkpoint.get("training_state", {}); checkpoint_stage = str(state.get("current_stage", "stage3a"))
    if checkpoint_stage not in ("stage2a", "stage2b", "stage2c", "stage3a", "stage3b", "stage3c"): parser.error(f"unsupported checkpoint stage {checkpoint_stage}")
    model = build_source_first_model(config, domain, n_dynamic_frames=checkpoint_compatible_dynamic_frame_count(checkpoint, observations), device=device, canonical_encoding_backend=detect_checkpoint_encoding_backend(checkpoint["model"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval(); model.canonical.inr.train()
    joint_stage = checkpoint_stage if checkpoint_stage.startswith("stage3") else "stage2c"
    label = args.checkpoint.parent.name; all_locations, csv_rows = [], []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for view, slice_id in locations:
            valid = sorted([item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason) and item.view == view and item.slice_id == slice_id], key=lambda item: item.timestamp_s)
            if len(valid) < 2: parser.error(f"{view}/{slice_id} needs at least two QC-valid non-hard-invalid frames")
            chosen = [valid[index] for index in choose_representative_indices(len(valid), min(args.frames, len(valid)))]
            shared_psf_seed = stable_diagnostic_seed(args.seed, view, slice_id, 0, purpose="foundation_temporal_psf")
            frame_payloads, frame_metrics, mask = [], [], None
            for frame, observation in enumerate(chosen):
                prediction = foundation_predictions(model, observation, joint_stage=joint_stage, seed=args.seed, psf_seed=shared_psf_seed, slice_chunk_size=args.slice_chunk_size)
                acquired = observation.image[0].detach().cpu(); prediction = {name: value.detach().cpu() for name, value in prediction.items()}
                mask = cardiac_intersection(model, observation).detach().cpu()
                metrics = {"acquired": image_metrics(acquired, acquired, mask), **{name: image_metrics(acquired, value, mask) for name, value in prediction.items()}}
                metrics["canonical_direct_to_psf"] = image_metrics(prediction["canonical_direct"], prediction["canonical_psf"], mask)
                metrics["psf_to_resp"] = image_metrics(prediction["canonical_psf"], prediction["resp_only"], mask)
                metrics["resp_to_joint"] = image_metrics(prediction["resp_only"], prediction["resp_plus_card"], mask)
                frame_metrics.append(metrics); csv_rows.extend(_flatten_frame_rows(label, view, slice_id, frame, observation.timestamp_s, metrics))
                frame_payloads.append({"timestamp_s": float(observation.timestamp_s), "dynamic_frame_id": int(observation.dynamic_frame_id), "acquired": acquired.numpy(), **{name: value.numpy() for name, value in prediction.items()}, "cardiac_effect": (prediction["resp_plus_card"] - prediction["resp_only"]).abs().numpy()})
            stacks = {name: torch.stack([torch.from_numpy(frame[name]) for frame in frame_payloads]) for name in ("acquired", "canonical_direct", "canonical_psf", "resp_only", "resp_plus_card")}
            assert mask is not None
            temporal = {name: temporal_metrics(stacks["acquired"], values, mask) for name, values in stacks.items()}
            temporal_arrays = {}
            for name, values in stacks.items():
                maps, _ = temporal_maps_and_metrics(stacks["acquired"], values, mask)
                temporal_arrays[f"{name}_temporal_std"] = maps["predicted_temporal_std"].numpy()
                temporal_arrays[f"{name}_consecutive_delta"] = maps["predicted_delta"].numpy()
            effect = torch.stack([torch.from_numpy(frame["cardiac_effect"]) for frame in frame_payloads])
            temporal["cardiac_effect"] = {"whole_fov_mean_temporal_amplitude": float(effect.float().std(dim=0, unbiased=False).mean()), "cardiac_intersection_mean_temporal_amplitude": float(effect[:, mask].float().std(dim=0, unbiased=False).mean()) if mask.any() else None}
            safe = f"{view}_{slice_id}"
            np.savez_compressed(
                args.output_dir / f"{safe}_foundation_arrays.npz",
                timestamp_s=np.asarray([frame["timestamp_s"] for frame in frame_payloads]),
                dynamic_frame_id=np.asarray([frame["dynamic_frame_id"] for frame in frame_payloads]),
                **{name: stacks[name].numpy() for name in stacks},
                cardiac_effect=effect.numpy(),
                abs_acquired_minus_canonical_direct=(stacks["acquired"] - stacks["canonical_direct"]).abs().numpy(),
                abs_acquired_minus_canonical_psf=(stacks["acquired"] - stacks["canonical_psf"]).abs().numpy(),
                abs_acquired_minus_resp_only=(stacks["acquired"] - stacks["resp_only"]).abs().numpy(),
                abs_acquired_minus_resp_plus_card=(stacks["acquired"] - stacks["resp_plus_card"]).abs().numpy(),
                cardiac_intersection=mask.numpy(),
            )
            np.savez_compressed(args.output_dir / f"{safe}_temporal_std_arrays.npz", cardiac_intersection=mask.numpy(), **{name: values.numpy() for name, values in stacks.items()}, **temporal_arrays)
            (args.output_dir / f"{safe}_temporal_std_stats.json").write_text(json.dumps({"arrays": {name: finite_array_stats(values) for name, values in stacks.items()}, "temporal": temporal}, indent=2) + "\n", encoding="utf-8")
            _plot_comparison(args.output_dir / f"{safe}_foundation_comparison.png", frame_payloads, view=view, slice_id=slice_id)
            _plot_temporal_std(args.output_dir / f"{safe}_temporal_std.png", stacks, view=view, slice_id=slice_id)
            _plot_temporal_std(args.output_dir / f"{safe}_temporal_std_shared_scale.png", stacks, view=view, slice_id=slice_id)
            _plot_temporal_std(args.output_dir / f"{safe}_temporal_std_individual_scale.png", stacks, view=view, slice_id=slice_id, individual_scale=True)
            all_locations.append({"view": view, "slice_id": slice_id, "selected_frames": [{"timestamp_s": frame["timestamp_s"], "dynamic_frame_id": frame["dynamic_frame_id"]} for frame in frame_payloads], "cardiac_intersection_pixels": int(mask.sum()), "per_frame": frame_metrics, "temporal": temporal})
    warning = "Checkpoint is the documented short source-first foundation (Stage2c global_step=400, stage_step=100); this audit reports it without drawing a convergence conclusion." if checkpoint_stage == "stage2c" and int(state.get("global_step", -1)) == 400 else None
    payload = {"status": "read_only_no_optimizer_step", "checkpoint": str(args.checkpoint), "checkpoint_stage": checkpoint_stage, "global_step": state.get("global_step"), "stage_step": state.get("stage_step"), "joint_render_stage": joint_stage, "joint_stage_note": "Stage2c checkpoint has no trained cardiac branch; Resp+Card is reported as Resp-only." if joint_stage == "stage2c" else "Resp+Card uses the checkpoint's Stage3 semantics.", "foundation_warning": warning, "seed": args.seed, "slice_chunk_size": args.slice_chunk_size, "psf_pairing": "Each fixed view/slice temporal comparison uses one shared stable PSF realization; every PSF-bearing branch resets that same seed with chunk iteration inside one RNG context.", "locations": all_locations}
    (args.output_dir / "foundation_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in csv_rows for key in row})
    with (args.output_dir / "foundation_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({"status": payload["status"], "locations": len(all_locations), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
