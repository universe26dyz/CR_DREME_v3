"""Read-only visualization adapter for the isolated Exp2-A gauge output."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..contract import ExperimentVisualizationAdapter, STAGES, stage_status
from ..dynamic_audit import (fixed_physical_grid, location_local_rms_metrics, mbc_points_from_world_grid, predict_observation_slice,
                             select_fixed_location_frames, static_dynamic_rms, weighted_dvf_from_precomputed_basis)
from ..geometry_utils import location_token
from ..renderers import fixed_window, render_independent_images
from .exp1_reference import _render_vector_field_planes


class Exp2AReferenceGaugeAdapter(ExperimentVisualizationAdapter):
    name = "exp2a_reference_gauge"

    @property
    def scientific_status(self) -> Mapping[str, str]:
        return {"experiment_scientific_status": "EXP2A_REFERENCE_GAUGE_PENDING_RETROSPECTIVE_AUDIT", "reason": "Exp2-A is isolated and must be interpreted only after the full-cohort Exp1 retrospective gauge audit."}

    def source_artifacts(self, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path) -> dict[str, Path]:
        paths = {"checkpoint_step_200": experiment_root / "checkpoint_step_200.pt", "checkpoint_last": experiment_root / "checkpoint_last.pt", "training_report": experiment_root / "training_report.json", "experiment_inputs": experiment_root / "experiment_inputs.json", "experiment_config": experiment_root / "effective_config.json"}
        return {key: path for key, path in paths.items() if path.exists()}

    def render(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output_dir: Path, device: str, stages: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
        requested = set(STAGES if stages is None else stages)
        unknown = requested - set(STAGES)
        if unknown:
            raise ValueError(f"unknown visualization stage(s): {', '.join(sorted(unknown))}")
        for stage in STAGES:
            (output_dir / stage).mkdir(parents=True, exist_ok=True)
        result = {name: stage_status("skipped", "stage not requested") for name in STAGES}
        if "V8_dynamic_zero_motion" in requested:
            result["V8_dynamic_zero_motion"] = stage_status("partial", "Exp2-A has no canonical-only boundary schedule")
        if "V9_dynamic_boundaries" in requested:
            result["V9_dynamic_boundaries"] = self.render_v9_dynamic_boundaries(experiment_root=experiment_root, manifest=manifest, qc_table=qc_table, canonical_domain=canonical_domain, output=output_dir / "V9_dynamic_boundaries", device=device)
        return result

    @staticmethod
    def _load_step200_context(*, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, device: str):
        from scripts.train_source_first import observations_from_manifest
        from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend, lps_to_ras_world_matrix
        from cardioresp4d.experiments.respiratory_reference_gauge_exp2a import verify_exp2a_input_bindings
        from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import apply_corrected_static_poses, derive_effective_external_domain, load_corrected_pose_map
        from cardioresp4d.training.build_model import build_source_first_model
        from cardioresp4d.visualization.reference_gauge import ReferenceImageProvider

        input_path = experiment_root / "experiment_inputs.json"
        if not input_path.is_file():
            raise FileNotFoundError("Exp2-A experiment_inputs.json is required for visualization")
        bindings = json.loads(input_path.read_text())
        verify_exp2a_input_bindings(bindings)
        for name, path in {"manifest": manifest, "qc_table": qc_table, "canonical_domain": canonical_domain}.items():
            if Path(bindings[name]["path"]).resolve() != Path(path).resolve():
                raise ValueError(f"Exp2-A visualization input path differs from immutable binding: {name}")
        checkpoint_path = experiment_root / "checkpoint_step_200.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Exp2-A checkpoint_step_200.pt is required for V9")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("score_gauge", {}).get("policy") != "raw_minus_same_encoder_reference_mean":
            raise ValueError("Exp2-A checkpoint score-gauge policy is not the fixed reference policy")
        if checkpoint.get("score_gauge", {}).get("reference_manifest_sha256") != bindings["mean_manifest"]["sha256"]:
            raise ValueError("Exp2-A checkpoint/reference manifest hash mismatch")
        config = checkpoint["effective_config"]
        domain = json.loads(Path(canonical_domain).read_text())
        poses = load_corrected_pose_map(Path(bindings["corrected_pose_manifest"]["path"]))
        effective_domain = derive_effective_external_domain(poses, domain["world_min_mm"], domain["world_max_mm"])
        model_domain = {**domain, "world_min_mm": effective_domain["model_float32_effective_domain"]["lower_lps_mm"], "world_max_mm": effective_domain["model_float32_effective_domain"]["upper_lps_mm"]}
        target_device = torch.device(device)
        observations, _, _ = observations_from_manifest(manifest, qc_table, target_device, normalization_mode=config["training"]["normalization"]["mode"], assign_slice_positions=True)
        observations = apply_corrected_static_poses(observations, poses)
        static_payload = torch.load(Path(bindings["static_model"]["path"]), map_location="cpu", weights_only=False)
        candidate = static_payload.get("model", static_payload.get("inr_state_dict"))
        internal_bounds = torch.as_tensor(checkpoint["static_initialization"]["internal_inr_bounds_mm"], dtype=torch.float32)
        backend = checkpoint["static_initialization"].get("encoding_backend") or detect_checkpoint_encoding_backend(dict(candidate))
        model = build_source_first_model(config, model_domain, n_dynamic_frames=max(item.dynamic_frame_id for item in observations) + 1, device=target_device, canonical_encoding_backend=backend, canonical_internal_bounds_world_mm=internal_bounds.to(target_device), canonical_world_to_internal_transform=lps_to_ras_world_matrix(device=target_device)).to(target_device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval(); model.canonical.inr.train()
        provider = ReferenceImageProvider.from_mean_artifacts(Path(bindings["mean_manifest"]["path"]), observations=observations)
        if provider.provenance["mean_manifest_sha256"] != bindings["mean_manifest"]["sha256"] or provider.provenance["mean_stacks"] != {name: item["sha256"] for name, item in bindings["mean_stacks"].items()}:
            raise ValueError("Exp2-A provider mean-stack binding mismatch")
        model.set_reference_score_provider(provider)
        return checkpoint, model, observations, bindings

    def render_v9_dynamic_boundaries(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output: Path, device: str = "cpu", **_: Any) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        checkpoint, model, observations, bindings = self._load_step200_context(experiment_root=experiment_root, manifest=manifest, qc_table=qc_table, canonical_domain=canonical_domain, device=device)
        selected = select_fixed_location_frames(observations, frames_per_location=3)
        all_frames: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for observation in observations:
            if observation.qc_valid:
                all_frames[(observation.view.upper(), observation.slice_id)].append(observation)
        for values in all_frames.values():
            values.sort(key=lambda item: (float(item.timestamp_s), int(item.dynamic_frame_id)))
        grid, _ = fixed_physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (16, 16, 16)); raw_basis = model.respiratory_mbc.raw_active(mbc_points_from_world_grid(grid), active_levels=1).detach()
        mbc_dir, dvf_dir, score_dir, reprojection_dir = (output / name for name in ("mbc", "dvf", "scores", "reprojection"))
        for directory in (mbc_dir, dvf_dir, score_dir, reprojection_dir): directory.mkdir(parents=True, exist_ok=True)
        _render_vector_field_planes(mbc_dir, "L1", raw_basis[0, 0], label="RAW MBC BASIS FIELD")
        np.savez_compressed(mbc_dir / "raw_mbc_basis.npz", world_grid_mm=grid.detach().cpu().numpy(), raw_mbc_basis_mm=raw_basis.cpu().numpy(), label="RAW MBC BASIS FIELD")
        rows = []
        for location, representative_frames in selected.items():
            frames = all_frames[location]; score_sequences = {key: [] for key in ("raw", "reference", "effective")}; dvf_sequences = {key: [] for key in score_sequences}
            with torch.no_grad():
                for frame in frames:
                    raw = model.encode_motion_scores_raw(frame)["resp_scores"][:, :1]; reference = model.encode_reference_scores(frame)["resp_scores"][:, :1]; effective = raw - reference
                    for key, score in (("raw", raw), ("reference", reference), ("effective", effective)):
                        score_sequences[key].append(score[0]); dvf_sequences[key].append(weighted_dvf_from_precomputed_basis(raw_basis, score)[0])
            score_sequences = {key: torch.stack(value) for key, value in score_sequences.items()}; dvf_sequences = {key: torch.stack(value) for key, value in dvf_sequences.items()}; token = location_token(*location); times = [float(frame.timestamp_s) for frame in frames]
            figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            for axis, key, title in zip(axes, ("raw", "reference", "effective"), ("raw score(t)", "reference score(t)", "effective score(t) = raw-reference")):
                values = score_sequences[key].detach().cpu().numpy().reshape(len(frames), -1)
                for index in range(values.shape[1]): axis.plot(times, values[:, index], marker="o", label=f"L1.c{index + 1}")
                axis.set_title(title); axis.grid(alpha=.25); axis.legend(fontsize=7)
            axes[-1].set_xlabel("timestamp (s)"); figure.tight_layout(); figure.savefig(score_dir / f"{location[0]}_{token}.png", dpi=140); plt.close(figure)
            for key in ("raw", "reference", "effective"):
                figure, axis = plt.subplots(figsize=(9, 3.5)); values = score_sequences[key].detach().cpu().numpy().reshape(len(frames), -1)
                for index in range(values.shape[1]): axis.plot(times, values[:, index], marker="o", label=f"L1.c{index + 1}")
                axis.set(xlabel="timestamp (s)", title=f"{key} score(t) — {location[0]}/{location[1]}"); axis.grid(alpha=.25); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(score_dir / f"{location[0]}_{token}_{key}.png", dpi=140); plt.close(figure)
            for key in ("raw", "effective"):
                values = dvf_sequences[key]; _render_vector_field_planes(dvf_dir, f"{location[0]}_{token}_{key}_frame_middle", values[len(values) // 2], label="raw score diagnostic DVF" if key == "raw" else "effective-score training DVF")
            figure, axis = plt.subplots(figsize=(9, 3.5))
            for key, values in (("raw", dvf_sequences["raw"]), ("effective", dvf_sequences["effective"])):
                axis.plot(times, values.square().mean(dim=(-1, -2)).sqrt().detach().cpu(), marker="o", label=f"{key} DVF RMS (mm)"); axis.plot(times, values.square().sum(dim=-1).sqrt().amax(dim=-1).detach().cpu(), marker="x", label=f"{key} DVF max (mm)")
            axis.set(xlabel="timestamp (s)", title=f"raw/effective DVF magnitude vs time — {location[0]}/{location[1]}"); axis.grid(alpha=.25); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(dvf_dir / f"{location[0]}_{token}_vs_time.png", dpi=140); plt.close(figure)
            for frame in representative_frames:
                predicted = predict_observation_slice(model, frame, "stage2a", seed=20260823).detach().cpu().numpy(); observed = frame.image[0].detach().cpu().numpy(); residual = np.abs(observed - predicted); shared_window = fixed_window(np.concatenate((observed.ravel(), predicted.ravel())), robust=True); residual_window = (0., max(float(residual.max()), 1e-6)); render_independent_images(reprojection_dir / f"{location[0]}_{token}_{frame.dynamic_frame_id:05d}.png", [observed, predicted, residual], ["observed", "effective-gauge motion prediction", "absolute residual"], [shared_window, shared_window, residual_window])
            rows.append({"view": location[0], "slice_id": location[1], "frame_count": len(frames), "raw_score": static_dynamic_rms(score_sequences["raw"]), "reference_score": static_dynamic_rms(score_sequences["reference"]), "effective_score": static_dynamic_rms(score_sequences["effective"]), "raw_dvf": location_local_rms_metrics((dvf_sequences["raw"],), kind="dvf"), "effective_dvf": location_local_rms_metrics((dvf_sequences["effective"],), kind="dvf")})
        training_report = json.loads((experiment_root / "training_report.json").read_text()) if (experiment_root / "training_report.json").is_file() else {}
        payload = {"experiment": self.name, "checkpoint": "checkpoint_step_200.pt", "device": device, "step_0": training_report.get("diagnostics", {}).get("step_0", {}), "step_200": rows, "input_bindings": bindings, "step_0_visualization": "metrics_only; no step-0 model checkpoint exists", "score_policy": checkpoint["score_gauge"]["policy"], "cohorts": {"scientific": "all QC-valid locations and frames for training diagnostics", "visualization": "one deterministic physical-middle location per view; full temporal curves; early/middle/late reprojection"}, "shared_grid_shape": [16, 16, 16], "labels": {"raw_dvf": "raw score diagnostic DVF", "effective_dvf": "effective-score training DVF", "mbc": "RAW MBC BASIS FIELD"}}
        (output / "exp2a_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
        return stage_status("rendered", diagnostics_file="exp2a_diagnostics.json", checkpoint="checkpoint_step_200.pt", score_pngs=True, dvf_pngs=True, mbc_pngs=True, reprojection_pngs=True, step_0_policy=payload["step_0_visualization"])
