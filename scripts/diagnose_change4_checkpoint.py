#!/usr/bin/env python3
"""Read-only v3_change4 checkpoint diagnostic; it never takes an optimizer step."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.losses.frequency_loss import nonuniform_dft_at_frequencies, resolved_band_frequencies
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import is_hard_invalid_reason
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.stage_contract import stage_contract
from train_source_first import observations_from_manifest


def _summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().reshape(-1).float().cpu()
    return {"min": float(flat.min()), "median": float(flat.median()), "mean": float(flat.mean()), "p95": float(torch.quantile(flat, .95)), "max": float(flat.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v3_change4 checkpoint diagnostic")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8"))
    observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    valid = [item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason)]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_source_first_model(config, domain, n_dynamic_frames=len(valid), device=device).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    prior = load_training_frequency_prior(args.frequency_bands, allow_template_fallback=False)
    locations = list(prior.locations.values())
    stage = checkpoint.get("training_state", {}).get("current_stage")
    if stage == "stage3":
        raise ValueError("legacy v3 Stage3 checkpoint is not compatible with v3_change4 diagnostic; use its Stage2c checkpoint")
    contract = stage_contract(str(stage))
    probe = valid[0]
    pixels = torch.tensor([[0., 0.]], device=device)
    with torch.no_grad():
        render = model.predict(probe, pixels, str(stage))
        motion = model.motion_regularizers(render["scores"], str(stage)) if contract.enable_motion else {}
        motion_stats = {key: float(value.detach().cpu()) for key, value in motion.items() if key.endswith(("rms_mm", "max_mm"))}
        ablation = None
        if contract.enable_cardiac:
            resp_only = model.predict(probe, pixels, "stage2c")["predicted_intensity"]
            joint = render["predicted_intensity"]
            target = probe.image[0, 0, 0]
            resp_mse, joint_mse = (resp_only - target).square().mean(), (joint - target).square().mean()
            ablation = {"resp_only_mse": float(resp_mse), "resp_plus_card_mse": float(joint_mse), "relative_mse_improvement_percent": float((resp_mse - joint_mse) / resp_mse.clamp_min(torch.finfo(resp_mse.dtype).eps) * 100.), "mean_absolute_prediction_change": float((joint - resp_only).abs().mean())}
        grouped = [item for item in valid if item.view == probe.view and item.slice_id == probe.slice_id]
        frequency = None
        if len(grouped) >= 3 and contract.enable_motion:
            times = torch.tensor([item.timestamp_s for item in grouped], dtype=torch.float64, device=device)
            encoded = [model.film_encoder(item.image.unsqueeze(0), center_mm=item.center_mm[None], row_direction=item.row_direction[None], column_direction=item.column_direction[None], normal=item.normal[None], pixel_spacing_mm=item.pixel_spacing_mm[None], slice_thickness_mm=torch.tensor([[item.slice_thickness_mm]], device=device)) for item in grouped]
            resp = torch.cat([item["resp_scores"][:, :contract.active_respiratory_levels].mean(dim=(1, 2), keepdim=True) for item in encoded], 0)
            card = torch.cat([item["card_scores"].mean(dim=(1, 2), keepdim=True) for item in encoded], 0)
            local = prior.for_location(probe.view, probe.slice_id)
            resp_freq = resolved_band_frequencies(local.respiratory_bands_hz, times); card_freq = resolved_band_frequencies(local.cardiac_bands_hz, times)
            amplitude = lambda score, frequencies: float(nonuniform_dft_at_frequencies(score, times, frequencies).abs().mean())
            frequency = {"resp_target_amp": amplitude(resp, resp_freq), "resp_wrong_amp": amplitude(resp, card_freq), "card_target_amp": amplitude(card, card_freq), "card_wrong_amp": amplitude(card, resp_freq), "resp_peak_hz": resp_freq[0], "card_peak_hz": card_freq[0], "respiratory_source": local.respiratory_source, "cardiac_source": local.cardiac_source}
        uncertainty = None
        if contract.enable_uncertainty:
            uncertainty = {key: _summary(value) for key, value in render["uncertainty"].items() if key in {"frame_variance", "pixel_scale", "variance"}}
    result = {
        "checkpoint_stage": stage,
        "prior_audit": {"number_of_locations": len(locations), "local_respiratory_count": sum(item.respiratory_source == "phase1_per_location" for item in locations), "resp_global_fallback_count": sum(item.respiratory_source == "phase1_global_fallback" for item in locations), "local_cardiac_count": sum(item.cardiac_source == "phase1_per_location" for item in locations), "card_global_fallback_count": sum(item.cardiac_source == "phase1_global_fallback" for item in locations), "prior": asdict(prior)},
        "motion_statistics": motion_stats, "cardiac_ablation": ablation, "frequency_semantics": frequency, "uncertainty": uncertainty,
        "status": "read_only_no_training",
        "device": str(device),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
