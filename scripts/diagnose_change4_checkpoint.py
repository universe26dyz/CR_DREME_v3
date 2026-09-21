#!/usr/bin/env python3
"""Read-only v3_change4 checkpoint diagnostic; it never takes an optimizer step."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformPrior
from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.losses.frequency_loss import cardiac_target_band_concentration, nonuniform_dft_at_frequencies, resolved_band_frequencies
from cardioresp4d.losses.motion_loss import cardiac_pca_waveform_subspace_loss
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import is_hard_invalid_reason
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.stage_contract import stage_contract
from cardioresp4d.diagnostics.checkpoint_metrics import summarize_records
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from train_source_first import observations_from_manifest


def _summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().reshape(-1).float().cpu()
    return {"min": float(flat.min()), "median": float(flat.median()), "mean": float(flat.mean()), "p95": float(torch.quantile(flat, .95)), "max": float(flat.max())}


def prepare_read_only_diagnostic_model(model) -> None:
    """Keep diagnostic modules in eval mode but expose NeSVoR's latent API.

    Pinned NeSVoR emits latent ``z`` for ``return_features=True`` only while
    its INR is in train mode.  This narrowly restores that upstream API; the
    diagnostic caller remains inside ``torch.no_grad()`` and never optimizes.
    """
    model.eval()
    model.canonical.inr.train()


def dominant_peak_hz(scores: torch.Tensor, timestamps_s: torch.Tensor, *, minimum_hz: float = .05, maximum_hz: float = 3., samples: int = 512) -> float:
    """Return the actual mean-channel NUDFT-power maximum, not a band edge."""
    ordered = timestamps_s.to(dtype=torch.float64).sort().values
    nyquist = .5 / float((ordered[1:] - ordered[:-1]).median())
    upper = min(float(maximum_hz), nyquist)
    if not upper > minimum_hz:
        raise ValueError("timestamp sampling has no diagnostic frequency range")
    frequencies = torch.linspace(minimum_hz, upper, samples, device=scores.device, dtype=scores.dtype)
    power = nonuniform_dft_at_frequencies(scores, timestamps_s, frequencies).abs().square().flatten(1).mean(1)
    return float(frequencies[power.argmax()].detach().cpu())


def frequency_semantics_record(view: str, slice_id: str, resp: torch.Tensor, card: torch.Tensor, timestamps_s: torch.Tensor, local, *, pca_waveform_prior=None, pca_waveform_ridge: float = 1e-4, pca_waveform_min_frames: int = 8) -> dict[str, object]:
    """Read-only per-location spectral/PCA audit shared by Change4/5A/5B."""
    resp_freq = resolved_band_frequencies(local.respiratory_bands_hz, timestamps_s)
    card_freq = resolved_band_frequencies(local.cardiac_bands_hz, timestamps_s)
    amplitude = lambda score, frequencies: float(nonuniform_dft_at_frequencies(score, timestamps_s, frequencies).abs().mean())
    concentration = cardiac_target_band_concentration(card, timestamps_s, local.cardiac_bands_hz)
    pca = {"cardiac_pca_waveform_r2": None, "cardiac_pca_waveform_loss": None, "cardiac_pca_matched_frames": 0, "selected_cardiac_pc": None}
    if pca_waveform_prior is not None:
        matched = pca_waveform_prior.match(view, slice_id, timestamps_s, device=card.device, dtype=card.dtype)
        if matched is not None:
            agreement = cardiac_pca_waveform_subspace_loss(card, matched.waveform, ridge=pca_waveform_ridge, min_frames=pca_waveform_min_frames)
            pca = {"cardiac_pca_waveform_r2": float(agreement["r2"]), "cardiac_pca_waveform_loss": float(agreement["loss"]), "cardiac_pca_matched_frames": int(agreement["matched_frame_count"]), "selected_cardiac_pc": matched.selected_pc}
    card_target, card_wrong = amplitude(card, card_freq), amplitude(card, resp_freq)
    card_peak, resp_peak = dominant_peak_hz(card, timestamps_s), dominant_peak_hz(resp, timestamps_s)
    return {"view": view, "slice_id": slice_id, "n_frames": int(timestamps_s.numel()), "n_valid_frames": int(timestamps_s.numel()), "eligible": True, "skip_reason": None, "resp_target_amp": amplitude(resp, resp_freq), "resp_wrong_amp": amplitude(resp, card_freq), "card_target_amp": card_target, "card_wrong_amp": card_wrong, "card_target_over_wrong": card_target / max(card_wrong, torch.finfo(torch.float32).eps), "card_peak_hz": card_peak, "resp_peak_hz": resp_peak, "same_peak_exact": card_peak == resp_peak, "same_peak_within_0.02_hz": abs(card_peak - resp_peak) <= .02, "card_score_mean": float(card.mean()), "card_score_std": float(card.std(unbiased=False)), "cardiac_target_fraction": float(concentration["fraction"]), "cardiac_target_concentration": float(concentration["loss"]), "respiratory_source": local.respiratory_source, "cardiac_source": local.cardiac_source, **pca}


def _skip_record(view: str, slice_id: str, count: int, reason: str) -> dict[str, object]:
    return {"view": view, "slice_id": slice_id, "n_frames": count, "n_valid_frames": count, "eligible": False, "skip_reason": reason, "selected_cardiac_pc": None, "cardiac_pca_waveform_r2": None, "cardiac_pca_waveform_loss": None, "cardiac_pca_matched_frames": 0}


def _encode_location(model, observations: list, contract, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ordered = sorted(observations, key=lambda item: item.timestamp_s)
    times = torch.tensor([item.timestamp_s for item in ordered], dtype=torch.float64, device=device)
    encoded = [model.film_encoder(item.image.unsqueeze(0), center_mm=item.center_mm[None], row_direction=item.row_direction[None], column_direction=item.column_direction[None], normal=item.normal[None], pixel_spacing_mm=item.pixel_spacing_mm[None], slice_thickness_mm=torch.tensor([[item.slice_thickness_mm]], device=device, dtype=item.image.dtype)) for item in ordered]
    return (torch.cat([item["resp_scores"][:, :contract.active_respiratory_levels] for item in encoded], 0), torch.cat([item["card_scores"] for item in encoded], 0), times)


def all_location_frequency_records(model, observations: list, contract, prior, device: torch.device, *, pca_waveform_prior=None, pca_waveform_ridge: float = 1e-4, pca_waveform_min_frames: int = 8) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list] = {}
    for observation in observations:
        grouped.setdefault((observation.view, observation.slice_id), []).append(observation)
    records: list[dict[str, object]] = []
    for (view, slice_id), observations in sorted(grouped.items()):
        valid = [item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason)]
        if len(valid) < 3:
            records.append(_skip_record(view, slice_id, len(valid), "no_qc_valid_frames_hard_invalid_excluded" if not valid else "fewer_than_3_valid_frames")); continue
        try:
            local = prior.for_location(view, slice_id)
            resp, card, times = _encode_location(model, valid, contract, device)
            records.append(frequency_semantics_record(view, slice_id, resp, card, times, local, pca_waveform_prior=pca_waveform_prior, pca_waveform_ridge=pca_waveform_ridge, pca_waveform_min_frames=pca_waveform_min_frames))
        except ValueError as exc:
            records.append(_skip_record(view, slice_id, len(observations), f"unevaluable:{exc}"))
    return records


def _aggregate_motion(model, valid: list, contract, device: torch.device) -> dict[str, object]:
    magnitudes = {"resp": [], "card": []}
    for key in sorted({(item.view, item.slice_id) for item in valid}):
        observation = sorted([item for item in valid if (item.view, item.slice_id) == key], key=lambda item: item.timestamp_s)[0]
        render = model.predict(observation, torch.tensor([[0., 0.]], device=device), contract.name)
        if contract.enable_motion:
            resp_grid, _ = model._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, model.respiratory_smoothness_shape)
            resp = ScoreWeightedMBCField(model.respiratory_mbc, render["scores"]["resp_scores"])(resp_grid)
            magnitudes["resp"].append(torch.linalg.vector_norm(resp, dim=-1).reshape(-1))
            if contract.enable_cardiac:
                card_grid, _ = model._physical_grid(model.cardiac_lower_world_mm, model.cardiac_upper_world_mm, model.cardiac_smoothness_shape)
                card = ScoreWeightedMBCField(model.cardiac_mbc, render["scores"]["card_scores"])(card_grid)
                magnitudes["card"].append(torch.linalg.vector_norm(card, dim=-1).reshape(-1))
    def stats(prefix: str) -> dict[str, float | None]:
        magnitude = torch.cat(magnitudes[prefix]) if magnitudes[prefix] else torch.empty(0)
        if not magnitude.numel(): return {"rms_mm": None, "magnitude_p50_mm": None, "magnitude_p95_mm": None, "magnitude_p99_mm": None, "magnitude_max_mm": None}
        return {"rms_mm": float(magnitude.square().mean().sqrt()), "magnitude_p50_mm": float(torch.quantile(magnitude, .50)), "magnitude_p95_mm": float(torch.quantile(magnitude, .95)), "magnitude_p99_mm": float(torch.quantile(magnitude, .99)), "magnitude_max_mm": float(magnitude.max())}
    return {"population": {"n_locations": len({(item.view, item.slice_id) for item in valid}), "n_frames_evaluated": len({(item.view, item.slice_id) for item in valid}), "sampling_rule": "earliest QC-valid non-hard-invalid frame per location", "evaluation_grid_shape": {"respiratory": list(model.respiratory_smoothness_shape), "cardiac": list(model.cardiac_smoothness_shape)}, "physical_domain": "respiratory full canonical world-mm; cardiac local cardiac-box world-mm"}, "respiratory": stats("resp"), "cardiac": stats("card")}


def _representatives(items: list, count: int) -> list:
    if not items:
        return []
    indices = torch.linspace(0, len(items) - 1, min(count, len(items))).round().long().tolist()
    return [items[index] for index in dict.fromkeys(indices)]


def _cardiac_pixels(model, observation, maximum: int) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    linear = torch.arange(height * width, device=observation.image.device)
    uv = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)
    world = model._pixel_world(observation, uv)
    inside = ((world >= model.cardiac_lower_world_mm.to(world)) & (world <= model.cardiac_upper_world_mm.to(world))).all(-1)
    return _representatives(list(uv[inside]), maximum) if inside.any() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v3_change4 checkpoint diagnostic")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--all-locations", action="store_true", help="evaluate every QC-valid location and write full-location outputs")
    parser.add_argument("--all-locations-dir", type=Path, help="directory for all_location_diagnostic.json and all_location_frequency_semantics.csv")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8"))
    observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    valid = [item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason)]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_backend = detect_checkpoint_encoding_backend(checkpoint["model"])
    model = build_source_first_model(config, domain, n_dynamic_frames=len(valid), device=device, canonical_encoding_backend=checkpoint_backend).to(device)
    model.load_state_dict(checkpoint["model"]); prepare_read_only_diagnostic_model(model)
    prior = load_training_frequency_prior(args.frequency_bands, allow_template_fallback=False)
    temporal_config = config["training"].get("temporal_auxiliary", {})
    pca_waveform_prior = PCAWaveformPrior.load(args.frequency_bands, strict=False)
    locations = list(prior.locations.values())
    stage = checkpoint.get("training_state", {}).get("current_stage")
    if stage == "stage3":
        raise ValueError("legacy v3 Stage3 checkpoint is not compatible with v3_change4 diagnostic; use its Stage2c checkpoint")
    contract = stage_contract(str(stage))
    with torch.no_grad():
        probe = valid[0]
        render = model.predict(probe, torch.tensor([[0., 0.]], device=device), str(stage))
        motion = model.motion_regularizers(render["scores"], str(stage)) if contract.enable_motion else {}
        motion_stats = {key: float(value.detach().cpu()) for key, value in motion.items() if key.endswith(("rms_mm", "max_mm"))}
        ablation = None
        if contract.enable_cardiac:
            ablation_records = []
            for view in ("SAX", "2CH", "4CH"):
                for observation in _representatives([item for item in valid if item.view == view], 10):
                    pixels = _cardiac_pixels(model, observation, 512)
                    if not pixels:
                        continue
                    pixels = torch.stack(pixels)
                    resp_only = model.predict(observation, pixels, "stage2c")["predicted_intensity"]
                    joint = model.predict(observation, pixels, str(stage))["predicted_intensity"]
                    target = observation.image[0, pixels[:, 1].long(), pixels[:, 0].long()]
                    ablation_records.append({"view": view, "slice_id": observation.slice_id, "n_pixels": int(pixels.shape[0]), "resp_only_sse": float((resp_only - target).square().sum()), "resp_plus_card_sse": float((joint - target).square().sum()), "absolute_prediction_change_sum": float((joint - resp_only).abs().sum())})
            count = sum(item["n_pixels"] for item in ablation_records)
            if count:
                resp_mse = sum(item["resp_only_sse"] for item in ablation_records) / count; joint_mse = sum(item["resp_plus_card_sse"] for item in ablation_records) / count
                ablation = {"records": ablation_records, "n_observations": len(ablation_records), "n_pixels": count, "resp_only_mse": resp_mse, "resp_plus_card_mse": joint_mse, "relative_mse_improvement_percent": (resp_mse - joint_mse) / max(resp_mse, torch.finfo(torch.float32).eps) * 100., "mean_absolute_prediction_change": sum(item["absolute_prediction_change_sum"] for item in ablation_records) / count}
        frequency_records = []
        if contract.enable_motion:
            for view in ("SAX", "2CH", "4CH"):
                locations_for_view = sorted({item.slice_id for item in valid if item.view == view})
                for slice_id in _representatives(locations_for_view, 3):
                    grouped = [item for item in valid if item.view == view and item.slice_id == slice_id]
                    if len(grouped) != 50:
                        continue
                    grouped.sort(key=lambda item: item.timestamp_s)
                    times = torch.tensor([item.timestamp_s for item in grouped], dtype=torch.float64, device=device)
                    encoded = [model.film_encoder(item.image.unsqueeze(0), center_mm=item.center_mm[None], row_direction=item.row_direction[None], column_direction=item.column_direction[None], normal=item.normal[None], pixel_spacing_mm=item.pixel_spacing_mm[None], slice_thickness_mm=torch.tensor([[item.slice_thickness_mm]], device=device)) for item in grouped]
                    resp = torch.cat([item["resp_scores"][:, :contract.active_respiratory_levels] for item in encoded], 0); card = torch.cat([item["card_scores"] for item in encoded], 0)
                    frequency_records.append(frequency_semantics_record(view, slice_id, resp, card, times, prior.for_location(view, slice_id), pca_waveform_prior=pca_waveform_prior, pca_waveform_ridge=float(temporal_config.get("pca_waveform_ridge", 1e-4)), pca_waveform_min_frames=int(temporal_config.get("pca_waveform_min_frames", 8))))
        frequency_keys = ("resp_target_amp", "resp_wrong_amp", "card_target_amp", "card_wrong_amp", "resp_peak_hz", "card_peak_hz", "card_score_std", "cardiac_target_fraction", "cardiac_target_concentration", "cardiac_pca_waveform_r2", "cardiac_pca_waveform_loss", "cardiac_pca_matched_frames")
        def aggregate(key: str, *, median: bool) -> float | None:
            values = [float(item[key]) for item in frequency_records if item.get(key) is not None]
            return None if not values else (float(torch.tensor(values).median()) if median else sum(values) / len(values))
        frequency = {"records": frequency_records, "summary_mean": {key: aggregate(key, median=False) for key in frequency_keys} if frequency_records else None, "summary_median": {key: aggregate(key, median=True) for key in frequency_keys} if frequency_records else None}
        uncertainty = None
        if contract.enable_uncertainty:
            uncertainty = {key: _summary(value) for key, value in render["uncertainty"].items() if key in {"frame_variance", "pixel_scale", "variance"}}
        full_location = None
        if args.all_locations and contract.enable_motion:
            records = all_location_frequency_records(model, observations, contract, prior, device, pca_waveform_prior=pca_waveform_prior, pca_waveform_ridge=float(temporal_config.get("pca_waveform_ridge", 1e-4)), pca_waveform_min_frames=int(temporal_config.get("pca_waveform_min_frames", 8)))
            continuous = ("cardiac_pca_waveform_r2", "cardiac_pca_waveform_loss", "cardiac_target_fraction", "cardiac_target_concentration", "card_target_amp", "card_wrong_amp", "card_target_over_wrong", "resp_target_amp", "resp_wrong_amp", "card_peak_hz", "resp_peak_hz", "card_score_mean", "card_score_std")
            boolean = ("same_peak_exact", "same_peak_within_0.02_hz")
            full_location = {"population": "all QC-valid observations excluding hard-invalid frames; all distinct locations retained with structured skips", "records": records, "summary": summarize_records(records, continuous=continuous, boolean=boolean), "aggregate_motion_statistics": _aggregate_motion(model, valid, contract, device)}
    result = {
        "checkpoint_stage": stage, "canonical_encoding_backend": model.canonical.encoding_backend,
        "prior_audit": {"number_of_locations": len(locations), "local_respiratory_count": sum(item.respiratory_source == "phase1_per_location" for item in locations), "resp_global_fallback_count": sum(item.respiratory_source == "phase1_global_fallback" for item in locations), "local_cardiac_count": sum(item.cardiac_source == "phase1_per_location" for item in locations), "card_global_fallback_count": sum(item.cardiac_source == "phase1_global_fallback" for item in locations), "prior": asdict(prior), "pca_waveform_prior": pca_waveform_prior.metadata},
        "motion_statistics": motion_stats, "cardiac_ablation": ablation, "frequency_semantics": frequency, "uncertainty": uncertainty,
        "status": "read_only_no_training",
        "device": str(device),
    }
    if full_location is not None:
        output_dir = args.all_locations_dir or args.output_json.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        full_path = output_dir / "all_location_diagnostic.json"
        csv_path = output_dir / "all_location_frequency_semantics.csv"
        full_path.write_text(json.dumps(full_location, indent=2) + "\n", encoding="utf-8")
        keys = sorted({key for record in full_location["records"] for key in record})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(full_location["records"])
        result["all_locations"] = {"json": str(full_path), "csv": str(csv_path), "n_locations": len(full_location["records"])}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
