#!/usr/bin/env python3
"""Validate a source-first real-data run without taking an optimizer step."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.losses.frequency_loss import resolved_band_frequencies
from cardioresp4d.training.build_model import build_source_first_model, effective_model_config
from cardioresp4d.training.runtime_state import is_hard_invalid_reason, set_reproducibility
from cardioresp4d.training.source_first_config import validate_source_dependencies, validate_source_first_config
from train_source_first import observations_from_manifest


def _location_report(observations, prior) -> list[dict]:
    grouped = defaultdict(list)
    for item in observations:
        if item.qc_valid and not is_hard_invalid_reason(item.qc_reason):
            grouped[(item.view, item.slice_id)].append(item)
    report = []
    for (view, slice_id), items in sorted(grouped.items()):
        times = torch.tensor(sorted(item.timestamp_s for item in items), dtype=torch.float64)
        selected = prior.for_location(view, slice_id)
        reason = None
        if len(items) < 3:
            reason = "fewer_than_three_valid_frames"
        elif float(times[-1] - times[0]) <= 0:
            reason = "nonpositive_timestamp_duration"
        else:
            try:
                frequencies = resolved_band_frequencies(selected.cardiac_bands_hz, times)
            except ValueError as exc:
                reason = str(exc); frequencies = []
        report.append({"view": view, "slice_id": slice_id, "n_valid_frames": len(items), "duration_s": float(times[-1] - times[0]) if len(items) else 0., "min_dt_s": float((times[1:] - times[:-1]).min()) if len(items) > 1 else None, "max_dt_s": float((times[1:] - times[:-1]).max()) if len(items) > 1 else None, "median_dt_s": float((times[1:] - times[:-1]).median()) if len(items) > 1 else None, "frequency_resolution_hz": 1. / float(times[-1] - times[0]) if len(items) > 1 and times[-1] > times[0] else None, "frequency_prior_source": selected.source, "resolved_cardiac_frequencies_hz": frequencies if reason is None else [], "status": "usable" if reason is None else "skipped", "reason": reason})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 source-first no-step real-data preflight")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path)
    parser.add_argument("--canonical-domain", required=True, type=Path); parser.add_argument("--frequency-bands", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    validate_source_first_config(config, PROJECT_ROOT); validate_source_dependencies()
    source_lock = verify_vendored_source_lock(PROJECT_ROOT)
    reproducibility = set_reproducibility(int(config["training"]["seed"]))
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8"))
    lower, upper = domain["world_min_mm"], domain["world_max_mm"]
    cardiac = domain["cardiac_box"]
    if any(high <= low for low, high in zip(lower, upper)) or any(low < outer_low or high > outer_high or high <= low for low, high, outer_low, outer_high in zip(cardiac["min_mm"], cardiac["max_mm"], lower, upper)):
        raise ValueError("canonical domain/card box bounds are invalid")
    coverage = domain.get("coverage")
    if not isinstance(coverage, dict) or set(("SAX", "2CH", "4CH")) - set(coverage):
        raise ValueError("canonical domain lacks per-view geometry coverage QC")
    coverage_report = {"plane_center_qc": coverage, "canonical_hole_interpretation": "plane-center coverage zeros are geometry-QC observations, never canonical-volume holes", "psf_aware_coverage_required": bool(config.get("coverage", {}).get("psf_aware", False)), "view_count_required": bool(config.get("coverage", {}).get("view_count", False)), "observation_count_required": bool(config.get("coverage", {}).get("observation_count", False))}
    observations, normalization, normalization_groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    prior = load_training_frequency_prior(args.frequency_bands, allow_template_fallback=False)
    model = build_source_first_model(config, domain, n_dynamic_frames=sum(item.qc_valid for item in observations), device=device).train()
    valid = [item for item in observations if item.qc_valid]
    examples = {item.view: item for item in valid}
    missing = sorted({"SAX", "2CH", "4CH"} - set(examples))
    if missing:
        raise ValueError("preflight lacks valid required views: " + ", ".join(missing))
    no_grad = {}
    with torch.no_grad():
        for stage in ("stage1", "stage2a", "stage2b", "stage2c", "stage3"):
            output = model.predict(examples["SAX"], torch.tensor([[0., 0.]], device=device), stage)
            no_grad[stage] = {"predicted_shape": list(output["predicted_intensity"].shape), "finite": bool(torch.isfinite(output["predicted_intensity"]).all()), "has_uncertainty": "uncertainty" in output}
    locations = _location_report(observations, prior)
    result = {"status": "PASS", "source_lock": source_lock, "source_lock_verified": bool(source_lock.get("verified")), "reproducibility": reproducibility, "effective_config": effective_model_config(model), "normalization_parameters": normalization, "normalization_groups": normalization_groups, "coverage": coverage_report, "counts": {"observations": len(observations), "valid": len(valid), "hard_invalid": sum(not item.qc_valid for item in observations)}, "location_frequency_preflight": locations, "no_grad_stage_forwards": no_grad, "device": str(device), "torch_version": torch.__version__}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
