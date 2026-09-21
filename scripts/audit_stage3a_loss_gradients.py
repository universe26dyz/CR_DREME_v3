#!/usr/bin/env python3
"""Read-only per-loss Stage3a gradient audit; it never creates an optimizer step."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformPrior
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import is_hard_invalid_reason
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer
from train_source_first import observations_from_manifest


def _norm(parameters) -> float:
    values = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return 0. if not values else float(torch.stack(values).sum().sqrt())


def report_loss_gradients(losses: dict[str, torch.Tensor], weight: float | dict[str, float], modules: dict[str, list[torch.nn.Parameter]]) -> dict[str, dict]:
    """Differentiate each scalar independently without stepping parameters."""
    result = {}
    for name, loss in losses.items():
        for parameters in modules.values():
            for parameter in parameters: parameter.grad = None
        configured_weight = float(weight.get(name, 1.) if isinstance(weight, dict) else weight)
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        raw = {module: _norm(parameters) for module, parameters in modules.items()}
        result[name] = {"raw": raw, "weighted": {module: configured_weight * value for module, value in raw.items()}, "configured_weight": configured_weight, "mathematically_reaches": [module for module, value in raw.items() if value > 0.]}
    for parameters in modules.values():
        for parameter in parameters: parameter.grad = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--view"); parser.add_argument("--slice-id")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8"))
    observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    valid = [item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason)]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_source_first_model(config, domain, n_dynamic_frames=len(valid), device=device, canonical_encoding_backend=detect_checkpoint_encoding_backend(checkpoint["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    selected = [item for item in valid if (args.view is None or item.view == args.view) and (args.slice_id is None or item.slice_id == args.slice_id)]
    if len(selected) < 3: parser.error("selected location has fewer than three QC-valid non-hard-invalid frames")
    prior = load_training_frequency_prior(args.frequency_bands, allow_template_fallback=False)
    auxiliary = config["training"].get("temporal_auxiliary", {})
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(valid, seed=0), pixel_samples=1, frequency_prior=prior, pca_waveform_prior=PCAWaveformPrior.load(args.frequency_bands, strict=False), pca_waveform_ridge=float(auxiliary.get("pca_waveform_ridge", 1e-4)), pca_waveform_min_frames=int(auxiliary.get("pca_waveform_min_frames", 8)), temporal_every=int(auxiliary.get("every_steps", 1)), temporal_batch_size=int(auxiliary.get("max_frames", 50)), cardiac_sampling_fraction=float(config["training"].get("cardiac_sampling_fraction", .8)), loss_weights=config["training"]["loss_weights"])
    trainer._configure_stage("stage3a")
    trainer.sampler.temporal_batch = lambda *, max_items: sorted(selected, key=lambda item: item.timestamp_s)[:max_items]  # type: ignore[method-assign]
    components = trainer._observation_components(selected[0], "stage3a")
    components.update(trainer._temporal_components("stage3a"))
    loss_names = ("data", "image", "mbc_normalization", "smooth_resp", "smooth_card", "zero_mean_score", "cardiac_leakage_in_resp", "respiratory_leakage_in_card", "cardiac_target_concentration", "cardiac_pca_waveform")
    losses = {name: components[name] for name in loss_names if name in components}
    shared = [parameter for module in (model.canonical, model.respiratory_mbc, model.film_encoder.image, model.film_encoder.film, model.film_encoder.geometry_mlp, model.film_encoder.head, model.film_encoder.resp, model.uncertainty) for parameter in module.parameters()]
    modules = {"cardiac_film_head": list(model.film_encoder.card.parameters()), "cardiac_mbc": list(model.cardiac_mbc.parameters()), "frozen_modules": shared}
    report = report_loss_gradients(losses, {"data": 1., **trainer.loss_weights}, modules)
    payload = {"status": "read_only_no_optimizer_step", "location": {"view": selected[0].view, "slice_id": selected[0].slice_id, "n_valid_frames": len(selected)}, "stage": "stage3a", "losses": report, "frozen_modules_have_gradients": any(value["raw"]["frozen_modules"] > 0. for value in report.values())}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"location": payload["location"], "losses": list(report), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
