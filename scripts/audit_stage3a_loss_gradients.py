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
from cardioresp4d.diagnostics.change5c import choose_representative_indices, cosine_similarity, deterministic_pixel_indices, diagnostic_rng, l2_norm, stable_diagnostic_seed
from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformPrior
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import is_hard_invalid_reason
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer
from train_source_first import observations_from_manifest


SEMANTIC = ("respiratory_leakage_in_card", "cardiac_target_concentration", "cardiac_pca_waveform", "zero_mean_score")
REGULARIZATION = ("image", "mbc_normalization", "smooth_resp", "smooth_card", "cardiac_leakage_in_resp")


def _clear_gradients(modules: dict[str, list[torch.nn.Parameter]]) -> None:
    for parameters in modules.values():
        for parameter in parameters:
            parameter.grad = None


def _gradient_vector(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([(parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1) for parameter in parameters]) if parameters else torch.empty(0)


def collect_loss_gradients(losses: dict[str, torch.Tensor], weights: dict[str, float], modules: dict[str, list[torch.nn.Parameter]], *, combined: dict[str, tuple[str, ...]] | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """Backpropagate individual and true weighted-sum losses without an optimizer."""
    terms: dict[str, torch.Tensor] = dict(losses)
    for name, members in (combined or {}).items():
        active = [losses[key] * float(weights.get(key, 1.)) for key in members if key in losses]
        if active:
            terms[name] = torch.stack(active).sum()
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, loss in terms.items():
        _clear_gradients(modules)
        if loss.requires_grad:
            loss.backward(retain_graph=True)
        result[name] = {module: _gradient_vector(parameters) for module, parameters in modules.items()}
    _clear_gradients(modules)
    return result


def deterministic_sample_pixels(trainer: UnifiedProgressiveTrainer, observation, *, seed: int) -> torch.Tensor:
    height, width = observation.image.shape[-2:]; total = height * width
    linear = torch.arange(total, device=observation.image.device)
    uv = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), -1).to(observation.image.dtype)
    world = trainer.model._pixel_world(observation, uv)
    inside = ((world >= trainer.model.cardiac_lower_world_mm.to(world)) & (world <= trainer.model.cardiac_upper_world_mm.to(world))).all(-1)
    selected = deterministic_pixel_indices(total, linear[inside], count=min(trainer.pixel_samples, total), cardiac_fraction=trainer.cardiac_sampling_fraction, seed=seed)
    return torch.stack((selected.remainder(width), torch.div(selected, width, rounding_mode="floor")), -1).to(observation.image.dtype)


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


def legacy_main() -> None:
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


def _describe(observation) -> dict:
    return {"view": observation.view, "slice_id": observation.slice_id, "dynamic_frame_id": int(observation.dynamic_frame_id), "timestamp_s": float(observation.timestamp_s)}


def _deterministic_components(trainer: UnifiedProgressiveTrainer, data_frames: list, temporal_frames: list, *, seed: int) -> dict[str, torch.Tensor]:
    original_pixels, original_temporal = trainer.sample_pixels, trainer.sampler.temporal_batch
    trainer.sample_pixels = lambda observation: deterministic_sample_pixels(trainer, observation, seed=seed + int(observation.dynamic_frame_id))  # type: ignore[method-assign]
    trainer.sampler.temporal_batch = lambda *, max_items: temporal_frames[:max_items]  # type: ignore[method-assign]
    try:
        rows = []
        for item in data_frames:
            psf_seed = stable_diagnostic_seed(seed, item.view, item.slice_id, item.dynamic_frame_id, purpose="psf")
            with diagnostic_rng(psf_seed, item.image.device):
                rows.append(trainer._observation_components(item, "stage3a"))
        components = {name: torch.stack([row[name] for row in rows]).mean() for name in rows[0]}
        components.update(trainer._temporal_components("stage3a"))
        return components
    finally:
        trainer.sample_pixels, trainer.sampler.temporal_batch = original_pixels, original_temporal  # type: ignore[method-assign]


def _gradient_summary(components: dict[str, torch.Tensor], trainer: UnifiedProgressiveTrainer, modules: dict[str, list[torch.nn.Parameter]]) -> dict:
    weights = {"data": 1., **trainer.loss_weights}
    loss_names = ("data", *REGULARIZATION, *SEMANTIC)
    losses = {name: components[name] for name in loss_names if name in components}
    vectors = collect_loss_gradients(losses, weights, modules, combined={"semantic_aux": SEMANTIC, "regularization_aux": REGULARIZATION, "total": tuple(losses)})
    result = {}
    for name, grouped in vectors.items():
        combined = name in {"semantic_aux", "regularization_aux", "total"}
        result[name] = {module: ({"combined_l2_norm": l2_norm(vector)} if combined else {"raw_l2_norm": l2_norm(vector), "weighted_l2_norm": abs(float(weights.get(name, 1.))) * l2_norm(vector), "configured_weight": float(weights.get(name, 1.))}) for module, vector in grouped.items()}
    pairs = (("data", "cardiac_target_concentration"), ("data", "cardiac_pca_waveform"), ("data", "zero_mean_score"), ("cardiac_target_concentration", "cardiac_pca_waveform"), ("data", "semantic_aux"), ("data", "total"))
    result["cosine_similarity"] = {"cardiac_film_head": {f"{left}_vs_{right}": cosine_similarity(vectors[left]["cardiac_film_head"], vectors[right]["cardiac_film_head"]) for left, right in pairs if left in vectors and right in vectors}, "undefined_policy": "null means at least one vector has zero L2 norm"}
    for module in ("cardiac_film_head", "cardiac_mbc"):
        data_norm = l2_norm(vectors["data"][module]) if "data" in vectors else 0.
        result.setdefault("combined_ratios", {})[module] = {"semantic_aux_to_data_norm_ratio": None if data_norm == 0. or "semantic_aux" not in vectors else l2_norm(vectors["semantic_aux"][module]) / data_norm, "regularization_aux_to_data_norm_ratio": None if data_norm == 0. or "regularization_aux" not in vectors else l2_norm(vectors["regularization_aux"][module]) / data_norm}
    result["configured_weights"] = weights
    return result


def _step_summary(audits: list[dict]) -> dict:
    def summarize(values: list[float]) -> dict[str, float | None]:
        if not values: return {"mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
        value = torch.tensor(values, dtype=torch.float64); return {"mean": float(value.mean()), "median": float(value.median()), "q25": float(torch.quantile(value, .25)), "q75": float(torch.quantile(value, .75)), "min": float(value.min()), "max": float(value.max())}
    ratios = {name: summarize([float(audit["combined_ratios"]["cardiac_film_head"][name]) for audit in audits if audit.get("combined_ratios", {}).get("cardiac_film_head", {}).get(name) is not None]) for name in ("semantic_aux_to_data_norm_ratio", "regularization_aux_to_data_norm_ratio")}
    cosines = {name: summarize([float(audit["cosine_similarity"]["cardiac_film_head"][name]) for audit in audits if audit.get("cosine_similarity", {}).get("cardiac_film_head", {}).get(name) is not None]) for name in ("data_vs_semantic_aux", "data_vs_total", "data_vs_cardiac_target_concentration", "data_vs_cardiac_pca_waveform", "data_vs_zero_mean_score", "cardiac_target_concentration_vs_cardiac_pca_waveform")}
    return {"cardiac_film_head_ratios": ratios, "cardiac_film_head_cosines": cosines}


def _data_norm_distribution(trainer: UnifiedProgressiveTrainer, data: list, temporal: list, *, seed: int, modules: dict[str, list[torch.nn.Parameter]]) -> dict[str, float]:
    norms = []
    for observation in data:
        component = _deterministic_components(trainer, [observation], temporal, seed=seed)
        vector = collect_loss_gradients({"data": component["data"]}, {"data": 1.}, modules)["data"]["cardiac_film_head"]
        norms.append(l2_norm(vector))
    values = torch.tensor(norms, dtype=torch.float64)
    return {"mean": float(values.mean()), "median": float(values.median()), "q25": float(torch.quantile(values, .25)), "q75": float(torch.quantile(values, .75)), "min": float(values.min()), "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--view"); parser.add_argument("--slice-id")
    parser.add_argument("--mode", choices=("location-matched", "training-step-matched"), default="location-matched"); parser.add_argument("--data-frames", type=int, default=8); parser.add_argument("--pseudo-steps", type=int, default=16); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8")); observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    valid = [item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason)]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_source_first_model(config, domain, n_dynamic_frames=len(valid), device=device, canonical_encoding_backend=detect_checkpoint_encoding_backend(checkpoint["model"])).to(device); model.load_state_dict(checkpoint["model"])
    auxiliary = config["training"].get("temporal_auxiliary", {})
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(valid, seed=args.seed), pixel_samples=args.pixel_samples, frequency_prior=load_training_frequency_prior(args.frequency_bands, allow_template_fallback=False), pca_waveform_prior=PCAWaveformPrior.load(args.frequency_bands, strict=False), pca_waveform_ridge=float(auxiliary.get("pca_waveform_ridge", 1e-4)), pca_waveform_min_frames=int(auxiliary.get("pca_waveform_min_frames", 8)), temporal_every=int(auxiliary.get("every_steps", 1)), temporal_batch_size=int(auxiliary.get("max_frames", 50)), cardiac_sampling_fraction=float(config["training"].get("cardiac_sampling_fraction", .8)), loss_weights=config["training"]["loss_weights"])
    trainer._configure_stage("stage3a")
    frozen = [parameter for module in (model.canonical, model.respiratory_mbc, model.film_encoder.image, model.film_encoder.film, model.film_encoder.geometry_mlp, model.film_encoder.head, model.film_encoder.resp, model.uncertainty) for parameter in module.parameters()]
    modules = {"cardiac_film_head": list(model.film_encoder.card.parameters()), "cardiac_mbc": list(model.cardiac_mbc.parameters()), "frozen_modules": frozen}
    selections, audits = [], []
    if args.mode == "location-matched":
        selected = sorted([item for item in valid if (args.view is None or item.view == args.view) and (args.slice_id is None or item.slice_id == args.slice_id)], key=lambda item: item.timestamp_s)
        if len(selected) < 3: parser.error("selected location has fewer than three QC-valid non-hard-invalid frames")
        data = [selected[index] for index in choose_representative_indices(len(selected), args.data_frames)]
        temporal = [selected[index] for index in choose_representative_indices(len(selected), trainer.temporal_batch_size)]
        selections.append({"data_frames": [_describe(item) for item in data], "temporal_frames": [_describe(item) for item in temporal]})
        audit = _gradient_summary(_deterministic_components(trainer, data, temporal, seed=args.seed), trainer, modules)
        audits.append(audit | {"data_gradient_distribution_cardiac_film_head": _data_norm_distribution(trainer, data, temporal, seed=args.seed, modules=modules)})
    else:
        for step in range(args.pseudo_steps):
            data = trainer.sampler.sample_step(); temporal = trainer.sampler.temporal_batch(max_items=trainer.temporal_batch_size)
            selections.append({"step": step, "data_frames": [_describe(item) for item in data], "temporal_frames": [_describe(item) for item in temporal]})
            audit = _gradient_summary(_deterministic_components(trainer, data, temporal, seed=args.seed), trainer, modules)
            audits.append(audit | {"data_gradient_distribution_cardiac_film_head": _data_norm_distribution(trainer, data, temporal, seed=args.seed, modules=modules)})
    leakage = any(value.get("frozen_modules", {}).get("raw_l2_norm", value.get("frozen_modules", {}).get("combined_l2_norm", 0.)) > 0. for audit in audits for value in audit.values() if isinstance(value, dict))
    payload = {"status": "read_only_no_optimizer_step", "stage": "stage3a", "mode": args.mode, "seed": args.seed, "pixel_selection_policy": {"pixel_samples": args.pixel_samples, "cardiac_sampling_fraction": trainer.cardiac_sampling_fraction, "semantics": "deterministic training-equivalent cardiac-priority/global sampling with replacement"}, "psf_seed_policy": {"seed_identity": "global seed + view + slice_id + dynamic_frame_id", "purpose": "psf", "rng_isolation": "torch.random.fork_rng CPU/CUDA", "cross_checkpoint_pairing": "experiment label is excluded"}, "selections": selections, "audits": audits, "training_step_aggregate": _step_summary(audits) if args.mode == "training-step-matched" else None, "frozen_modules_have_gradients": leakage, "cross_checkpoint_contract": "With identical manifest/config/seed, serialized frame IDs, pixels, and PSF realization are identical across checkpoints."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "audit_units": len(audits), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
