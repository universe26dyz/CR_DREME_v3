"""Isolated Exp2-A hard reference-score gauge protocol contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.visualization.dynamic_audit import fixed_physical_grid, static_dynamic_rms


EXP2A_PROTOCOL = {
    "experiment": "respiratory_reference_gauge_exp2a",
    "seed": 20260823,
    "observations_per_update": 32,
    "pixel_samples": 256,
    "steps": 200,
    "active_respiratory_levels": 1,
    "canonical_frozen": True,
    "encoder_lr": 5e-4,
    "mbc_lr": 5e-4,
    "canonical_lr": None,
    "normalization": "per_series",
    "score_policy": "raw_minus_same_encoder_reference_mean",
}


def exp2a_trainable_parameter_groups(parameters: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
    """Return exactly the Exp2-A owners, excluding canonical and inactive MBC levels."""
    return {"film": list(parameters.get("film", [])), "respiratory_mbc": list(parameters.get("respiratory_mbc", []))[:1]}


def validate_exp2a_protocol(config: Mapping[str, Any]) -> None:
    for key, expected in EXP2A_PROTOCOL.items():
        if key in config and config[key] != expected:
            raise ValueError(f"Exp2-A protocol mismatch for {key}: expected {expected!r}")


def exp2a_provenance(*, manifest: Path, qc_table: Path, canonical_domain: Path, static_manifest: Path, mean_manifest: Path, config: Mapping[str, Any], static_model: Path | None = None) -> dict[str, Any]:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"protocol": dict(EXP2A_PROTOCOL), "manifest_sha256": digest(manifest), "qc_table_sha256": digest(qc_table), "canonical_domain_sha256": digest(canonical_domain), "static_manifest_sha256": digest(static_manifest), "mean_manifest_sha256": digest(mean_manifest), "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    if static_model is not None:
        result["static_model_sha256"] = digest(static_model)
    return result


def exp2a_parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Freeze canonical and inactive levels; return only Exp2-A optimizer owners."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.respiratory_mbc.set_active_levels(1)
    film = list(model.film_encoder.parameters())
    mbc = list(model.respiratory_mbc.levels[0].parameters())
    for parameter in film + mbc:
        parameter.requires_grad_(True)
    return {"film": film, "respiratory_mbc": mbc}


def exp2a_gauge_diagnostics(model: Any, sampler: Any, *, grid_shape: tuple[int, int, int] = (16, 16, 16)) -> dict[str, float]:
    """Compute raw/reference/effective score and DVF gauges on one fixed grid."""
    cohort = tuple((view, slice_id, tuple(sorted(frames, key=lambda item: item.timestamp_s))) for view in sorted(sampler._grouped) for slice_id, frames in sorted(sampler._grouped[view].items()) if frames)
    if not cohort:
        raise ValueError("Exp2-A gauge diagnostics require a non-empty cohort")
    grid, _ = fixed_physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, grid_shape)
    score_values = {name: [] for name in ("raw", "reference", "effective")}; dvf_values = {name: [] for name in score_values}
    with torch.no_grad():
        for _, _, frames in cohort:
            scores_by_kind = {"raw": [], "reference": [], "effective": []}
            for observation in frames:
                raw = model.encode_motion_scores_raw(observation)["resp_scores"][:, :1]
                reference = model.encode_reference_scores(observation)["resp_scores"][:, :1]
                effective = model.encode_motion_scores(observation)["resp_scores"][:, :1]
                scores_by_kind["raw"].append(raw[0]); scores_by_kind["reference"].append(reference[0]); scores_by_kind["effective"].append(effective[0])
            for name, values in scores_by_kind.items():
                sequence = torch.stack(values); score_values[name].append(sequence)
                dvf_values[name].append(torch.stack([ScoreWeightedMBCField(model.respiratory_mbc, value[None])(grid)[0] for value in values]))
    result: dict[str, float] = {}
    for name in ("raw", "reference", "effective"):
        score_metrics = static_dynamic_rms(torch.cat(score_values[name], dim=0)); dvf_metrics = static_dynamic_rms(torch.cat(dvf_values[name], dim=0))
        result.update({f"{name}_score_static_rms": score_metrics["static_rms"], f"{name}_score_dynamic_rms": score_metrics["dynamic_rms"], f"{name}_score_static_dynamic_ratio": score_metrics["static_dynamic_ratio"], f"{name}_dvf_static_rms_mm": dvf_metrics["static_rms"], f"{name}_dvf_dynamic_rms_mm": dvf_metrics["dynamic_rms"], f"{name}_dvf_static_dynamic_ratio": dvf_metrics["static_dynamic_ratio"]})
    return result
