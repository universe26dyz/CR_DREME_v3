"""Formal config-to-runtime builder for the source-first dynamic mainline."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from .model import SourceFirstDynamicModel


def _triple(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a three-value list")
    result = tuple(int(item) for item in value)
    if min(result) <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def build_source_first_model(config: Mapping[str, Any], domain: Mapping[str, Any], *, n_dynamic_frames: int, device: torch.device, canonical_encoding_backend: str | None = None) -> SourceFirstDynamicModel:
    """Construct every formal source-backed object from the validated config."""
    model = config["model"]
    canonical = model["canonical"]
    respiratory = model["respiratory_mbc"]
    cardiac = model["cardiac_mbc"]
    psf = model["psf"]
    uncertainty = model["uncertainty"]
    film = model.get("film", {})
    training = config["training"]
    regularization = training["motion_regularization"]
    legacy_grid = regularization.get("evaluation_grid", {}).get("shape", [16, 16, 16])
    resp_grid = regularization.get("respiratory_evaluation_grid", {}).get("shape", legacy_grid)
    card_grid = regularization.get("cardiac_evaluation_grid", {}).get("shape", legacy_grid)
    image_reg = training.get("image_regularization", {"mode": "edge", "delta": 1.})
    return SourceFirstDynamicModel(
        torch.as_tensor(domain["world_min_mm"], dtype=torch.float32, device=device),
        torch.as_tensor(domain["world_max_mm"], dtype=torch.float32, device=device),
        cardiac_lower_world_mm=torch.as_tensor(domain["cardiac_box"]["min_mm"], dtype=torch.float32, device=device),
        cardiac_upper_world_mm=torch.as_tensor(domain["cardiac_box"]["max_mm"], dtype=torch.float32, device=device),
        n_dynamic_frames=n_dynamic_frames,
        canonical_coarsest_resolution=float(canonical["coarsest_resolution"]),
        canonical_finest_resolution=float(canonical["finest_resolution"]),
        canonical_level_scale=float(canonical["level_scale"]),
        canonical_features_per_level=int(canonical["features_per_level"]),
        canonical_log2_hashmap_size=int(canonical["log2_hashmap_size"]),
        latent_dim=int(canonical["latent_dim"]),
        inr_width=int(canonical["width"]),
        inr_depth=int(canonical["depth"]),
        canonical_spatial_scaling=float(canonical.get("spatial_scaling", 1.)),
        canonical_encoding_backend=canonical_encoding_backend,
        motion_hidden_dim=int(respiratory["hidden_dim"]),
        respiratory_grid_shapes=tuple(_triple(shape, "model.respiratory_mbc.logical_control_shapes") for shape in respiratory["logical_control_shapes"]),
        respiratory_cps=_triple(respiratory["cps"], "model.respiratory_mbc.cps") if isinstance(respiratory["cps"], (list, tuple)) else int(respiratory["cps"]),
        cardiac_grid_shape=_triple(cardiac["logical_control_shape"], "model.cardiac_mbc.logical_control_shape"),
        cardiac_cps=_triple(cardiac["cps"], "model.cardiac_mbc.cps") if isinstance(cardiac["cps"], (list, tuple)) else int(cardiac["cps"]),
        cardiac_taper_mm=float(cardiac["taper_mm"]),
        psf_samples=int(psf["n_samples"]),
        film_channels=int(film.get("channels", respiratory["hidden_dim"])),
        film_position_bands=int(film.get("position_bands", 4)),
        uncertainty_frame_embedding_dim=int(uncertainty.get("frame_embedding_dim", 8)),
        uncertainty_width=int(uncertainty.get("width", canonical["width"])),
        uncertainty_depth=int(uncertainty.get("depth", canonical["depth"])),
        respiratory_smoothness_shape=_triple(resp_grid, "training.motion_regularization.respiratory_evaluation_grid.shape"),
        cardiac_smoothness_shape=_triple(card_grid, "training.motion_regularization.cardiac_evaluation_grid.shape"),
        image_regularization_mode=str(image_reg.get("mode", "edge")),
        image_regularization_delta=float(image_reg.get("delta", 1.)),
    )


def effective_model_config(model: SourceFirstDynamicModel) -> dict[str, Any]:
    """Serialize facts from constructed objects, never requested YAML values."""
    canonical = model.canonical.args
    resp = model.respiratory_mbc
    cardiac = model.cardiac_mbc
    return {
        "classes": {
            "canonical": f"{type(model.canonical.inr).__module__}.{type(model.canonical.inr).__name__}",
            "psf": f"{type(model.psf).__module__}.{type(model.psf).__name__}",
            "respiratory_siren": f"{type(resp.levels[0].siren).__module__}.{type(resp.levels[0].siren).__name__}",
            "cardiac_ffd": f"{type(cardiac.basis.ffd).__module__}.{type(cardiac.basis.ffd).__name__}",
        },
        "canonical": {**{key: getattr(canonical, key) for key in ("coarsest_resolution", "finest_resolution", "level_scale", "n_features_per_level", "log2_hashmap_size", "n_features_z", "width", "depth")}, "encoding_backend": model.canonical.encoding_backend},
        "psf": {"n_samples": model.psf.n_samples},
        "respiratory_mbc": {"logical_control_shapes": [list(level.logical_control_shape) for level in resp.levels], "cps": list(resp.levels[0].cps), "dense_evaluation_shapes": [list(level.dense_evaluation_shape) for level in resp.levels], "padded_control_shapes": [list(level.padded_control_shape) for level in resp.levels], "grid_spacing_mm": [level.grid_spacing_mm.detach().cpu().tolist() for level in resp.levels]},
        "cardiac_mbc": {"logical_control_shape": list(cardiac.basis.logical_control_shape), "cps": list(cardiac.basis.cps), "dense_evaluation_shape": list(cardiac.basis.dense_evaluation_shape), "padded_control_shape": list(cardiac.basis.padded_control_shape), "taper_mm": cardiac.taper_mm},
        "film": {"channels": model.film_encoder.image[0].out_channels, "position_bands": model.film_encoder.position_bands},
        "uncertainty": {"frame_embedding_dim": model.uncertainty.frame_embedding.embedding_dim, "width": model.uncertainty.sigma_net[0].out_features if len(model.uncertainty.sigma_net) else None},
        "smoothness": {"respiratory_shape": list(model.respiratory_smoothness_shape), "cardiac_shape": list(model.cardiac_smoothness_shape)},
        "image_regularization": {"mode": model.image_regularization_mode, "delta": model.image_regularization_delta},
    }
