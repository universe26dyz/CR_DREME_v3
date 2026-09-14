"""Hard gates for a formal v3 source-first training configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

NESVOR_COMMIT = "2e96a91bdd30174210caea911e03a2778c65adbe"
SINR_COMMIT = "1a524ca7ae453b55310595fe957245088a108233"


def validate_source_first_config(config: Mapping[str, Any], project_root: str | Path) -> None:
    """Reject legacy/cardiac-only settings before a v3 trainer is constructed."""
    root = Path(project_root)
    if not (root / "SOURCE_PROVENANCE.md").is_file():
        raise ValueError("SOURCE_PROVENANCE.md is required before source-first training")
    model = _mapping(config, "model")
    canonical = _mapping(model, "canonical")
    respiratory = _mapping(model, "respiratory_mbc")
    cardiac = _mapping(model, "cardiac_mbc")
    training = _mapping(config, "training")
    domain = _mapping(config, "domain")
    if canonical.get("implementation") != "nesvor_official":
        raise ValueError("v3 mainline requires model.canonical.implementation = nesvor_official")
    if canonical.get("upstream_commit") != NESVOR_COMMIT:
        raise ValueError("v3 mainline requires the audited NeSVoR commit")
    if respiratory.get("implementation") != "sinr_official" or cardiac.get("implementation") != "sinr_official":
        raise ValueError("v3 mainline requires source-based SINR for respiratory and cardiac MBCs")
    if respiratory.get("upstream_commit") != SINR_COMMIT:
        raise ValueError("v3 mainline requires the audited SINR commit")
    if domain.get("reconstruction_domain") == "cardiac_box" and config.get("experiment_mode") != "cardiac_only_ablation":
        raise ValueError("cardiac_box is only permitted for explicit cardiac_only_ablation")
    if domain.get("reconstruction_domain") != "full_acquisition_supported" and config.get("experiment_mode") != "cardiac_only_ablation":
        raise ValueError("v3 mainline requires a full acquisition-supported reconstruction domain")
    if domain.get("cardiac_box_is_crop") is not False and config.get("experiment_mode") != "cardiac_only_ablation":
        raise ValueError("v3 mainline requires cardiac_box_is_crop = false")
    if _mapping(model, "uncertainty").get("enable_stage") != "stage3c":
        raise ValueError("v3_change4 formal uncertainty schedule is stage3c")
    if training.get("view_balanced") is not True:
        raise ValueError("v3 formal training requires view_balanced = true")
    if training.get("fixed_location_balanced") is not True:
        raise ValueError("v3 formal training requires fixed_location_balanced = true")
    if not isinstance(training.get("seed"), int):
        raise ValueError("v3 formal training requires integer training.seed")
    for key in ("psf", "film", "uncertainty", "respiratory_mbc", "cardiac_mbc"):
        if not isinstance(model.get(key), Mapping):
            raise ValueError(f"source-first configuration requires model.{key}")
    regularization = _mapping(training, "motion_regularization")
    if not isinstance(regularization.get("respiratory_evaluation_grid"), Mapping) or not isinstance(regularization.get("cardiac_evaluation_grid"), Mapping):
        raise ValueError("v3 formal motion regularization requires separate respiratory and cardiac grids")
    for name in ("stage3a", "stage3b", "stage3c"):
        if not isinstance(training.get(name), Mapping):
            raise ValueError(f"v3_change4 source-first configuration requires training.{name}")
    if training["stage3a"].get("cardiac_warmup") is not True or training["stage3a"].get("uncertainty_enabled") is not False:
        raise ValueError("v3_change4 stage3a must be cardiac warm-up without uncertainty")
    if training["stage3b"].get("joint_cardiorespiratory") is not True or training["stage3b"].get("uncertainty_enabled") is not False:
        raise ValueError("v3_change4 stage3b must be joint refinement without uncertainty")
    if training["stage3c"].get("uncertainty_enabled") is not True:
        raise ValueError("v3_change4 stage3c must enable uncertainty")


def validate_source_dependencies() -> None:
    """Fail early unless the exact upstream-backed adapter modules import."""
    from cardioresp4d.adapters.film import FiLMAdapter  # noqa: F401
    from cardioresp4d.adapters.nesvor_inr import NeSVoRCanonicalAdapter  # noqa: F401
    from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter  # noqa: F401
    from cardioresp4d.adapters.nesvor_uncertainty import NeSVoRDynamicFrameUncertainty  # noqa: F401
    from cardioresp4d.adapters.sinr_mbc import SINRFFDBasis  # noqa: F401


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"source-first configuration requires mapping {key}")
    return nested
