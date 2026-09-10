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
    if _mapping(model, "uncertainty").get("enable_stage") != "stage3":
        raise ValueError("v3 formal uncertainty schedule is stage3")
    if training.get("view_balanced") is not True:
        raise ValueError("v3 formal training requires view_balanced = true")


def validate_source_dependencies() -> None:
    """Fail early unless the exact upstream-backed adapter modules import."""
    from cardioresp4d.adapters.film import FiLMAdapter  # noqa: F401
    from cardioresp4d.adapters.nesvor_inr import NeSVoRCanonicalAdapter  # noqa: F401
    from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter  # noqa: F401
    from cardioresp4d.adapters.sinr_mbc import SINRFFDBasis  # noqa: F401


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"source-first configuration requires mapping {key}")
    return nested
