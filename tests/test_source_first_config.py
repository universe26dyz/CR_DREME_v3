"""Hard rejections that prevent a v3 config from silently selecting legacy code."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.training.source_first_config import validate_source_first_config  # noqa: E402


def _valid() -> dict:
    return {"model": {"canonical": {"implementation": "nesvor_official", "upstream_commit": "2e96a91bdd30174210caea911e03a2778c65adbe"}, "psf": {}, "film": {}, "respiratory_mbc": {"implementation": "sinr_official", "upstream_commit": "1a524ca7ae453b55310595fe957245088a108233"}, "cardiac_mbc": {"implementation": "sinr_official"}, "uncertainty": {"enable_stage": "stage3c"}}, "training": {"view_balanced": True, "fixed_location_balanced": True, "seed": 0, "stage3a": {"cardiac_warmup": True, "uncertainty_enabled": False}, "stage3b": {"joint_cardiorespiratory": True, "uncertainty_enabled": False}, "stage3c": {"uncertainty_enabled": True}, "motion_regularization": {"respiratory_evaluation_grid": {}, "cardiac_evaluation_grid": {}}}, "domain": {"reconstruction_domain": "full_acquisition_supported", "cardiac_box_is_crop": False}}


class SourceFirstConfigTest(unittest.TestCase):
    def test_rejects_legacy_canonical_cardiac_only_and_unbalanced_configs(self) -> None:
        for section, key, value in (("canonical", "implementation", "custom_hash"), ("respiratory_mbc", "implementation", "custom_sinr")):
            config = _valid(); config["model"][section][key] = value
            with self.assertRaises(ValueError): validate_source_first_config(config, ROOT)
        config = _valid(); config["domain"]["reconstruction_domain"] = "cardiac_box"
        with self.assertRaises(ValueError): validate_source_first_config(config, ROOT)
        config = _valid(); config["training"]["view_balanced"] = False
        with self.assertRaises(ValueError): validate_source_first_config(config, ROOT)

    def test_accepts_the_pinned_source_contract(self) -> None:
        validate_source_first_config(_valid(), ROOT)


if __name__ == "__main__":
    unittest.main()
