from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import yaml
import pytest

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _respiratory_only_motion
from experiments.c4_diagnostics.exp2c4f_multilocation_constant_geometry_probe import _geometry_audit, _select_reference, _with_reference_geometry, run
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _observations():
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=1, frames_per_location=3, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return motion, _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def test_reference_selection_and_formal_missing_reference_failure() -> None:
    _, observations = _observations()
    reference = _select_reference(observations, view="2CH", slice_id="missing", smoke=True)
    assert reference.view == "2CH"
    with pytest.raises(ValueError, match="reference location"):
        _select_reference(observations, view="2CH", slice_id="missing", smoke=False)


def test_clamp_preserves_non_geometry_metadata_and_does_not_mutate_original() -> None:
    _, observations = _observations(); reference = _select_reference(observations, view="2CH", slice_id=None, smoke=True); original = next(item for item in observations if item.view == "SAX")
    clamped = _with_reference_geometry(original, reference)
    assert clamped.image is original.image and clamped.timestamp_s == original.timestamp_s and clamped.view == original.view and clamped.slice_id == original.slice_id
    torch.testing.assert_close(clamped.center_mm, reference.center_mm)
    assert not torch.equal(original.center_mm, reference.center_mm)


def test_geometry_audit_reports_original_variation_and_clamped_constancy() -> None:
    _, observations = _observations(); reference = _select_reference(observations, view="2CH", slice_id=None, smoke=True)
    audit = _geometry_audit(observations, [_with_reference_geometry(item, reference) for item in observations], reference)
    assert audit["original_unique_location_count"] == 3
    assert audit["center_mm_original_varies"]
    assert audit["center_mm_clamped_constant"]


def test_cpu_smoke_writes_outputs_and_keeps_renderer_and_mbc_unused(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "c4f", device="cpu", overwrite=False, seed=0, reference_view="2CH", reference_slice_id="missing", locations_per_view=1, frames_per_location=3, steps=2, observations_per_update=3, smoke=True)
    summary = run(args)
    assert summary["renderer_calls"] == 0 and summary["learned_respiratory_mbc_calls"] == 0
    assert all(summary["frozen_module_audit"].values()) and summary["geometry_audit"]["clamped_geometry_constant"]
    for name in ("summary.json", "metrics_initial.json", "metrics_final.json", "training_curve.csv", "recovery_arrays.npz"):
        assert (args.output_dir / name).is_file()
