from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from cardioresp4d.training.sampler import DynamicObservation


def _observation(*, view: str = "SAX", slice_id: str = "a", timestamp: float = 0.0, value: float = 1.0, center_z: float = 0.0) -> DynamicObservation:
    return DynamicObservation(
        image=torch.full((1, 4, 5), value), view=view, slice_id=slice_id,
        dynamic_frame_id=0, center_mm=torch.tensor([0.0, 0.0, center_z]),
        row_direction=torch.tensor([1.0, 0.0, 0.0]),
        column_direction=torch.tensor([0.0, 1.0, 0.0]), normal=torch.tensor([0.0, 0.0, 1.0]),
        pixel_spacing_mm=torch.tensor([1.2, 1.3]), slice_thickness_mm=4.0,
        qc_valid=True, qc_reason="valid", timestamp_s=timestamp, normalized_slice_position=center_z,
    )


def test_temporal_mean_groups_deterministically_and_preserves_provenance() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import build_temporal_mean_locations

    frames = [_observation(timestamp=2., value=5.), _observation(timestamp=1., value=1.), _observation(slice_id="b", timestamp=0., value=3.)]
    locations = build_temporal_mean_locations(list(reversed(frames)))
    assert [(item.view, item.slice_id) for item in locations] == [("SAX", "a"), ("SAX", "b")]
    torch.testing.assert_close(locations[0].image, torch.full((1, 4, 5), 3.0))
    assert locations[0].frame_count == 2
    assert locations[0].timestamps_s == (1.0, 2.0)
    assert locations[0].geometry_hash


def test_temporal_mean_rejects_location_geometry_changes() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import build_temporal_mean_locations

    frames = [_observation(), replace(_observation(timestamp=1.), center_mm=torch.tensor([0.0, 0.0, 0.01]))]
    with pytest.raises(ValueError, match="geometry"):
        build_temporal_mean_locations(frames, geometry_tolerance=1e-4)


def test_temporal_mean_export_writes_nifti_stack_and_geometry_manifest(tmp_path) -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import build_temporal_mean_locations, export_temporal_mean_stacks

    locations = build_temporal_mean_locations([_observation(view=view, slice_id="a", value=value) for view, value in (("SAX", 1.), ("2CH", 2.), ("4CH", 3.))])
    manifest = export_temporal_mean_stacks(locations, tmp_path)
    payload = __import__("json").loads(manifest.read_text())
    assert {item["stack_file"] for item in payload["locations"]} == {"mean_stack_SAX.nii.gz", "mean_stack_2CH.nii.gz", "mean_stack_4CH.nii.gz"}
    assert all(item["stack_thickness_mm"] == 4.0 for item in payload["locations"])


def test_formal_and_smoke_nesvor_commands_are_distinct_and_complete() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import nesvor_reconstruct_command

    formal = nesvor_reconstruct_command(["sax.nii.gz", "2ch.nii.gz", "4ch.nii.gz"], output_dir="out", thicknesses_mm=[4.0, 4.0, 4.0], smoke=False)
    smoke = nesvor_reconstruct_command(["sax.nii.gz"], output_dir="out", thicknesses_mm=[4.0], smoke=True)
    assert formal[:2] == ["nesvor", "reconstruct"]
    assert "--registration" in formal and formal[formal.index("--registration") + 1] == "stack"
    assert formal[formal.index("--n-iter") + 1] == "10000"
    assert formal[formal.index("--n-samples") + 1] == "256"
    assert smoke[smoke.index("--n-iter") + 1] != "10000"
    assert "--scanner-space" in formal and "--no-slice-scale" in formal
    assert formal[formal.index("--output-intensity-mean") + 1] == "0"
    assert "--output-slices" in formal and "--output-log" in formal


def test_reference_encoder_has_eight_film_layers_and_gradients() -> None:
    from cardioresp4d.models.s2vdreme_reference_encoder import S2VDREMEReferenceSingleFrameFiLMEncoder

    torch.manual_seed(0)
    encoder = S2VDREMEReferenceSingleFrameFiLMEncoder(block_channels=(4,) * 8, block_strides=(1,) * 8, condition_dim=8)
    image = torch.rand(2, 1, 8, 8)
    position = torch.tensor([[0.0], [0.5]])
    output = encoder(image, normalized_slice_position=position, view_id=torch.tensor([0, 1]))
    assert output["resp_scores"].shape == (2, 3, 3)
    assert len(encoder.inspect_film(image, normalized_slice_position=position, view_id=torch.tensor([0, 1]))) == 8
    changed_position = encoder.inspect_film(image, normalized_slice_position=position + .1, view_id=torch.tensor([0, 1]))
    base = encoder.inspect_film(image, normalized_slice_position=position, view_id=torch.tensor([0, 1]))
    assert not torch.equal(base[0]["gamma"], changed_position[0]["gamma"])
    output["resp_scores"].square().mean().backward()
    assert all(block.weight.grad is not None and torch.isfinite(block.weight.grad).all() for block in encoder.blocks)


def test_reference_config_selects_isolated_eight_layer_encoder() -> None:
    from cardioresp4d.models.s2vdreme_reference_encoder import S2VDREMEReferenceSingleFrameFiLMEncoder
    from cardioresp4d.training.build_model import build_source_first_model

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "respiratory_s2vdreme_reference_v1.yaml").read_text())
    domain = {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}
    model = build_source_first_model(config, domain, n_dynamic_frames=3, device=torch.device("cpu"))
    assert isinstance(model.film_encoder, S2VDREMEReferenceSingleFrameFiLMEncoder)


def test_reference_config_rejects_vnext_auxiliaries_and_joint_ownership() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import configure_reference_segment, reference_schedule, validate_reference_config
    from cardioresp4d.training.build_model import build_source_first_model

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "respiratory_s2vdreme_reference_v1.yaml").read_text())
    validate_reference_config(config)
    domain = {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}
    model = build_source_first_model(config, domain, n_dynamic_frames=3, device=torch.device("cpu"))
    warm, joint = reference_schedule()[:2]
    configure_reference_segment(model, warm)
    assert not any(parameter.requires_grad for parameter in model.canonical.parameters())
    assert not any(parameter.requires_grad for parameter in model.cardiac_mbc.parameters())
    configure_reference_segment(model, joint)
    assert all(parameter.requires_grad for parameter in model.canonical.parameters())


def test_reference_schedule_ownership_loss_contract_and_boundaries() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import (
        REFERENCE_CHECKPOINT_BOUNDARIES, REFERENCE_LOSS_WEIGHTS, reference_schedule,
    )

    schedule = reference_schedule()
    assert [(item.name, item.steps) for item in schedule] == [
        ("L1_warm", 200), ("L1_joint", 600), ("L2_warm", 200), ("L2_joint", 600), ("L3_warm", 200), ("L3_joint", 600),
    ]
    assert schedule[0].owners == frozenset({"respiratory_encoder", "respiratory_mbc"})
    assert schedule[1].owners == frozenset({"canonical", "respiratory_encoder", "respiratory_mbc"})
    assert REFERENCE_LOSS_WEIGHTS == {"image": 2e-6, "mbc_normalization": 1e-2, "smooth_resp": 1e-3}
    assert REFERENCE_CHECKPOINT_BOUNDARIES == tuple(f"checkpoint_{item.name}_end.pt" for item in schedule)


def test_per_location_diagnostics_do_not_count_between_location_dc_as_dynamic() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import reference_score_diagnostics

    values = (torch.ones(3, 1, 3), torch.full((2, 1, 3), 5.0))
    metrics = reference_score_diagnostics(values)
    assert metrics["respiratory_score_dynamic_rms"] == pytest.approx(0.0)
    assert metrics["respiratory_score_total_rms"] ** 2 == pytest.approx(metrics["respiratory_score_mean_rms"] ** 2)


def test_joint_audit_reports_motion_and_parameter_changes_without_verdict() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import joint_stage_audits

    fields = {"reconstruction_mse_full": 1., "reconstruction_mse_SAX": 1., "reconstruction_mse_2CH": 1., "reconstruction_mse_4CH": 1., "respiratory_score_static_rms": 2., "respiratory_score_dynamic_rms": 3., "respiratory_dvf_static_rms_mm": 4., "respiratory_dvf_dynamic_rms_mm": 5., "respiratory_encoder_parameter_norm": 6., "respiratory_mbc_parameter_norm": 7., "canonical_parameter_norm": 8.}
    audits = joint_stage_audits([{"segment": "L1_warm", **fields}, {"segment": "L1_joint", **{key: value + 1 for key, value in fields.items()}}])
    assert len(audits) == 1
    assert audits[0]["delta_reconstruction_mse_full"] == 1.0
    assert audits[0]["delta_respiratory_dvf_dynamic_rms_mm"] == 1.0


def test_static_checkpoint_admission_requires_exact_canonical_state() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import strict_canonical_state_from_static_checkpoint

    expected = {"a": torch.ones(2), "b": torch.zeros(1)}
    admitted = strict_canonical_state_from_static_checkpoint({"model": {"a": torch.ones(2), "b": torch.zeros(1)}}, expected)
    assert set(admitted) == set(expected)
    with pytest.raises(ValueError, match="exact"):
        strict_canonical_state_from_static_checkpoint({"inr_state_dict": {"a": torch.ones(3)}}, expected)


def test_static_state_admission_rejects_shifted_bbox_despite_matching_shapes() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import strict_canonical_state_from_static_checkpoint

    expected = {"bounding_box": torch.tensor([[-1., -1., -1.], [1., 1., 1.]]), "a": torch.ones(2)}
    accepted = strict_canonical_state_from_static_checkpoint({"model": {"bounding_box": expected["bounding_box"].clone(), "a": torch.ones(2)}}, expected, internal_bounds_world_mm=expected["bounding_box"])
    torch.testing.assert_close(accepted["bounding_box"], expected["bounding_box"])
    with pytest.raises(ValueError, match="bounding_box"):
        strict_canonical_state_from_static_checkpoint({"model": {"bounding_box": expected["bounding_box"] + 1., "a": torch.ones(2)}}, expected, internal_bounds_world_mm=expected["bounding_box"])


def test_static_inr_lps_ras_bridge_is_exact_and_differentiable() -> None:
    from cardioresp4d.adapters.nesvor_inr import NeSVoRCanonicalAdapter, lps_to_ras_world_matrix

    bounds = torch.tensor([[-4., -5., -6.], [4., 5., 6.]])
    bridge = NeSVoRCanonicalAdapter(bounds, width=8, depth=1, n_features_z=4, finest_resolution=4., encoding_backend="torch_hash", input_world_to_internal=lps_to_ras_world_matrix())
    bridge.train()
    points_lps = torch.tensor([[1.5, -2.0, 3.0], [-1.0, 2.5, -4.0]], requires_grad=True)
    points_ras = torch.tensor([[-1.5, 2.0, 3.0], [1.0, -2.5, -4.0]])
    expected, _, _ = bridge.inr(points_ras)
    actual = bridge(points_lps)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert points_lps.grad is not None and torch.isfinite(points_lps.grad).all() and points_lps.grad.abs().sum() > 0
    torch.testing.assert_close(bridge.world_to_internal(points_lps.detach()), points_ras)


def test_static_inr_bridge_keeps_external_motion_bounds_separate() -> None:
    from cardioresp4d.adapters.nesvor_inr import lps_to_ras_world_matrix
    from cardioresp4d.training.build_model import build_source_first_model

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "respiratory_s2vdreme_reference_v1.yaml").read_text())
    domain = {"world_min_mm": [-10., -9., -8.], "world_max_mm": [10., 9., 8.], "cardiac_box": {"min_mm": [-2., -2., -2.], "max_mm": [2., 2., 2.]}}
    internal = torch.tensor([[-7., -6., -5.], [7., 6., 5.]])
    model = build_source_first_model(config, domain, n_dynamic_frames=1, device=torch.device("cpu"), canonical_internal_bounds_world_mm=internal, canonical_world_to_internal_transform=lps_to_ras_world_matrix())
    torch.testing.assert_close(model.canonical_lower_world_mm, torch.tensor(domain["world_min_mm"]))
    torch.testing.assert_close(model.canonical_upper_world_mm, torch.tensor(domain["world_max_mm"]))
    torch.testing.assert_close(model.canonical.inr.bounding_box, internal)
    torch.testing.assert_close(model.canonical.world_to_internal(torch.tensor([[1., 2., 3.]])), torch.tensor([[-1., -2., 3.]]))


def test_raw_static_encoding_backend_detection_is_strict() -> None:
    from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend

    assert detect_checkpoint_encoding_backend({"encoding.params": torch.zeros(1)}) == "tinycudann"
    assert detect_checkpoint_encoding_backend({"encoding.box_offsets": torch.zeros(1)}) == "torch_hash"
    with pytest.raises(ValueError, match="ambiguous"):
        detect_checkpoint_encoding_backend({"encoding.params": torch.zeros(1), "encoding.box_offsets": torch.zeros(1)})


def test_static_state_admission_uses_internal_bbox_and_fixed_lps_ras_bridge() -> None:
    from cardioresp4d.adapters.nesvor_inr import lps_to_ras_world_matrix
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import strict_canonical_state_from_static_checkpoint

    internal = torch.tensor([[-5., -4., -3.], [5., 4., 3.]])
    expected = {"bounding_box": internal.clone(), "encoding.box_offsets": torch.zeros(1)}
    admitted = strict_canonical_state_from_static_checkpoint({"model": {key: value.clone() for key, value in expected.items()}}, expected, internal_bounds_world_mm=internal, world_to_internal_transform=lps_to_ras_world_matrix(), expected_encoding_backend="torch_hash")
    torch.testing.assert_close(admitted["bounding_box"], internal)
    with pytest.raises(ValueError, match="coordinate bridge"):
        strict_canonical_state_from_static_checkpoint({"model": {key: value.clone() for key, value in expected.items()}}, expected, internal_bounds_world_mm=internal, world_to_internal_transform=torch.eye(3), expected_encoding_backend="torch_hash")


def test_corrected_pose_map_propagates_per_location_without_changing_frame_content(tmp_path) -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import apply_corrected_static_poses, build_corrected_pose_map

    mean_manifest = tmp_path / "mean.json"
    mean_manifest.write_text(json.dumps({"locations": [
        {"view": "SAX", "slice_id": "a", "stack_file": "mean_stack_SAX.nii.gz", "stack_index": 0, "geometry": {"center_mm": [0., 0., 0.], "row_direction": [1., 0., 0.], "column_direction": [0., 1., 0.], "normal": [0., 0., 1.], "pixel_spacing_mm": [1., 1.], "slice_thickness_mm": 4., "image_shape": [1, 4, 5]}},
        {"view": "SAX", "slice_id": "b", "stack_file": "mean_stack_SAX.nii.gz", "stack_index": 1, "geometry": {"center_mm": [0., 0., 4.], "row_direction": [1., 0., 0.], "column_direction": [0., 1., 0.], "normal": [0., 0., 1.], "pixel_spacing_mm": [1., 1.], "slice_thickness_mm": 4., "image_shape": [1, 4, 5]}},
    ]}))
    slices = tmp_path / "slices"; slices.mkdir()
    for index, x in enumerate((10., 20.)):
        affine = np.eye(4); affine[:3, 3] = [-(x + 2.), 0., 0.]  # RAS origin; slice centre maps to LPS +x.
        nib.save(nib.Nifti1Image(np.zeros((5, 4, 1), dtype=np.float32), affine), slices / f"{index}.nii.gz")
    pose_map = build_corrected_pose_map(mean_manifest, slices)
    frames = [_observation(slice_id="a", timestamp=2., value=3.), _observation(slice_id="a", timestamp=1., value=4.), _observation(slice_id="b", timestamp=0., value=5.)]
    corrected = apply_corrected_static_poses(list(reversed(frames)), pose_map)
    assert [item.slice_id for item in corrected] == ["a", "a", "b"]
    assert [item.timestamp_s for item in corrected] == [1., 2., 0.]
    assert [float(item.center_mm[0]) for item in corrected] == [10., 10., 20.]
    assert [float(item.image.mean()) for item in corrected] == [4., 3., 5.]
    with pytest.raises(ValueError, match="missing"):
        apply_corrected_static_poses(frames, {("SAX", "a"): pose_map[("SAX", "a")]})


def test_diagnostic_rng_is_repeatable_and_does_not_perturb_training_rng() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import diagnostic_rng_context

    torch.manual_seed(99); before = torch.get_rng_state().clone()
    with diagnostic_rng_context(): first = torch.rand(4)
    assert torch.equal(torch.get_rng_state(), before)
    with diagnostic_rng_context(): second = torch.rand(4)
    torch.testing.assert_close(first, second)


def test_acquisition_positions_survive_rotated_corrected_pose_propagation() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import CorrectedStaticPose, apply_corrected_static_poses
    from cardioresp4d.training.sampler import assign_normalized_slice_positions

    raw = assign_normalized_slice_positions([_observation(slice_id="low", center_z=-4., timestamp=0.), _observation(slice_id="low", center_z=-4., timestamp=1.), _observation(slice_id="high", center_z=4.)])
    rotated = {("SAX", "low"): CorrectedStaticPose("SAX", "low", "x", 0, {}, {"center_mm": [0., 0., -4.], "row_direction": [0.98, 0.2, 0.], "column_direction": [-0.2, 0.98, 0.], "normal": [0., 0., 1.], "pixel_spacing_mm": [1.2, 1.3], "slice_thickness_mm": 4., "image_shape": [1, 4, 5]}, [], "x"), ("SAX", "high"): CorrectedStaticPose("SAX", "high", "x", 1, {}, {"center_mm": [0., 0., 4.], "row_direction": [1., 0., 0.], "column_direction": [0., 0.98, 0.2], "normal": [0., -0.2, 0.98], "pixel_spacing_mm": [1.2, 1.3], "slice_thickness_mm": 4., "image_shape": [1, 4, 5]}, [], "x")}
    corrected = apply_corrected_static_poses(list(reversed(raw)), rotated)
    assert [item.normalized_slice_position for item in corrected] == pytest.approx([1., -1., -1.])


def test_corrected_pose_containment_is_fail_closed_and_reports_margins() -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import corrected_pose_containment_audit

    inside = {("SAX", "a"): {"center_mm": [0., 0., 0.], "row_direction": [1., 0., 0.], "column_direction": [0., 1., 0.], "normal": [0., 0., 1.], "pixel_spacing_mm": [1., 1.], "slice_thickness_mm": 4., "image_shape": [1, 4, 5]}}
    report = corrected_pose_containment_audit(inside, lower_world_mm=[-3., -3., -3.], upper_world_mm=[3., 3., 3.])
    assert report["passed"] and report["location_count"] == 1
    outside = {("SAX", "a"): {**inside[("SAX", "a")], "center_mm": [5., 0., 0.]}}
    with pytest.raises(ValueError, match="SAX/a"):
        corrected_pose_containment_audit(outside, lower_world_mm=[-3., -3., -3.], upper_world_mm=[3., 3., 3.])


def test_corrected_pose_internal_inr_containment_uses_lps_ras_corners() -> None:
    from cardioresp4d.adapters.nesvor_inr import lps_to_ras_world_matrix
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import corrected_pose_internal_inr_containment_audit

    pose = {("SAX", "a"): {"center_mm": [2., -1., 0.], "row_direction": [1., 0., 0.], "column_direction": [0., 1., 0.], "normal": [0., 0., 1.], "pixel_spacing_mm": [1., 1.], "slice_thickness_mm": 4., "image_shape": [1, 3, 3]}}
    report = corrected_pose_internal_inr_containment_audit(pose, internal_lower_world_mm=[-4., -4., -1.], internal_upper_world_mm=[1., 3., 1.], world_to_internal_transform=lps_to_ras_world_matrix())
    assert report["inside_location_count"] == 1 and report["outside_location_count"] == 0
    outside = corrected_pose_internal_inr_containment_audit(pose, internal_lower_world_mm=[-1., -1., -1.], internal_upper_world_mm=[1., 1., 1.], world_to_internal_transform=lps_to_ras_world_matrix())
    assert outside["outside_location_count"] == 1
    assert outside["maximum_violation_mm"] == pytest.approx(2.0)
    assert outside["worst_view"] == "SAX" and outside["worst_slice_id"] == "a"


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    script_dir = root / "scripts"
    spec = importlib.util.spec_from_file_location(name, script_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))
    return module


def test_static_validator_requires_gpu_for_tinycudann_and_allows_torch_hash_cpu() -> None:
    validator = _load_script("validate_respiratory_s2vdreme_reference_static")

    assert validator.resolve_validation_device("cpu", "torch_hash", cuda_available=False) == torch.device("cpu")
    with pytest.raises(ValueError, match="tinycudann.*CPU"):
        validator.resolve_validation_device("cpu", "tinycudann", cuda_available=False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        validator.resolve_validation_device("cuda:0", "torch_hash", cuda_available=False)


def test_static_validator_writes_exact_torch_hash_cpu_report(tmp_path, monkeypatch) -> None:
    from cardioresp4d.training.build_model import build_source_first_model

    validator = _load_script("validate_respiratory_s2vdreme_reference_static")
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "respiratory_s2vdreme_reference_v1.yaml"
    config = yaml.safe_load(config_path.read_text())
    domain = {"world_min_mm": [-4., -4., -4.], "world_max_mm": [4., 4., 4.], "cardiac_box": {"min_mm": [-2., -2., -2.], "max_mm": [2., 2., 2.]}}
    domain_path = tmp_path / "domain.json"; domain_path.write_text(json.dumps(domain))
    model_path = tmp_path / "model.pt"
    source = build_source_first_model(config, domain, n_dynamic_frames=1, device=torch.device("cpu"))
    torch.save({"model": source.canonical.inr.state_dict(), "args": source.canonical.args}, model_path)
    output = tmp_path / "validation.json"
    monkeypatch.setattr(sys, "argv", ["validate", "--experiment-config", str(config_path), "--canonical-domain", str(domain_path), "--static-model", str(model_path), "--output", str(output), "--device", "cpu"])
    validator.main()
    report = json.loads(output.read_text())
    assert report["accepted"] and report["encoding_backend"] == "torch_hash" and report["strict_load"]
    assert report["world_to_internal_matrix"] == [[-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def test_formal_preflight_writes_admission_artifacts_without_training(tmp_path, monkeypatch) -> None:
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import CorrectedStaticPose
    from cardioresp4d.training.build_model import build_source_first_model

    launcher = _load_script("train_respiratory_s2vdreme_reference_v1")
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "respiratory_s2vdreme_reference_v1.yaml"
    config = yaml.safe_load(config_path.read_text())
    domain = {"world_min_mm": [-4., -4., -4.], "world_max_mm": [4., 4., 4.], "cardiac_box": {"min_mm": [-2., -2., -2.], "max_mm": [2., 2., 2.]}}
    domain_path = tmp_path / "domain.json"; domain_path.write_text(json.dumps(domain))
    static_model = tmp_path / "model.pt"
    source = build_source_first_model(config, domain, n_dynamic_frames=1, device=torch.device("cpu"))
    torch.save({"model": source.canonical.inr.state_dict(), "args": source.canonical.args}, static_model)
    manifest_path = tmp_path / "manifest.csv"; manifest_path.write_text("manifest\n")
    qc_path = tmp_path / "qc.csv"; qc_path.write_text("qc\n")
    volume_path = tmp_path / "static.nii.gz"; volume_path.write_bytes(b"volume")
    poses_path = tmp_path / "poses.json"; poses_path.write_text("{}")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    static_manifest = tmp_path / "static_manifest.json"
    static_manifest.write_text(json.dumps({"source_manifest_sha256": digest(manifest_path), "qc_table_sha256": digest(qc_path), "scientific_contract": {"output_intensity_mean": 0}, "outputs": {"model": {"path": str(static_model), "sha256": digest(static_model)}, "volume": {"path": str(volume_path), "sha256": digest(volume_path)}}, "corrected_pose_manifest": {"path": str(poses_path), "sha256": digest(poses_path)}}))
    geometry = {"center_mm": [0., 0., 0.], "row_direction": [1., 0., 0.], "column_direction": [0., 1., 0.], "normal": [0., 0., 1.], "pixel_spacing_mm": [1., 1.], "slice_thickness_mm": 4., "image_shape": [1, 4, 5]}
    pose_map = {(view, "a"): CorrectedStaticPose(view, "a", "stack", 0, geometry, geometry, [], "slice.nii.gz") for view in ("SAX", "2CH", "4CH")}
    monkeypatch.setattr(launcher, "verify_vendored_source_lock", lambda _: {"verified": True})
    monkeypatch.setattr(launcher, "observations_from_manifest", lambda *args, **kwargs: ([_observation(view=view) for view in ("SAX", "2CH", "4CH")], {}, []))
    monkeypatch.setattr(launcher, "load_corrected_pose_map", lambda _: pose_map)
    monkeypatch.setattr(launcher, "static_reference_diagnostics", lambda *_: {"reconstruction_mse_full": 0.0})
    monkeypatch.setattr(launcher.S2VDREMEReferenceRunner, "run_segment", lambda *_: pytest.fail("preflight must not run a training segment"))
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--experiment-config", str(config_path), "--manifest", str(manifest_path), "--qc-table", str(qc_path), "--canonical-domain", str(domain_path), "--static-manifest", str(static_manifest), "--output-dir", str(output), "--device", "cpu", "--preflight-only"])
    launcher.main()
    report = json.loads((output / "preflight_report.json").read_text())
    assert report["status"] == "PASS" and report["formal_training_started"] is False
    assert (output / "static_reference_ready.json").is_file()
    assert not list(output.glob("checkpoint_*.pt"))


def test_formal_and_smoke_fallback_policies_are_explicit_and_distinct() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    formal = yaml.safe_load((root / "configs" / "respiratory_s2vdreme_reference_v1.yaml").read_text())
    smoke = yaml.safe_load((root / "configs" / "respiratory_s2vdreme_reference_v1_smoke.yaml").read_text())
    assert formal["static_reference"]["canonical_initialization"]["fallback"]["mode"] == "calibration_required"
    assert smoke["static_reference"]["canonical_initialization"]["fallback"]["mode"] == "fixed_steps"
    assert smoke["static_reference"]["canonical_initialization"]["fallback"]["steps"] == 2
