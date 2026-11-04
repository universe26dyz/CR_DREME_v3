from __future__ import annotations

from dataclasses import replace
import json

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
    accepted = strict_canonical_state_from_static_checkpoint({"model": {"bounding_box": expected["bounding_box"].clone(), "a": torch.ones(2)}}, expected, dynamic_bounds_world_mm=expected["bounding_box"])
    torch.testing.assert_close(accepted["bounding_box"], expected["bounding_box"])
    with pytest.raises(ValueError, match="bounding_box"):
        strict_canonical_state_from_static_checkpoint({"model": {"bounding_box": expected["bounding_box"] + 1., "a": torch.ones(2)}}, expected, dynamic_bounds_world_mm=expected["bounding_box"])


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
