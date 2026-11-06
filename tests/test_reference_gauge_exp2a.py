from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


def _observation(view: str = "SAX", slice_id: str = "s0", shape: tuple[int, int] = (2, 3)):
    from cardioresp4d.training.sampler import DynamicObservation

    return DynamicObservation(
        image=torch.arange(np.prod(shape), dtype=torch.float32).reshape(1, *shape), view=view,
        slice_id=slice_id, dynamic_frame_id=0, center_mm=torch.zeros(3),
        row_direction=torch.tensor([1., 0., 0.]), column_direction=torch.tensor([0., 1., 0.]),
        normal=torch.tensor([0., 0., 1.]), pixel_spacing_mm=torch.ones(2), slice_thickness_mm=1.,
        qc_valid=True, qc_reason="", timestamp_s=0., normalized_slice_position=0.,
    )


def test_reference_mapping_recovers_authoritative_transposed_mean_and_hashes(tmp_path: Path) -> None:
    from cardioresp4d.visualization.reference_gauge import ReferenceImageProvider

    stack = np.zeros((3, 2, 2), dtype=np.float32); stack[..., 1] = np.arange(6, dtype=np.float32).reshape(3, 2)
    stack_path = tmp_path / "mean_stack_SAX.nii.gz"; nib.save(nib.Nifti1Image(stack, np.eye(4)), str(stack_path))
    manifest = tmp_path / "mean_slice_manifest.json"; manifest.write_text(json.dumps({"locations": [{"view": "SAX", "slice_id": "s0", "stack_file": stack_path.name, "stack_index": 1, "image_shape": [1, 2, 3]}]}))
    provider = ReferenceImageProvider.from_mean_artifacts(manifest, observations=[_observation()])
    expected = torch.from_numpy(stack[..., 1].T.copy())[None]
    torch.testing.assert_close(provider(_observation()), expected)
    assert provider.provenance["mean_manifest_sha256"]
    assert provider.provenance["mean_stacks"][stack_path.name]


def test_reference_mapping_never_maps_placeholder_and_hard_fails_missing_or_shape(tmp_path: Path) -> None:
    from cardioresp4d.visualization.reference_gauge import ReferenceImageProvider

    stack = np.zeros((3, 2, 2), dtype=np.float32); stack[..., 0] = 9.; stack_path = tmp_path / "mean_stack_SAX.nii.gz"; nib.save(nib.Nifti1Image(stack, np.eye(4)), str(stack_path))
    manifest = tmp_path / "mean_slice_manifest.json"; manifest.write_text(json.dumps({"locations": [{"view": "SAX", "slice_id": "real", "stack_file": stack_path.name, "stack_index": 1, "image_shape": [1, 2, 3]}]}))
    provider = ReferenceImageProvider.from_mean_artifacts(manifest)
    with pytest.raises(KeyError, match="missing authoritative reference"):
        provider(_observation(slice_id="placeholder"))
    with pytest.raises(ValueError, match="shape mismatch"):
        ReferenceImageProvider.from_mean_artifacts(manifest, observations=[_observation(slice_id="real", shape=(4, 4))])


def test_effective_reference_score_cancels_equal_images_and_shared_dc_with_gradients() -> None:
    from cardioresp4d.models.reference_gauge import effective_reference_scores

    encoder = torch.nn.Linear(2, 3, bias=True)
    dynamic = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True); reference = torch.tensor([[0.5, 1.5], [2.5, 3.5]], requires_grad=True)
    raw = {"resp_scores": encoder(dynamic)}; ref = {"resp_scores": encoder(reference)}
    effective = effective_reference_scores(raw, ref)["resp_scores"]
    equal = effective_reference_scores({"resp_scores": encoder(dynamic)}, {"resp_scores": encoder(dynamic)})["resp_scores"]
    torch.testing.assert_close(equal, torch.zeros_like(equal))
    offset = torch.full_like(raw["resp_scores"], 7.)
    torch.testing.assert_close(effective_reference_scores({"resp_scores": raw["resp_scores"] + offset}, {"resp_scores": ref["resp_scores"] + offset})["resp_scores"], effective)
    dynamic_grad = torch.autograd.grad(raw["resp_scores"].sum(), encoder.parameters(), retain_graph=True)
    reference_grad = torch.autograd.grad(ref["resp_scores"].sum(), encoder.parameters(), retain_graph=True)
    effective_grad = torch.autograd.grad(effective.sum(), encoder.parameters())
    for actual, expected in zip(effective_grad, (a - b for a, b in zip(dynamic_grad, reference_grad))):
        torch.testing.assert_close(actual, expected)


def test_reference_gauge_recomputes_reference_branch_and_is_not_trainable() -> None:
    from cardioresp4d.models.reference_gauge import ReferenceScoreGauge

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
        def __call__(self, image): return {"resp_scores": image.mean(dim=(-1, -2, -3), keepdim=False).reshape(image.shape[0], 1, 1) * self.weight}
    encoder = Encoder(); reference = torch.ones((1, 1, 2, 2)); gauge = ReferenceScoreGauge(lambda _: reference)
    observation = _observation(); raw = encoder(observation.image); ref = encoder(gauge.reference_image(observation)); first = gauge(raw, ref)["resp_scores"].clone()
    with torch.no_grad(): encoder.weight.mul_(2.)
    second = gauge(encoder(observation.image), encoder(gauge.reference_image(observation)))["resp_scores"]
    assert not torch.equal(first, second)
    assert all(not isinstance(value, torch.nn.Parameter) for value in gauge.__dict__.values())


def test_exp2a_protocol_owns_only_encoder_and_l1_mbc() -> None:
    from cardioresp4d.experiments.respiratory_reference_gauge_exp2a import EXP2A_PROTOCOL, exp2a_trainable_parameter_groups

    assert EXP2A_PROTOCOL["observations_per_update"] == 32 and EXP2A_PROTOCOL["steps"] == 200 and EXP2A_PROTOCOL["active_respiratory_levels"] == 1
    names = exp2a_trainable_parameter_groups({"canonical": ["c"], "film": ["f"], "respiratory_mbc": ["l1", "l2", "l3"]})
    assert names == {"film": ["f"], "respiratory_mbc": ["l1"]}


def test_exp1_default_score_path_is_raw_and_no_reference_branch_is_called() -> None:
    from cardioresp4d.training.model import SourceFirstDynamicModel

    model = object.__new__(SourceFirstDynamicModel)
    model.reference_score_provider = None
    calls = {"raw": 0, "reference": 0}
    def raw(_):
        calls["raw"] += 1
        return {"resp_scores": torch.tensor([[[1., 2., 3.]]])}
    def reference(_):
        calls["reference"] += 1
        return {"resp_scores": torch.tensor([[[9., 9., 9.]]])}
    model.encode_motion_scores_raw = raw
    model.encode_reference_scores = reference
    result = SourceFirstDynamicModel.encode_motion_scores(model, object())
    torch.testing.assert_close(result["resp_scores"], torch.tensor([[[1., 2., 3.]]]))
    assert calls == {"raw": 1, "reference": 0}


def test_exp1_provider_free_predict_and_regularizers_match_raw_branch() -> None:
    from cardioresp4d.training.model import SourceFirstDynamicModel

    model = SourceFirstDynamicModel(torch.tensor([-2., -2., -2.]), torch.tensor([2., 2., 2.]), cardiac_lower_world_mm=torch.tensor([-1., -1., -1.]), cardiac_upper_world_mm=torch.tensor([1., 1., 1.]), n_dynamic_frames=1, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
    observation = _observation(shape=(3, 3))
    raw = model.encode_motion_scores_raw(observation); effective = model.encode_motion_scores(observation)
    for left, right in zip(raw["resp_scores"], effective["resp_scores"]):
        torch.testing.assert_close(left, right)
    model.respiratory_mbc.set_active_levels(1)
    raw_regularizers = model.motion_regularizers(raw, "stage2a")
    effective_regularizers = model.motion_regularizers(effective, "stage2a")
    for key in raw_regularizers:
        torch.testing.assert_close(raw_regularizers[key], effective_regularizers[key])
    pixels = torch.tensor([[0., 0.], [1., 1.]], dtype=torch.float32)
    torch.manual_seed(17); raw_prediction = model.predict(observation, pixels, "stage2a")
    torch.manual_seed(17); effective_prediction = model.predict(observation, pixels, "stage2a")
    torch.testing.assert_close(raw_prediction["predicted_intensity"], effective_prediction["predicted_intensity"])


def test_reference_branch_outputs_are_explicit_and_not_aliased() -> None:
    from cardioresp4d.models.reference_gauge import effective_reference_scores

    raw = {"resp_scores": torch.ones(1, 1, 3)}; reference = {"resp_scores": torch.full((1, 1, 3), 0.25)}
    result = effective_reference_scores(raw, reference)
    assert result["resp_scores"] is not raw["resp_scores"]
    assert result["raw_resp_scores"] is raw["resp_scores"]
    assert result["reference_resp_scores"] is reference["resp_scores"]


def test_comparison_consumes_authoritative_exp1_boundary_metrics(tmp_path: Path) -> None:
    from scripts.compare_exp1_exp2a_reference_gauge import build_comparison_payload

    gauge = {"pooled": {"L1_warm": {"all_locations": {"raw_score": {"static_rms": 2., "dynamic_rms": 1., "static_dynamic_ratio": 2.}, "effective_score": {"static_rms": 0., "dynamic_rms": 1., "static_dynamic_ratio": 0.}, "raw_dvf": {"static_rms_mm": 3., "dynamic_rms_mm": 1., "static_dynamic_ratio": 3.}, "gauge_dvf": {"static_rms_mm": 0., "dynamic_rms_mm": 1., "static_dynamic_ratio": 0.}}}}}
    report = {"diagnostics": {"step_200": {}}, "runner_report": {"reconstruction_mse_full": 0.2}}
    payload = build_comparison_payload(gauge, report, [{"segment": "L1_warm", "reconstruction_mse_full": 0.1}])
    assert payload["comparison"]["exp1_reconstruction_mse_full"] == 0.1
    assert payload["comparison"]["primary_comparison"] == "Exp1 checkpoint_L1_warm_end.pt vs Exp2-A checkpoint_step_200.pt"


def test_exp2a_runner_state_isolated_and_rejects_exp1_schedule() -> None:
    from cardioresp4d.experiments.respiratory_reference_gauge_exp2a import Exp2AReferenceRunner

    state = {"experiment": "respiratory_reference_gauge_exp2a", "segment": "exp2a_step_200", "steps": 200, "stage": "stage2a", "active_levels": 1, "owners": ["respiratory_encoder", "respiratory_mbc"], "canonical_frozen": True}
    assert state["experiment"] == "respiratory_reference_gauge_exp2a" and state["owners"][-1] == "respiratory_mbc"
    assert Exp2AReferenceRunner.__name__ == "Exp2AReferenceRunner"


def test_retrospective_score_and_dvf_location_metrics_are_json_serializable() -> None:
    from scripts.audit_exp1_reference_score_gauge import dvf_sequence_metrics, score_sequence_metrics

    scores = torch.tensor([[1., 2., 3.], [2., 4., 6.]])
    dvf = torch.randn(2, 7, 3)
    score_payload = score_sequence_metrics(scores); dvf_payload = dvf_sequence_metrics(dvf)
    json.dumps({"score": score_payload, "dvf": dvf_payload})
    assert len(score_payload["temporal_mean_components"]) == 3
    assert isinstance(dvf_payload["magnitude_max_mm"], float)


def test_exp2a_input_bindings_fail_closed_when_mean_stack_changes(tmp_path: Path) -> None:
    from cardioresp4d.experiments.respiratory_reference_gauge_exp2a import exp2a_input_bindings, verify_exp2a_input_bindings

    stack = tmp_path / "mean_stack_SAX.nii.gz"; nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.float32), np.eye(4)), str(stack))
    mean = tmp_path / "mean.json"; mean.write_text(json.dumps({"locations": [{"view": "SAX", "slice_id": "s0", "stack_file": stack.name, "stack_index": 0}]}))
    files = [tmp_path / name for name in ("manifest", "qc", "domain", "static.json", "model.pt", "pose.json")]
    for path in files: path.write_bytes(path.name.encode())
    bindings = exp2a_input_bindings(manifest=files[0], qc_table=files[1], canonical_domain=files[2], static_manifest=files[3], mean_manifest=mean, static_model=files[4], corrected_pose_manifest=files[5])
    verify_exp2a_input_bindings(bindings)
    stack.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mean_stacks"):
        verify_exp2a_input_bindings(bindings)
