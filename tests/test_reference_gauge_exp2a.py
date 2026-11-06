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
    dynamic = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True); reference = dynamic.detach().clone().requires_grad_()
    raw = {"resp_scores": encoder(dynamic)}; ref = {"resp_scores": encoder(reference)}
    effective = effective_reference_scores(raw, ref)["resp_scores"]
    torch.testing.assert_close(effective, torch.zeros_like(effective))
    offset = torch.full_like(raw["resp_scores"], 7.)
    torch.testing.assert_close(effective_reference_scores({"resp_scores": raw["resp_scores"] + offset}, {"resp_scores": ref["resp_scores"] + offset})["resp_scores"], effective)
    effective.sum().backward()
    assert encoder.weight.grad is not None and torch.isfinite(encoder.weight.grad).all()


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
