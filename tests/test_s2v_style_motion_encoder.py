from __future__ import annotations

from dataclasses import replace
import pytest
import torch
import yaml

from cardioresp4d.models.s2v_style_motion_encoder import S2VStyleSingleFrameFiLMEncoder
from cardioresp4d.training.sampler import DynamicObservation, assign_normalized_slice_positions, view_id_for
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.models.film_motion_encoder import GeometryFiLMMotionEncoder
from cardioresp4d.adapters.film import FiLMAdapter
from experiments.c4_diagnostics.exp2_checkpoint_common import reconstruct_motion_only_model
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, PhantomCanonicalAdapter


def _observation(*, view: str, slice_id: str, center_z: float, normal: tuple[float, float, float] = (0.0, 0.0, 1.0), timestamp_s: float = 0.0) -> DynamicObservation:
    return DynamicObservation(
        image=torch.ones(1, 8, 8), view=view, slice_id=slice_id, dynamic_frame_id=0,
        center_mm=torch.tensor([0.0, 0.0, center_z]), row_direction=torch.tensor([1.0, 0.0, 0.0]),
        column_direction=torch.tensor([0.0, 1.0, 0.0]), normal=torch.tensor(normal),
        pixel_spacing_mm=torch.ones(2), slice_thickness_mm=4.0, qc_valid=True,
        qc_reason="valid", timestamp_s=timestamp_s,
    )


def test_stack_relative_positions_are_physical_constant_and_ordered() -> None:
    observations = [
        _observation(view="SAX", slice_id="low", center_z=-6.0, timestamp_s=0.0),
        _observation(view="SAX", slice_id="low", center_z=-6.0, timestamp_s=1.0),
        _observation(view="SAX", slice_id="mid", center_z=0.0),
        _observation(view="SAX", slice_id="high", center_z=6.0),
    ]

    assigned = assign_normalized_slice_positions(observations)

    assert [item.normalized_slice_position for item in assigned] == pytest.approx([-1.0, -1.0, 0.0, 1.0])


def test_stack_position_sign_aligns_normals_and_rejects_inconsistent_planes() -> None:
    assigned = assign_normalized_slice_positions([
        _observation(view="2CH", slice_id="low", center_z=-2.0),
        _observation(view="2CH", slice_id="high", center_z=2.0, normal=(0.0, 0.0, -1.0)),
    ])
    assert [item.normalized_slice_position for item in assigned] == pytest.approx([-1.0, 1.0])
    with pytest.raises(ValueError, match="consistent"):
        assign_normalized_slice_positions([
            _observation(view="4CH", slice_id="a", center_z=0.0),
            _observation(view="4CH", slice_id="b", center_z=1.0, normal=(1.0, 0.0, 0.0)),
        ])


def test_single_location_stack_and_view_mapping() -> None:
    assigned = assign_normalized_slice_positions([_observation(view="SAX", slice_id="only", center_z=3.0)])
    assert assigned[0].normalized_slice_position == pytest.approx(0.0)
    assert {view: view_id_for(view) for view in ("SAX", "2CH", "4CH")} == {"SAX": 0, "2CH": 1, "4CH": 2}


def test_s2v_encoder_uses_only_image_view_and_position() -> None:
    torch.manual_seed(0)
    encoder = S2VStyleSingleFrameFiLMEncoder(position_bands=4, view_embedding_dim=8, condition_dim=16, block_channels=(4, 8, 8, 8))
    image = torch.rand(2, 1, 8, 8)
    output = encoder(image, normalized_slice_position=torch.tensor([[0.0], [1.0]]), view_id=torch.tensor([0, 2]))
    assert output["resp_scores"].shape == (2, 3, 3)
    assert output["card_scores"].shape == (2, 1, 3)
    assert encoder.position_feature_dim == 9
    assert all(torch.isfinite(value).all() for value in output.values())
    assert len(encoder.film_heads) == 4
    assert [head.out_features for head in encoder.film_heads] == [8, 16, 16, 16]
    assert isinstance(encoder.film, FiLMAdapter)


def test_baseline_synthetic_observations_skip_s2v_position_validation() -> None:
    domain = {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}
    frames = [
        GeometryFrame("SAX", "a", torch.tensor([0.0, 0.0, -1.0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.ones(2), 2.0, 0.0, 4, 4),
        GeometryFrame("SAX", "b", torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0]), torch.ones(2), 2.0, 1.0, 4, 4),
    ]
    observations = _build_observations(frames, image_shape=(4, 4), phantom=AnalyticCanonicalPhantom(domain), motion=GroundTruthMotion.from_domain(domain), psf_samples=1, seed=0, assign_slice_positions=False)
    assert all(item.normalized_slice_position is None for item in observations)


def test_factory_preserves_baseline_encoder_and_selects_s2v_config() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    domain = {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}
    baseline = build_source_first_model(yaml.safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text()), domain, n_dynamic_frames=3, device=torch.device("cpu"))
    s2v = build_source_first_model(yaml.safe_load((root / "configs/source_first_change4_s2vstyle.yaml").read_text()), domain, n_dynamic_frames=3, device=torch.device("cpu"))
    assert isinstance(baseline.film_encoder, GeometryFiLMMotionEncoder)
    assert isinstance(s2v.film_encoder, S2VStyleSingleFrameFiLMEncoder)
    baseline_clone = build_source_first_model(yaml.safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text()), domain, n_dynamic_frames=3, device=torch.device("cpu"))
    baseline_clone.load_state_dict(baseline.state_dict(), strict=True)


def test_s2v_dispatch_requires_list_assigned_position_and_ignores_timestamp() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    domain = {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}
    model = build_source_first_model(yaml.safe_load((root / "configs/source_first_change4_s2vstyle.yaml").read_text()), domain, n_dynamic_frames=3, device=torch.device("cpu")).eval()
    observation = _observation(view="SAX", slice_id="one", center_z=0.0)
    with pytest.raises(ValueError, match="normalized_slice_position"):
        model.encode_motion_scores(observation)
    assigned = assign_normalized_slice_positions([observation])[0]
    with torch.no_grad():
        first = model.encode_motion_scores(assigned)
        second = model.encode_motion_scores(replace(assigned, timestamp_s=123.0))
    for name in first:
        torch.testing.assert_close(first[name], second[name])


def test_s2v_motion_only_checkpoint_reconstructs_with_strict_state_loading() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/source_first_change4_s2vstyle.yaml").read_text())
    domain = {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}
    source = build_source_first_model(config, domain, n_dynamic_frames=3, device=torch.device("cpu"))
    source.canonical = PhantomCanonicalAdapter(AnalyticCanonicalPhantom(domain))
    restored = reconstruct_motion_only_model(config, domain, source.state_dict(), n_dynamic_frames=3)
    assert isinstance(restored.film_encoder, S2VStyleSingleFrameFiLMEncoder)
