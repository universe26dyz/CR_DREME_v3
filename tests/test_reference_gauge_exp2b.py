from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_gain_algebra_preserves_unscaled_outputs_and_encoder_gradient() -> None:
    from cardioresp4d.models.reference_gauge import effective_reference_scores

    weight = torch.nn.Parameter(torch.tensor(2.0))
    raw = {"resp_scores": weight * torch.tensor([[[2.0, 3.0, 4.0]]])}
    reference = {"resp_scores": weight * torch.tensor([[[1.0, 1.0, 1.0]]])}
    result = effective_reference_scores(raw, reference, score_gain=10.0)
    torch.testing.assert_close(result["unscaled_effective_resp_scores"], torch.tensor([[[2.0, 4.0, 6.0]]]))
    torch.testing.assert_close(result["used_resp_scores"], torch.tensor([[[20.0, 40.0, 60.0]]]))
    result["used_resp_scores"].sum().backward()
    assert weight.grad is not None and weight.grad.item() == pytest.approx(60.0)


def test_gain_is_a_non_trainable_model_policy_and_default_gauge_is_unchanged() -> None:
    from cardioresp4d.models.reference_gauge import effective_reference_scores
    from cardioresp4d.training.model import SourceFirstDynamicModel

    raw = {"resp_scores": torch.tensor([[[1.0, 2.0, 3.0]]])}; reference = {"resp_scores": torch.tensor([[[0.25, 0.5, 0.75]]])}
    default = effective_reference_scores(raw, reference)
    torch.testing.assert_close(default["resp_scores"], raw["resp_scores"] - reference["resp_scores"])
    model = object.__new__(SourceFirstDynamicModel)
    model.set_reference_score_gain(10.0)
    assert model.reference_score_gain == 10.0 and not isinstance(model.reference_score_gain, torch.nn.Parameter)
    with pytest.raises(ValueError, match="positive"):
        model.set_reference_score_gain(0.0)


def test_exp2b_protocol_config_and_gain_are_explicit() -> None:
    from scripts.train_respiratory_reference_gauge_exp2b import _load_config
    from cardioresp4d.experiments.respiratory_reference_gauge_exp2b import exp2b_protocol, validate_exp2b_protocol, validate_exp2b_score_gain

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = _load_config(root / "configs" / "respiratory_reference_gauge_exp2b.yaml")
    validate_exp2b_protocol(config)
    assert exp2b_protocol(10.0)["score_gain"] == 10.0
    assert validate_exp2b_score_gain(100.0) == 100.0
    with pytest.raises(ValueError, match="one of"):
        validate_exp2b_score_gain(1.0)


@pytest.mark.parametrize("gain", [10.0, 100.0])
def test_exp2b_diagnostics_distinguish_unscaled_and_gain_applied_score_and_dvf(monkeypatch, gain: float) -> None:
    import cardioresp4d.experiments.respiratory_reference_gauge_exp2b as exp2b

    class FakeMBC:
        def raw_active(self, points, *, active_levels):
            assert points.shape == (1, 8, 3)
            return torch.ones(1, active_levels, points.shape[1], 3)

    frame0 = SimpleNamespace(timestamp_s=0.0, value=0.0)
    frame1 = SimpleNamespace(timestamp_s=1.0, value=1.0)
    sampler = SimpleNamespace(_grouped={"SAX": {"s0": [frame0, frame1]}})

    class FakeModel:
        reference_score_gain = gain
        canonical_lower_world_mm = torch.tensor([-1.0, -1.0, -1.0])
        canonical_upper_world_mm = torch.tensor([1.0, 1.0, 1.0])
        respiratory_mbc = FakeMBC()

        @staticmethod
        def _physical_grid(lower, upper, shape):
            return torch.zeros(1, 8, 3), torch.ones(3)

        @staticmethod
        def encode_motion_scores_raw(observation):
            value = observation.value
            return {"resp_scores": torch.tensor([[[1.0 + value, 2.0 + 2.0 * value, 3.0 + 3.0 * value]]])}

        @staticmethod
        def encode_reference_scores(observation):
            return {"resp_scores": torch.tensor([[[1.0, 1.0, 1.0]]])}

    monkeypatch.setattr(exp2b, "boundary_motion_diagnostics", lambda *_args, **_kwargs: {"reconstruction_mse_full": 0.1})
    result = exp2b.exp2b_gauge_diagnostics(FakeModel(), sampler, score_gain=gain)
    assert result["score_gain"] == gain
    assert result["used_score_dynamic_rms"] == pytest.approx(gain * result["unscaled_effective_score_dynamic_rms"])
    assert result["used_dvf_dynamic_rms_mm"] == pytest.approx(gain * result["unscaled_effective_dvf_dynamic_rms_mm"])
    assert result["score_gain_algebra_max_abs_error"] == 0.0
    assert result["dvf_gain_algebra_max_abs_error"] == pytest.approx(0.0)


def test_exp2b_diagnostics_ignore_inactive_l2_l3_scores(monkeypatch) -> None:
    import cardioresp4d.experiments.respiratory_reference_gauge_exp2b as exp2b

    class FakeMBC:
        def raw_active(self, points, *, active_levels):
            return torch.ones(1, active_levels, points.shape[1], 3)

    frame0 = SimpleNamespace(timestamp_s=0.0, value=0.0)
    frame1 = SimpleNamespace(timestamp_s=1.0, value=1.0)
    sampler = SimpleNamespace(_grouped={"SAX": {"s0": [frame0, frame1]}})

    def make_model(inactive_scale: float):
        class FakeModel:
            reference_score_gain = 10.0
            canonical_lower_world_mm = torch.tensor([-1.0, -1.0, -1.0])
            canonical_upper_world_mm = torch.tensor([1.0, 1.0, 1.0])
            respiratory_mbc = FakeMBC()

            @staticmethod
            def _physical_grid(lower, upper, shape):
                return torch.zeros(1, 8, 3), torch.ones(3)

            @staticmethod
            def encode_motion_scores_raw(observation):
                value = observation.value
                return {"resp_scores": torch.tensor([[[1.0 + value, 2.0 + value, 3.0 + value], [100.0 * inactive_scale, 200.0 * inactive_scale, 300.0 * inactive_scale], [400.0 * inactive_scale, 500.0 * inactive_scale, 600.0 * inactive_scale]]])}

            @staticmethod
            def encode_reference_scores(observation):
                return {"resp_scores": torch.zeros(1, 3, 3)}

        return FakeModel()

    monkeypatch.setattr(exp2b, "boundary_motion_diagnostics", lambda *_args, **_kwargs: {})
    baseline = exp2b.exp2b_gauge_diagnostics(make_model(1.0), sampler, score_gain=10.0)
    changed_inactive = exp2b.exp2b_gauge_diagnostics(make_model(1000.0), sampler, score_gain=10.0)
    for key in ("raw_score_dynamic_rms", "reference_score_dynamic_rms", "unscaled_effective_score_dynamic_rms", "used_score_dynamic_rms", "unscaled_effective_dvf_dynamic_rms_mm", "used_dvf_dynamic_rms_mm"):
        assert changed_inactive[key] == pytest.approx(baseline[key])


def test_exp2b_runner_rejects_resume_gain_mismatch() -> None:
    from cardioresp4d.experiments.respiratory_reference_gauge_exp2b import Exp2BReferenceRunner

    class FakeSampler:
        def state_dict(self): return {}
        def load_state_dict(self, state): pass

    runner = Exp2BReferenceRunner(object(), FakeSampler(), score_gain=10.0, observations_per_update=32)
    state = runner.state_dict(); state["score_gain"] = 100.0
    with pytest.raises(ValueError, match="score-gain"):
        runner.load_boundary_state(state)


def test_exp2b_provenance_binds_score_gain(tmp_path) -> None:
    from cardioresp4d.experiments.respiratory_reference_gauge_exp2b import exp2b_provenance

    paths = []
    for index in range(5):
        path = tmp_path / f"input_{index}"
        path.write_bytes(str(index).encode())
        paths.append(path)
    kwargs = {"manifest": paths[0], "qc_table": paths[1], "canonical_domain": paths[2], "static_manifest": paths[3], "mean_manifest": paths[4], "config": {"exp2b": {"score_gain": 10.0}}}
    gain10 = exp2b_provenance(**kwargs, score_gain=10.0)
    gain100 = exp2b_provenance(**kwargs, score_gain=100.0)
    assert gain10 != gain100 and gain10["protocol"]["score_gain"] == 10.0 and gain100["protocol"]["score_gain"] == 100.0


def test_exp2b_comparison_extracts_required_metrics_and_fold_changes() -> None:
    from scripts.compare_exp2a_exp2b_score_gain import build_score_gain_comparison

    def report(gain: float, dynamic: float, used: float) -> dict:
        return {
            "score_gain": gain,
            "diagnostics": {"step_200": {"reconstruction_mse_full": 0.1, "unscaled_effective_score_dynamic_rms": dynamic, "used_score_dynamic_rms": used, "unscaled_effective_dvf_dynamic_rms_mm": dynamic, "used_dvf_dynamic_rms_mm": used}},
            "runner_report": {"gradient_summary": {"encoder_norm": {"mean": gain}, "encoder_rms": {"mean": gain / 10.0}, "l1_mbc_norm": {"mean": gain / 2.0}}, "loss_components": {key: {"mean": 1.0} for key in ("data", "smooth_resp", "mbc_normalization", "total")}},
        }

    result = build_score_gain_comparison(report(1.0, 1.0, 1.0), report(10.0, 2.0, 20.0), report(100.0, 3.0, 300.0))
    assert result["scientific_interpretation"] == "pending"
    assert result["experiments"]["gain100"]["used_score_dynamic_rms"] == 300.0
    assert result["comparisons"]["gain10_vs_gain1"]["used_score_dynamic_rms_fold_vs_gain1"] == 20.0


def test_comparison_reads_real_exp2a_report_schema() -> None:
    from scripts.compare_exp2a_exp2b_score_gain import build_score_gain_comparison

    exp2a = {
        "diagnostics": {"step_200": {"effective_score_dynamic_rms": 2.0, "effective_dvf_dynamic_rms_mm": 3.0}},
        "runner_report": {
            "reconstruction_mse_full": 0.11,
            "reconstruction_mse_SAX": 0.12,
            "reconstruction_mse_2CH": 0.13,
            "reconstruction_mse_4CH": 0.14,
            "loss_mean": 0.5,
        },
    }

    def exp2b_report(gain: float, used_score: float, used_dvf: float) -> dict:
        return {
            "score_gain": gain,
            "diagnostics": {"step_200": {"unscaled_effective_score_dynamic_rms": 2.0, "used_score_dynamic_rms": used_score, "unscaled_effective_dvf_dynamic_rms_mm": 3.0, "used_dvf_dynamic_rms_mm": used_dvf}},
            "runner_report": {"loss_mean": 0.5},
        }

    result = build_score_gain_comparison(exp2a, exp2b_report(10.0, 20.0, 30.0), exp2b_report(100.0, 200.0, 300.0))
    gain1 = result["experiments"]["gain1"]
    assert gain1["unscaled_effective_score_dynamic_rms"] == pytest.approx(2.0)
    assert gain1["used_score_dynamic_rms"] == pytest.approx(2.0)
    assert gain1["unscaled_effective_dvf_dynamic_rms_mm"] == pytest.approx(3.0)
    assert gain1["used_dvf_dynamic_rms_mm"] == pytest.approx(3.0)
    assert gain1["reconstruction_mse_full"] == pytest.approx(0.11)
    assert gain1["reconstruction_mse_SAX"] == pytest.approx(0.12)
    assert gain1["reconstruction_mse_2CH"] == pytest.approx(0.13)
    assert gain1["reconstruction_mse_4CH"] == pytest.approx(0.14)
    assert result["comparisons"]["gain10_vs_gain1"]["used_score_dynamic_rms_fold_vs_gain1"] == pytest.approx(10.0)
    assert result["scientific_interpretation"] == "pending"


def test_gain10_and_gain100_keep_same_seeded_model_initialization() -> None:
    from cardioresp4d.training.model import SourceFirstDynamicModel

    def build():
        return SourceFirstDynamicModel(torch.tensor([-2.0, -2.0, -2.0]), torch.tensor([2.0, 2.0, 2.0]), cardiac_lower_world_mm=torch.tensor([-1.0, -1.0, -1.0]), cardiac_upper_world_mm=torch.tensor([1.0, 1.0, 1.0]), n_dynamic_frames=1, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1, canonical_encoding_backend="torch_hash")

    torch.manual_seed(20260823); gain10_model = build(); torch.manual_seed(20260823); gain100_model = build()
    gain10_model.set_reference_score_gain(10.0); gain100_model.set_reference_score_gain(100.0)
    assert all(torch.equal(left, right) for left, right in zip(gain10_model.state_dict().values(), gain100_model.state_dict().values()))
