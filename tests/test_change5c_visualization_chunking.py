"""CPU contracts for bounded Change5C visualization batching."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_change5c_closure as closure  # noqa: E402
import visualize_checkpoint_dynamics as dynamics  # noqa: E402


class _Observation:
    def __init__(self) -> None:
        self.image = torch.zeros(1, 2, 4)
        self.view, self.slice_id, self.dynamic_frame_id = "SAX", "SAX_s026", 42
        self.timestamp_s, self.qc_valid, self.qc_reason = 0.0, True, "valid"


class _PredictModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, observation, pixels: torch.Tensor, stage: str) -> dict:
        self.calls.append({"stage": stage, "pixels": pixels.detach().clone(), "psf_random": torch.rand(3)})
        return {"predicted_intensity": pixels[:, 0] + 10 * pixels[:, 1]}


class _Canonical:
    def __init__(self) -> None:
        self.inr = SimpleNamespace(train=lambda: None)


class _VisualizationModel(_PredictModel):
    def __init__(self) -> None:
        super().__init__()
        self.canonical = _Canonical()
        self.canonical_lower_world_mm = torch.zeros(3)
        self.canonical_upper_world_mm = torch.ones(3)
        self.cardiac_lower_world_mm = torch.zeros(3)
        self.cardiac_upper_world_mm = torch.ones(3)

    def to(self, device): return self
    def load_state_dict(self, state): return None
    def eval(self): return self


class Change5CVisualizationChunkingTest(unittest.TestCase):
    def test_slice_prediction_bounds_chunks_and_preserves_pixel_order(self) -> None:
        observation, model = _Observation(), _PredictModel()
        output = dynamics._slice_prediction(model, observation, "stage2c", seed=7, slice_chunk_size=3)
        self.assertEqual((2, 4), tuple(output.shape))
        self.assertTrue(torch.equal(output, torch.tensor([[0., 1., 2., 3.], [10., 11., 12., 13.]])))
        self.assertEqual([3, 3, 2], [call["pixels"].shape[0] for call in model.calls])
        self.assertTrue(torch.equal(torch.cat([call["pixels"] for call in model.calls]), torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [0., 1.], [1., 1.], [2., 1.], [3., 1.]])))

    def test_slice_prediction_pairs_psf_rng_across_resp_joint_and_checkpoints(self) -> None:
        observation = _Observation()
        resp, joint = _PredictModel(), _PredictModel()
        dynamics._slice_prediction(resp, observation, "stage2c", seed=7, slice_chunk_size=3)
        dynamics._slice_prediction(joint, observation, "stage3a", seed=7, slice_chunk_size=3)
        self.assertEqual(len(resp.calls), len(joint.calls))
        for left, right in zip(resp.calls, joint.calls):
            self.assertTrue(torch.equal(left["psf_random"], right["psf_random"]))

    def test_main_uses_independent_slice_and_volume_chunk_sizes(self) -> None:
        observation, model, volume_chunks = _Observation(), _VisualizationModel(), []

        def canonical_query(canonical, motion, grid, *, chunk_size):
            volume_chunks.append(chunk_size)
            return torch.zeros(grid.shape[:-1]), grid.clone()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = root / "config.yaml"; config.write_text("training:\n  normalization:\n    mode: none\n", encoding="utf-8")
            (root / "domain.json").write_text("{}", encoding="utf-8")
            output = root / "output"
            argv = ["visualize_checkpoint_dynamics.py", "--source-config", str(config), "--manifest", str(root / "manifest.csv"), "--qc-table", str(root / "qc.csv"), "--canonical-domain", str(root / "domain.json"), "--checkpoint", str(root / "checkpoint.pt"), "--view", "SAX", "--slice-id", "SAX_s026", "--output-dir", str(output), "--frames", "1", "--grid-shape", "2", "2", "2", "--slice-chunk-size", "2", "--volume-chunk-size", "7"]
            checkpoint = {"training_state": {"current_stage": "stage2a"}, "model": {}}
            with patch.object(sys, "argv", argv), patch.object(dynamics, "validate_source_first_config"), patch.object(dynamics, "observations_from_manifest", return_value=([observation], {}, [])), patch.object(dynamics.torch, "load", return_value=checkpoint), patch.object(dynamics, "detect_checkpoint_encoding_backend", return_value="hash"), patch.object(dynamics, "build_source_first_model", return_value=model), patch.object(dynamics, "_motion_for", return_value=(None, None, {})), patch.object(dynamics, "chunked_canonical_query", side_effect=canonical_query), patch.object(dynamics, "_save_png"):
                dynamics.main()
        self.assertTrue(all(call["pixels"].shape[0] <= 2 for call in model.calls))
        self.assertEqual([7, 7, 7], volume_chunks)

    def test_runner_passes_and_records_visualization_chunk_sizes(self) -> None:
        command_builder = getattr(closure, "visualization_command", None)
        sampling_builder = getattr(closure, "closure_sampling_metadata", None)
        self.assertIsNotNone(command_builder)
        self.assertIsNotNone(sampling_builder)
        if command_builder is None or sampling_builder is None:
            return
        command = command_builder("python", Path("config.yaml"), Path("manifest.csv"), Path("qc.csv"), Path("domain.json"), Path("checkpoint.pt"), "SAX", "SAX_s026", "cuda", 7, Path("output"))
        self.assertEqual("1024", command[command.index("--slice-chunk-size") + 1])
        self.assertEqual("65536", command[command.index("--volume-chunk-size") + 1])
        self.assertEqual({"slice_chunk_size": 1024, "volume_chunk_size": 65536}, sampling_builder()["visualization"])


if __name__ == "__main__":
    unittest.main()
