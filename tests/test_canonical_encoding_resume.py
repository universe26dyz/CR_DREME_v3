"""Checkpoint-stable NeSVoR encoding backend selection."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.adapters.nesvor_inr import NeSVoRCanonicalAdapter, detect_checkpoint_encoding_backend  # noqa: E402
import nesvor.inr.models as nesvor_models  # noqa: E402


class CanonicalEncodingResumeTest(unittest.TestCase):
    def test_detects_legacy_torch_hash_checkpoint_even_if_runtime_preference_changes(self) -> None:
        state = {"canonical.inr.encoding.box_offsets": torch.zeros(1), "canonical.inr.encoding.embeddings.0.weight": torch.zeros(2, 2)}
        self.assertEqual("torch_hash", detect_checkpoint_encoding_backend(state))

    def test_rejects_ambiguous_or_unknown_legacy_checkpoint_encoding(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            detect_checkpoint_encoding_backend({"canonical.inr.encoding.params": torch.zeros(1), "canonical.inr.encoding.box_offsets": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "cannot determine"):
            detect_checkpoint_encoding_backend({"canonical.inr.density_net.0.weight": torch.zeros(1)})

    def test_torch_hash_checkpoint_loads_exactly_when_runtime_prefers_tinycudann(self) -> None:
        bounds = torch.tensor([[-4., -4., -4.], [4., 4., 4.]])
        saved = NeSVoRCanonicalAdapter(bounds, width=8, depth=1, n_features_z=4, finest_resolution=4., encoding_backend="torch_hash")
        state = saved.state_dict()
        checkpoint_state = {"canonical." + key: value for key, value in state.items()}
        backend = detect_checkpoint_encoding_backend(checkpoint_state)
        previous = nesvor_models.USE_TORCH
        nesvor_models.USE_TORCH = False  # simulate a later process with tcnn available
        try:
            resumed = NeSVoRCanonicalAdapter(bounds, width=8, depth=1, n_features_z=4, finest_resolution=4., encoding_backend=backend)
            resumed.load_state_dict(state)
        finally:
            nesvor_models.USE_TORCH = previous
        self.assertEqual("torch_hash", resumed.encoding_backend)
        self.assertTrue(all(torch.equal(value, resumed.state_dict()[key]) for key, value in state.items()))

    def test_tinycudann_checkpoint_backend_fails_fast_when_unavailable(self) -> None:
        if not nesvor_models.USE_TORCH:
            self.skipTest("tinycudann is available on this host")
        with self.assertRaisesRegex(RuntimeError, "requires tinycudann"):
            NeSVoRCanonicalAdapter(torch.tensor([[-4., -4., -4.], [4., 4., 4.]]), width=8, depth=1, n_features_z=4, finest_resolution=4., encoding_backend="tinycudann")


if __name__ == "__main__":
    unittest.main()
