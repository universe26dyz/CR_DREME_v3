"""CPU contracts for read-only gradient-norm accounting."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_stage3a_loss_gradients import report_loss_gradients  # noqa: E402


class GradientAuditTest(unittest.TestCase):
    def test_reports_raw_and_weighted_norms_without_optimizer_step(self) -> None:
        card_head, card_mbc, frozen = torch.nn.Parameter(torch.tensor(2.)), torch.nn.Parameter(torch.tensor(3.)), torch.nn.Parameter(torch.tensor(4.), requires_grad=False)
        before = (card_head.detach().clone(), card_mbc.detach().clone(), frozen.detach().clone())
        result = report_loss_gradients({"synthetic": card_head * card_mbc}, 0.5, {"cardiac_film_head": [card_head], "cardiac_mbc": [card_mbc], "frozen": [frozen]})
        self.assertGreater(result["synthetic"]["raw"]["cardiac_film_head"], 0.)
        self.assertAlmostEqual(result["synthetic"]["raw"]["cardiac_film_head"] * .5, result["synthetic"]["weighted"]["cardiac_film_head"])
        self.assertEqual(0., result["synthetic"]["raw"]["frozen"])
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, (card_head, card_mbc, frozen))))


if __name__ == "__main__":
    unittest.main()
