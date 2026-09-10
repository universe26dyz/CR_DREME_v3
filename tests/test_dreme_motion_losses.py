from __future__ import annotations
import sys, unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cardioresp4d.losses.motion_loss import dreme_mbc_normalization, dreme_zero_mean_scores

class DremeMotionLosses(unittest.TestCase):
 def test_eq6_minimum_is_unit_mbc_not_zero(self):
  unit=torch.ones(1,4,8,3); zero=torch.zeros_like(unit); double=unit*2
  self.assertLess(float(dreme_mbc_normalization(unit)),1e-6)
  self.assertGreater(float(dreme_mbc_normalization(zero)),.1); self.assertGreater(float(dreme_mbc_normalization(double)),.1)
 def test_eq7_removes_mean_not_amplitude(self):
  self.assertLess(float(dreme_zero_mean_scores(torch.tensor([[-1.],[1.],[-1.],[1.]]))),1e-7)
  self.assertLess(float(dreme_zero_mean_scores(torch.tensor([[-10.],[10.]]))),1e-7)
  self.assertGreater(float(dreme_zero_mean_scores(torch.ones(4,1))),.1)
