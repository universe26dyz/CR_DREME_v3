from __future__ import annotations
import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cardioresp4d.losses.frequency_loss import dreme_cardiac_leakage_in_resp,dreme_respiratory_leakage_in_card
class DremeFrequencyLosses(unittest.TestCase):
 def test_eq8_and_eq9_detect_contamination_with_irregular_times(self):
  t=torch.tensor([0.,.09,.21,.31,.48,.62,.79,1.]); cardiac=torch.sin(2*torch.pi*2*t)[:,None]; respiratory=torch.sin(2*torch.pi*.25*t)[:,None]
  self.assertGreater(float(dreme_cardiac_leakage_in_resp(respiratory+cardiac,t,[(1.8,2.2)],[(1.3,1.7)])),float(dreme_cardiac_leakage_in_resp(respiratory,t,[(1.8,2.2)],[(1.3,1.7)])))
  value=dreme_respiratory_leakage_in_card(cardiac+respiratory,t,[(.2,.3)]); self.assertTrue(torch.isfinite(value)); value.backward() if value.requires_grad else None
