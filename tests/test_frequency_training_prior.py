from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
class FrequencyTrainingPrior(unittest.TestCase):
 def test_phase1_schema_preserves_multiple_cardiac_bins(self):
  payload={'respiratory':{'verified_band_hz':[[.2,.3]]},'cardiac':{'union_resolution_bins_hz':[[1.,1.2],[2.,2.2]]}}
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'frequency_bands.json'; path.write_text(json.dumps(payload)); prior=load_training_frequency_prior(path)
  self.assertEqual([(1.,1.2),(2.,2.2)],prior.cardiac_bands_hz); self.assertEqual([(.2,.3)],prior.respiratory_bands_hz)
