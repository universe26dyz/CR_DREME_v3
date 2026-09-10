from __future__ import annotations
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
class SourceLock(unittest.TestCase):
 def test_pinned_key_source_hashes_verify(self): self.assertTrue(verify_vendored_source_lock(ROOT)['verified'])
