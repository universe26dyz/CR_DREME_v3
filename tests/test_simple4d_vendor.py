import hashlib
import json
from pathlib import Path
from third_party.simple4d.simple4d.networks import MotionCodeVAE, DeformCodeAdaptive
from cardioresp4d.simple4d.encoder import Simple4DViewConditionedMotionCodeVAE


def test_vendor():
    root = Path(__file__).resolve().parents[1]
    vendor = root / 'third_party/simple4d'
    manifest = json.loads((root / 'third_party/simple4d_VENDOR.json').read_text())
    assert manifest['commit'] == 'c466b276a438b3e052ad3b44d38f398872d8aeae'
    assert not (vendor / '.git').exists()
    assert 'Required Notice' in (vendor / 'LICENSE.md').read_text()
    for path, digest in manifest['key_files'].items():
        assert hashlib.sha256((vendor / path).read_bytes()).hexdigest() == digest
    assert issubclass(Simple4DViewConditionedMotionCodeVAE, MotionCodeVAE)
    assert Path(__import__(DeformCodeAdaptive.__module__, fromlist=['']).__file__).resolve().is_relative_to(vendor)
