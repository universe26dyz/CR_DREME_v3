"""Locate unmodified pinned source trees; no upstream algorithm is copied here."""
from __future__ import annotations

import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_upstream_to_path(name: str) -> Path:
    if name == "SINR":
        root = Path(os.environ.get("CARDIORESP4D_SINR_ROOT", _PROJECT_ROOT.parents[1] / "external" / "SINR"))
    else:
        root = _PROJECT_ROOT / "third_party" / name
    if not root.is_dir():
        raise ImportError(f"Pinned upstream source is unavailable: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
