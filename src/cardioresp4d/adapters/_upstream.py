"""Locate unmodified pinned source trees; no upstream algorithm is copied here."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_upstream_to_path(name: str) -> Path:
    """Expose a vendored source root, with an explicit debug-only override.

    The source-first runtime must be self-contained in a clean v3 checkout.
    ``CARDIORESP4D_<NAME>_ROOT`` is deliberately optional and never supplies a
    default outside this repository.
    """
    root = Path(os.environ.get(f"CARDIORESP4D_{name.upper()}_ROOT", _PROJECT_ROOT / "third_party" / name))
    if not root.is_dir():
        raise ImportError(f"Pinned upstream source is unavailable: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
