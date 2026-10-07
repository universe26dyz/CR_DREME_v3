"""Resume selection for the formal C4 paper-aligned segment launcher."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_c4_paperaligned_full as launcher  # noqa: E402


def test_resume_starts_after_last_completed_segment(tmp_path: Path) -> None:
    schedule = launcher.segment_schedule()
    first = tmp_path / schedule[0][0]
    first.mkdir()
    torch.save({"training_state": {"current_segment": "s1a"}}, first / "source_first_last.pt")
    index, parent = launcher.next_unfinished_segment(tmp_path, schedule)
    assert index == 1
    assert parent == first / "source_first_last.pt"
