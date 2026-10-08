"""Resume selection for the formal C4 paper-aligned segment launcher."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

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


def test_preflight_path_does_not_pollute_training_output(tmp_path: Path) -> None:
    training_output = tmp_path / "v1_change4_paperaligned_full"
    preflight_output = tmp_path / "c4_paperaligned_loss_scale_preflight.json"
    preflight_output.write_text("{}", encoding="utf-8")
    launcher.prepare_training_output_dir(training_output, resume=False)
    assert preflight_output.parent != training_output
    assert list(training_output.iterdir()) == []


def test_server_document_separates_preflight_and_training_outputs() -> None:
    document = (ROOT / "C4_PAPERALIGNED_SERVER_RUN.md").read_text(encoding="utf-8")
    assert "PREFLIGHT_OUT=/data/dengyz/dataset/CR_DREME_v3/c4_paperaligned_loss_scale_preflight.json" in document
    assert "--output-json ${PREFLIGHT_OUT}" in document
    assert "--output-dir ${TRAIN_OUT}" in document
    assert "${TRAIN_OUT}/loss_scale_preflight.json" not in document
    assert "mkdir -p ${TRAIN_OUT}" not in document
    assert "> ${TRAIN_OUT}/" not in document


def test_full_launcher_still_refuses_nonempty_training_output(tmp_path: Path) -> None:
    output = tmp_path / "v1_change4_paperaligned_full"
    output.mkdir()
    (output / "unrelated_preflight.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        launcher.prepare_training_output_dir(output, resume=False)
