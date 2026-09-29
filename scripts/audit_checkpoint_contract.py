#!/usr/bin/env python3
"""Read-only Stage3a checkpoint/config consistency report for Change5C."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cardioresp4d.training.trainer import effective_loss_weights


GROUPS = {
    "canonical_inr": lambda key: key.startswith("canonical."),
    "respiratory_mbc": lambda key: key.startswith("respiratory_mbc."),
    "shared_film_image_geometry_resp": lambda key: key.startswith("film_encoder.") and not key.startswith("film_encoder.card."),
    "cardiac_film_head": lambda key: key.startswith("film_encoder.card."),
    "cardiac_mbc": lambda key: key.startswith("cardiac_mbc."),
}
RELEVANT_LOSS_KEYS = ("cardiac_target_concentration", "cardiac_pca_waveform", "respiratory_leakage_in_card", "cardiac_leakage_in_resp", "zero_mean_score", "smooth_card", "mbc_normalization")


def checkpoint_effective_loss_weights(checkpoint: dict) -> dict[str, float]:
    """Read the actual serialized training report; never substitute YAML values."""
    weights = checkpoint.get("report", {}).get("effective_config", {}).get("loss_weights")
    if not isinstance(weights, dict):
        raise ValueError("checkpoint-effective loss_weights are missing from checkpoint['report']['effective_config']['loss_weights']")
    return {str(key): float(value) for key, value in weights.items()}


def relevant_weight_agreement(checkpoint_weights: dict[str, float], supplied_weights: dict[str, float]) -> dict:
    supplied_effective = effective_loss_weights(supplied_weights)
    by_key = {key: {"checkpoint_effective": checkpoint_weights.get(key), "supplied_config_effective": supplied_effective.get(key), "agree": key in checkpoint_weights and float(checkpoint_weights[key]) == float(supplied_effective[key])} for key in RELEVANT_LOSS_KEYS}
    return {"by_key": by_key, "all_relevant_keys_agree": all(value["agree"] for value in by_key.values())}


def state_fingerprint(state: dict[str, torch.Tensor], predicate) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in state if predicate(key)):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _experiment(value: str) -> tuple[str, Path, Path]:
    try:
        label, config, checkpoint = value.split("=", 1)[0], *value.split("=", 1)[1].split(",", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--experiment must be LABEL=CONFIG,CHECKPOINT") from exc
    return label, Path(config), Path(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", type=_experiment, required=True, metavar="LABEL=CONFIG,CHECKPOINT")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for label, config_path, checkpoint_path in args.experiment:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]; training_state = checkpoint.get("training_state", {})
        checkpoint_weights = checkpoint_effective_loss_weights(checkpoint); supplied_weights = config["training"].get("loss_weights", {})
        records.append({"label": label, "checkpoint_path": str(checkpoint_path), "config_path": str(config_path), "current_stage": training_state.get("current_stage"), "global_step": training_state.get("global_step"), "stage_step": training_state.get("stage_step"), "checkpoint_effective_loss_weights": checkpoint_weights, "supplied_config_loss_weights": supplied_weights, "relevant_loss_weight_agreement": relevant_weight_agreement(checkpoint_weights, supplied_weights), "encoding_backend": next(("tinycudann" if "encoding.params" in key else "torch" for key in state if key.startswith("canonical.")), "unknown"), "trainable_stage_identity": "stage3a: cardiac FiLM head + cardiac MBC only" if training_state.get("current_stage") == "stage3a" else "reported checkpoint stage is not stage3a", "fingerprints": {name: state_fingerprint(state, predicate) for name, predicate in GROUPS.items()}})
    frozen = ("canonical_inr", "respiratory_mbc", "shared_film_image_geometry_resp")
    payload = {"status": "read_only", "records": records, "frozen_group_comparison": {group: {"all_equal": len({record["fingerprints"][group] for record in records}) == 1, "by_experiment": {record["label"]: record["fingerprints"][group] for record in records}} for group in frozen}, "note": "Fingerprint differences are reported, not treated as an error: continuation provenance may legitimately differ."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiments": [record["label"] for record in records], "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
