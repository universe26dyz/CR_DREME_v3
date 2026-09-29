#!/usr/bin/env python3
"""Fixed-location Change5C score, PCA, and NUDFT comparison visualization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformPrior
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.losses.frequency_loss import nonuniform_dft_at_frequencies
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import checkpoint_compatible_dynamic_frame_count, is_hard_invalid_reason
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.stage_contract import stage_contract
from diagnose_change4_checkpoint import _encode_location, frequency_semantics_record
from train_source_first import observations_from_manifest


def _experiment(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "," not in value.split("=", 1)[1]: raise argparse.ArgumentTypeError("--experiment must be LABEL=CONFIG,CHECKPOINT")
    label, rest = value.split("=", 1); config, checkpoint = rest.split(",", 1); return label, Path(config), Path(checkpoint)


def _display(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float(); return (value - value.mean()) / value.std(unbiased=False).clamp_min(torch.finfo(value.dtype).eps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", type=_experiment, required=True); parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--view", required=True); parser.add_argument("--slice-id", required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--output-png", type=Path, required=True); parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required") from exc
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8")); prior = load_training_frequency_prior(args.frequency_bands, allow_template_fallback=False); pca = PCAWaveformPrior.load(args.frequency_bands, strict=False)
    outputs = []
    for label, config_path, checkpoint_path in args.experiment:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
        observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
        selected = sorted([item for item in observations if item.qc_valid and not is_hard_invalid_reason(item.qc_reason) and item.view == args.view and item.slice_id == args.slice_id], key=lambda item: item.timestamp_s)
        if len(selected) < 3: parser.error(f"{label}: selected location has fewer than three valid frames")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); stage = str(checkpoint["training_state"]["current_stage"]); contract = stage_contract(stage)
        n_dynamic_frames = checkpoint_compatible_dynamic_frame_count(checkpoint, observations)
        model = build_source_first_model(config, domain, n_dynamic_frames=n_dynamic_frames, device=device, canonical_encoding_backend=detect_checkpoint_encoding_backend(checkpoint["model"])).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
        with torch.no_grad():
            resp, card, timestamps = _encode_location(model, selected, contract, device)
        local = prior.for_location(args.view, args.slice_id); record = frequency_semantics_record(args.view, args.slice_id, resp, card, timestamps, local, pca_waveform_prior=pca, pca_waveform_ridge=float(config["training"].get("temporal_auxiliary", {}).get("pca_waveform_ridge", 1e-4)), pca_waveform_min_frames=int(config["training"].get("temporal_auxiliary", {}).get("pca_waveform_min_frames", 8)))
        frequencies = torch.linspace(.0, max(float(local.cardiac_bands_hz[-1][-1]), float(local.respiratory_bands_hz[-1][-1])) * 1.5, 512, device=device, dtype=timestamps.dtype)
        outputs.append({"label": label, "timestamps": timestamps.cpu(), "resp": resp.cpu(), "card": card.cpu(), "card_spectrum": nonuniform_dft_at_frequencies(card, timestamps, frequencies).abs().mean(-1).cpu(), "resp_spectrum": nonuniform_dft_at_frequencies(resp, timestamps, frequencies).abs().mean(-1).cpu(), "frequencies": frequencies.cpu(), "card_bands": local.cardiac_bands_hz, "resp_bands": local.respiratory_bands_hz, "record": record, "pca": pca.match(args.view, args.slice_id, timestamps, device=device, dtype=card.dtype)})
    figure, axes = plt.subplots(len(outputs), 3, figsize=(17, 3.8 * len(outputs)), squeeze=False)
    for row, output in enumerate(outputs):
        times = output["timestamps"].numpy(); card, resp = output["card"], output["resp"]
        for channel in range(card.shape[1]): axes[row, 0].plot(times, _display(card[:, channel]).numpy(), label=f"card {channel}")
        match = output["pca"]
        if match is not None: axes[row, 0].plot(times, _display(match.waveform).numpy(), "k--", label=f"Phase1 PCA PC {match.selected_pc}")
        axes[row, 0].set_title("Cardiac scores + Phase1 PCA (DISPLAY-ONLY standardized)"); axes[row, 0].legend(fontsize=7)
        for channel in range(resp.shape[1]): axes[row, 1].plot(times, _display(resp[:, channel]).numpy(), label=f"resp {channel}")
        axes[row, 1].set_title("Respiratory scores (DISPLAY-ONLY standardized)"); axes[row, 1].legend(fontsize=7)
        frequency = output["frequencies"].numpy(); axes[row, 2].plot(frequency, output["card_spectrum"].numpy(), label="cardiac NUDFT"); axes[row, 2].plot(frequency, output["resp_spectrum"].numpy(), label="respiratory NUDFT")
        for low, high in output["card_bands"]: axes[row, 2].axvspan(low, high, color="tab:red", alpha=.12)
        for low, high in output["resp_bands"]: axes[row, 2].axvspan(low, high, color="tab:blue", alpha=.10)
        record = output["record"]; axes[row, 2].axvline(record["card_peak_hz"], color="tab:red", linestyle="--", label=f"card dominant peak {record['card_peak_hz']:.3g} Hz"); axes[row, 2].axvline(record["resp_peak_hz"], color="tab:blue", linestyle=":", label=f"resp dominant peak {record['resp_peak_hz']:.3g} Hz"); axes[row, 2].set_title(f"NUDFT bands; PCA R²={record['cardiac_pca_waveform_r2']}; target frac={record['cardiac_target_fraction']:.3g}; target/wrong={record['card_target_over_wrong']:.3g}; std={record['card_score_std']:.3g}"); axes[row, 2].legend(fontsize=7)
        axes[row, 0].set_ylabel(output["label"])
    figure.suptitle(f"{args.view}/{args.slice_id}: true timestamps; PCA sign/scale are arbitrary")
    figure.tight_layout(); args.output_png.parent.mkdir(parents=True, exist_ok=True); figure.savefig(args.output_png, dpi=150); plt.close(figure)
    args.output_json.write_text(json.dumps({"location": {"view": args.view, "slice_id": args.slice_id}, "records": [{"label": item["label"], **item["record"]} for item in outputs], "display_normalization": "per-channel z-score for plotting only; metrics use raw scores"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_png)}, indent=2))


if __name__ == "__main__":
    main()
