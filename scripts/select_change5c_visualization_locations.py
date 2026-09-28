#!/usr/bin/env python3
"""Select compact deterministic Change5C visualization locations with reasons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cardioresp4d.diagnostics.change5c import select_visualization_locations


def _records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8")); return payload.get("records", payload.get("all_locations", {}).get("records", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-semantic", type=Path, required=True); parser.add_argument("--candidate-semantic", type=Path, required=True); parser.add_argument("--candidate-ablation", type=Path, required=True); parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    ablation = json.loads(args.candidate_ablation.read_text(encoding="utf-8"))["records"]
    fixed = (("SAX", "SAX_s026"), ("SAX", "SAX_s001"), ("2CH", "2CH_s001"), ("4CH", "4CH_s001"))
    locations = select_visualization_locations(_records(args.baseline_semantic), _records(args.candidate_semantic), ablation, fixed=fixed)
    payload = {"locations": locations, "selection_contract": "Fixed representatives plus paired PCA and cardiac-ablation extremes; deduplicated locations retain all selection reasons."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_locations": len(locations), "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
