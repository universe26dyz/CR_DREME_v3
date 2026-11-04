#!/usr/bin/env python3
"""Write or execute the auditable Experiment-1 static NeSVoR command."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import nesvor_reconstruct_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-manifest", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.mean_manifest.read_text())
    by_stack = {item["stack_file"]: item["stack_thickness_mm"] for item in payload["locations"]}
    stacks = [str(args.mean_manifest.parent / stack) for stack in sorted(by_stack)]
    command = nesvor_reconstruct_command(stacks, output_dir=args.output_dir, thicknesses_mm=[by_stack[Path(path).name] for path in stacks], smoke=args.smoke)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"mean_manifest": str(args.mean_manifest), "mean_manifest_sha256": hashlib.sha256(args.mean_manifest.read_bytes()).hexdigest(), "mode": "smoke" if args.smoke else "formal", "command": command}
    (args.output_dir / "nesvor_static_command.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(subprocess.list2cmdline(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
