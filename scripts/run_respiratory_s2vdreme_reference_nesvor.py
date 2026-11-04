#!/usr/bin/env python3
"""Write or execute the auditable Experiment-1 static NeSVoR command."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import nesvor_reconstruct_command, write_static_reference_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-manifest", required=True, type=Path); parser.add_argument("--source-manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nesvor-executable", default="nesvor")
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    executable = shutil.which(args.nesvor_executable)
    if executable is None: parser.error(f"cannot resolve pinned NeSVoR executable: {args.nesvor_executable}")
    version = subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    if "0.6.0rc1" not in version: parser.error(f"NeSVoR executable version is not compatible with pinned 0.6.0rc1: {version}")
    payload = json.loads(args.mean_manifest.read_text())
    by_stack = {item["stack_file"]: item["stack_thickness_mm"] for item in payload["locations"]}
    stacks = [str(args.mean_manifest.parent / stack) for stack in sorted(by_stack)]
    command = nesvor_reconstruct_command(stacks, output_dir=args.output_dir, thicknesses_mm=[by_stack[Path(path).name] for path in stacks], smoke=args.smoke); command[0] = executable
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"mean_manifest": str(args.mean_manifest), "mean_manifest_sha256": hashlib.sha256(args.mean_manifest.read_bytes()).hexdigest(), "mode": "smoke" if args.smoke else "formal", "command": command, "nesvor_executable": executable, "nesvor_version": version}
    (args.output_dir / "nesvor_static_command.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(subprocess.list2cmdline(command))
    if args.execute:
        subprocess.run(command, check=True)
        print(write_static_reference_manifest(output_dir=args.output_dir, source_manifest=args.source_manifest, qc_table=args.qc_table, mean_slice_manifest=args.mean_manifest, command=command, executable=executable, version=version, mode=provenance["mode"]))


if __name__ == "__main__":
    main()
