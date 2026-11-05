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

from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import (
    HISTORICAL_NESVOR_RUNTIME_PROFILE,
    nesvor_reconstruct_command,
    verify_historical_nesvor_runtime,
    write_static_reference_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-manifest", required=True, type=Path); parser.add_argument("--source-manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nesvor-executable", type=Path)
    parser.add_argument("--nesvor-python", type=Path)
    parser.add_argument("--nesvor-source-root", type=Path)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runtime_args = (args.nesvor_executable, args.nesvor_python, args.nesvor_source_root)
    if any(value is not None for value in runtime_args) and not all(value is not None for value in runtime_args):
        parser.error("--nesvor-executable, --nesvor-python, and --nesvor-source-root must be supplied together")
    if args.execute and not all(value is not None for value in runtime_args):
        parser.error("--execute requires explicit --nesvor-executable, --nesvor-python, and --nesvor-source-root")
    runtime_provenance = (
        verify_historical_nesvor_runtime(executable=args.nesvor_executable, runtime_python=args.nesvor_python, source_root=args.nesvor_source_root)
        if all(value is not None for value in runtime_args)
        else {"runtime_profile": HISTORICAL_NESVOR_RUNTIME_PROFILE, "verification_status": "not_performed", "reason": "command-only run without explicit historical runtime identity"}
    )
    payload = json.loads(args.mean_manifest.read_text())
    by_stack = {item["stack_file"]: item["stack_thickness_mm"] for item in payload["locations"]}
    stacks = [str(args.mean_manifest.parent / stack) for stack in sorted(by_stack)]
    command = nesvor_reconstruct_command(stacks, output_dir=args.output_dir, thicknesses_mm=[by_stack[Path(path).name] for path in stacks], smoke=args.smoke)
    if args.nesvor_executable is not None:
        command[0] = str(args.nesvor_executable.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"mean_manifest": str(args.mean_manifest), "mean_manifest_sha256": hashlib.sha256(args.mean_manifest.read_bytes()).hexdigest(), "mode": "smoke" if args.smoke else "formal", "command": command, "historical_runtime": runtime_provenance}
    (args.output_dir / "nesvor_static_command.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(subprocess.list2cmdline(command))
    if args.execute:
        subprocess.run(command, check=True)
        print(write_static_reference_manifest(output_dir=args.output_dir, source_manifest=args.source_manifest, qc_table=args.qc_table, mean_slice_manifest=args.mean_manifest, command=command, runtime_provenance=runtime_provenance, mode=provenance["mode"]))


if __name__ == "__main__":
    main()
