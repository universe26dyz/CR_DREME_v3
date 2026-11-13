#!/usr/bin/env python3
"""Admit a corrected regular SAX stack to upstream SIMPLE-4D NPZ format."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map,apply_corrected_static_poses
from cardioresp4d.experiments.simple4d_native_sax import build_native_sax_series,save_native_npz


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','static-manifest','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError('output directory must be empty')
    static=json.loads(args.static_manifest.read_text()); pose=Path(static['corrected_pose_manifest']['path'])
    if static.get('source_manifest_sha256')!=digest(args.manifest) or static.get('qc_table_sha256')!=digest(args.qc_table) or static['corrected_pose_manifest'].get('sha256')!=digest(pose):
        raise ValueError('static source / QC / corrected-pose binding mismatch')
    observations,normalization,groups=observations_from_manifest(args.manifest,args.qc_table,torch.device('cpu'),normalization_mode='per_series',assign_slice_positions=True)
    observations=apply_corrected_static_poses(observations,load_corrected_pose_map(pose))
    series,metadata=build_native_sax_series(observations)
    args.output_dir.mkdir(parents=True,exist_ok=True); data=args.output_dir/'sax_input.npz'; save_native_npz(series,data)
    metadata.update({'normalization':'per_series','normalization_parameters':normalization,'normalization_groups':groups,'source_sha256':{'manifest':digest(args.manifest),'qc_table':digest(args.qc_table),'static_manifest':digest(args.static_manifest),'corrected_pose_manifest':digest(pose)},'output_npz_sha256':digest(data)})
    (args.output_dir/'input_adapter_manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'frames':len(series),'positions':series.num_positions,'output':str(data)}))


if __name__=='__main__': main()
