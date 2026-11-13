#!/usr/bin/env python3
"""Joint-from-scratch S0 using the vendored upstream SIMPLE4D training_step."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from third_party.simple4d.simple4d.config import load_config
from third_party.simple4d.simple4d.data import SliceSeries
from third_party.simple4d.simple4d.model import SIMPLE4D
from cardioresp4d.experiments.simple4d_native_sax import NativeSAXRunner,validate_native_config,validate_native_backend,validate_native_resume
from cardioresp4d.training.runtime_state import set_reproducibility


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/simple4d_native_sax_v1.yaml'); p.add_argument('--stop-after',type=int,choices=(200,2000,5000,10000),required=True)
    p.add_argument('--backend',choices=('tcnn','pytorch_fallback'),default='tcnn'); p.add_argument('--device',type=int,default=0); p.add_argument('--resume',type=Path)
    a=p.parse_args(); data=a.input_dir/'sax_input.npz'; manifest_path=a.input_dir/'input_adapter_manifest.json'; manifest=json.loads(manifest_path.read_text())
    if manifest.get('output_npz_sha256')!=digest(data) or not manifest.get('scope','').startswith('SAX-only'): raise ValueError('S0 adapter provenance mismatch')
    vendor_path=ROOT/'third_party/simple4d_VENDOR.json'; vendor=json.loads(vendor_path.read_text())
    if vendor['commit']!='c466b276a438b3e052ad3b44d38f398872d8aeae' or any(digest(ROOT/'third_party/simple4d'/name)!=value for name,value in vendor['key_files'].items()): raise ValueError('vendor provenance mismatch')
    from third_party.simple4d.simple4d import networks
    validate_native_backend(a.backend,networks.HAS_TCNN,torch.cuda.is_available())
    cfg=load_config(a.config,data=str(data),out_dir=str(a.output_dir),device=a.device,num_workers=0)
    validate_native_config(cfg)
    cfg.slice_thickness=manifest['slice_thickness_mm']
    series=SliceSeries.load(data); train,held=series.split(cfg.n_holdout)
    if len(train)<=0 or len(held)!=cfg.n_holdout: raise ValueError('insufficient frames for upstream holdout policy')
    if not any(series.depth_crop[0] <= series.slice_index[j] < series.num_positions-series.depth_crop[1] for j in held): raise ValueError('no held-out frames inside upstream depth crop')
    if a.input_dir.resolve() in a.output_dir.resolve().parents or a.input_dir.resolve()==a.output_dir.resolve(): raise ValueError('S0 output must be separate from its input directory')
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.resume: raise FileExistsError('fresh output directory must be empty')
    set_reproducibility(cfg.seed)
    model=SIMPLE4D(cfg,series.frames.shape[1],series.frames.shape[2],series.num_positions,series.spacing,series.depth_crop)
    backend='tcnn' if model.feature_net.use_tcnn else 'pytorch_fallback'
    if backend!=a.backend: raise RuntimeError('constructed backend does not match requested backend')
    runner=NativeSAXRunner(model,series,train,batch_size=cfg.batch_size,seed=cfg.loader_seed)
    provenance={'input_npz_sha256':digest(data),'input_manifest_sha256':digest(manifest_path),'config_sha256':digest(a.config),'vendor_manifest_sha256':digest(vendor_path),'vendor_commit':vendor['commit'],'backend':backend,'single_process_epoch_permutation':True}
    if a.resume:
        last=a.output_dir/'checkpoint_last.pt'
        if a.output_dir.exists() and any(a.output_dir.iterdir()):
            if not last.is_file() or torch.load(last,map_location='cpu',weights_only=False).get('provenance')!=provenance: raise ValueError('existing S0 output directory provenance mismatch')
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        validate_native_resume(state,provenance)
        runner.load_state_dict(state['runner'])
    if runner.global_step>a.stop_after: raise ValueError('resume step exceeds target')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'input_adapter_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output_dir/'effective_config.json').write_text(json.dumps(vars(cfg),indent=2)+'\n')
    (a.output_dir/'vendor_provenance.json').write_text(json.dumps({'manifest':vendor,'sha256':provenance['vendor_manifest_sha256']},indent=2)+'\n')
    (a.output_dir/'backend_provenance.json').write_text(json.dumps({'backend':backend,'not_numerically_interchangeable':True},indent=2)+'\n')
    while runner.global_step<a.stop_after:
        runner.update()
        if runner.global_step in (200,2000,5000,10000):
            payload={'schema':'simple4d_s0_v1','provenance':provenance,'runner':runner.state_dict()}
            torch.save(payload,a.output_dir/f'checkpoint_step_{runner.global_step}.pt'); torch.save(payload,a.output_dir/'checkpoint_last.pt')
            with (a.output_dir/'training_metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(runner.metrics[0])); writer.writeheader(); writer.writerows(runner.metrics)
            (a.output_dir/'training_report.json').write_text(json.dumps({'step':runner.global_step,'scope':'SAX-only native regular stack','backend':backend,'provenance':provenance,'heldout_frames':len(held)},indent=2)+'\n')
    model.save(a.output_dir/'model_final.pth')
    from third_party.simple4d.simple4d.evaluation import heldout_metrics
    result=heldout_metrics(model,series,held)
    (a.output_dir/'upstream_heldout_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
