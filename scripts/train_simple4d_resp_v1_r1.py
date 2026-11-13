#!/usr/bin/env python3
"""Joint canonical-motion R1; initialized from a validated R0 model only."""
import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from train_respiratory_s2vdreme_reference_v1 import _load_config
from audit_simple4d_resp_v1_checkpoints import load_inputs,output_sha256
from cardioresp4d.experiments.simple4d_resp_v1_r0 import diagnostic_metrics, state_digest, summarize_metrics
from cardioresp4d.experiments.simple4d_resp_v1_r1 import R1Runner,R1_OWNERS,R1_BOUNDARIES,initialize_from_r0
from cardioresp4d.training.sampler import ViewLocationBalancedSampler


def validate_r1_config(config,r0):
    expected=copy.deepcopy(r0)
    expected['experiment'].update(stage='R1',mode='joint_canonical_motion_refinement')
    expected['training']['boundaries']=R1_BOUNDARIES
    expected['training']['optimizer']['canonical_lr']=2e-5
    expected['canonical']={'source':'r0_checkpoint_nesvor_initialized','r1_trainable':True,'bootstrap_config':'respiratory_s2vdreme_reference_v1.yaml'}
    expected['source_r0']={'allowed_steps':[2000,10000],'load_model_weights_only':True}
    if config != expected: raise ValueError('R1 config differs from fixed joint-refinement protocol')


def write_json(path,data): path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def r1_diagnostics(model,sampler,runner):
    result=diagnostic_metrics(model,sampler)
    for key in list(result):
        if key.startswith('static_baseline_'):
            result['current_r1_canonical_no_motion_'+key[len('static_baseline_'):]]=result.pop(key)
    result.update(runner.canonical_drift())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','canonical-domain','static-manifest','output-dir','source-r0-checkpoint'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--experiment-config',type=Path,default=ROOT/'configs/simple4d_resp_v1_r1.yaml'); parser.add_argument('--device',default='cuda:0'); parser.add_argument('--stop-after',choices=R1_BOUNDARIES,required=True); parser.add_argument('--resume',type=Path)
    args=parser.parse_args(); args.device=torch.device(args.device)
    if args.device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    source=args.source_r0_checkpoint.resolve(); output=args.output_dir.resolve(); source_dir=source.parent
    if output==source_dir or source_dir in output.parents or output in source.parents: raise ValueError('R1 output directory must not overlap the R0 source directory')
    if args.resume is None and output.exists() and any(output.iterdir()): raise FileExistsError('fresh R1 output directory must be empty')
    config=_load_config(args.experiment_config); r0_config=_load_config(ROOT/'configs/simple4d_resp_v1.yaml'); validate_r1_config(config,r0_config)
    initial_source_hash=output_sha256(source)
    load_args=SimpleNamespace(manifest=args.manifest,qc_table=args.qc_table,canonical_domain=args.canonical_domain,static_manifest=args.static_manifest,device=args.device)
    r0_current_provenance={}
    r0_config_loaded,r0_current_provenance,model,cohort=load_inputs(load_args)
    if r0_config_loaded != r0_config: raise ValueError('R0 configuration mismatch')
    source_checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    current=model.state_dict(); incoming=source_checkpoint['model']
    if any(not torch.equal(incoming[k].cpu(),v.cpu()) for k,v in current.items() if k.startswith('canonical.')): raise ValueError('R0 canonical differs from admitted static state')
    init=initialize_from_r0(model,source_checkpoint,r0_current_provenance,r0_config)
    source_step=init['source_r0_step']; init['r1_initial_model_state_sha256']=state_digest(model); init['source_r0_model_state_sha256']=init['r1_initial_model_state_sha256']
    source_diagnostics=source_checkpoint.get('diagnostics',{}).get(f'step_{source_step}',{})
    source_baseline={k:v for k,v in source_diagnostics.items() if k.startswith('static_baseline_')}
    if not {'static_baseline_sampled_mse_full','static_baseline_sampled_mae_full'} <= source_baseline.keys(): raise ValueError('source R0 checkpoint lacks authoritative static baseline diagnostics')
    if output_sha256(source)!=initial_source_hash: raise RuntimeError('source checkpoint changed during initialization')
    sampler=ViewLocationBalancedSampler([frame for _,_,frames in cohort for frame in frames],seed=12)
    runner=R1Runner(model,sampler,source_step)
    owned={id(p) for name in R1_OWNERS for p in getattr(model,name).parameters()}
    if any(p.requires_grad and id(p) not in owned for p in model.parameters()): raise RuntimeError('unexpected trainable parameter')
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    provenance={'source_r0_path':str(source),'source_r0_sha256':initial_source_hash,'source_r0_step':source_step,'source_r0_provenance':source_checkpoint['provenance'],'r1_config_sha256':config_hash,'input_provenance':r0_current_provenance,'model_seed':6,'sampler_seed':12}
    diagnostics={}
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('effective_config') != config or saved.get('source_r0_checkpoint_sha256') != initial_source_hash: raise ValueError('R1 resume effective config or source hash mismatch')
        runner.restore(saved,provenance)
        init=saved['initialization_audit']; diagnostics=saved['diagnostics']; source_baseline=saved['source_static_nesvor_baseline']
    else:
        diagnostics['r1_step_0']=r1_diagnostics(model,sampler,runner)
    target=R1_BOUNDARIES[args.stop_after]
    if runner.global_step>target: raise ValueError('resume checkpoint is beyond requested R1 boundary')
    if args.resume and output.exists() and any(output.iterdir()) and not (output/'experiment_inputs.json').is_file(): raise ValueError('existing output directory lacks R1 identity metadata')
    if args.resume and (output/'experiment_inputs.json').exists():
        existing=json.loads((output/'experiment_inputs.json').read_text())
        if existing.get('provenance') != provenance: raise ValueError('existing output directory has different R1 provenance')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'experiment_inputs.json',{'manifest':str(args.manifest.resolve()),'qc_table':str(args.qc_table.resolve()),'canonical_domain':str(args.canonical_domain.resolve()),'static_manifest':str(args.static_manifest.resolve()),'source_r0_checkpoint':str(source),'provenance':provenance})
    write_json(output/'effective_config.json',config)
    while runner.global_step<target:
        runner.update()
        if runner.global_step in R1_BOUNDARIES.values():
            diagnostics[f'r1_step_{runner.global_step}']=r1_diagnostics(model,sampler,runner)
            if output_sha256(source)!=initial_source_hash: raise RuntimeError('source checkpoint changed during R1')
            checkpoint=runner.checkpoint(provenance); checkpoint.update({'effective_config':config,'initialization_audit':init,'diagnostics':diagnostics,'source_static_nesvor_baseline':source_baseline,'source_r0_checkpoint_path':str(source),'source_r0_checkpoint_sha256':initial_source_hash})
            torch.save(checkpoint,output/f'checkpoint_r1_step_{runner.global_step}.pt'); torch.save(checkpoint,output/'checkpoint_last.pt')
            gradients,losses=summarize_metrics(runner.metrics)
            report={'source_r0_step':source_step,'r1_joint_step':runner.global_step,'source_r0_checkpoint':str(source),'source_r0_sha256':initial_source_hash,'source_static_nesvor_baseline':source_baseline,'initialization_audit':init,'canonical_drift':runner.canonical_drift(),'owners':list(R1_OWNERS),'canonical_trainable':True,'optimizer_lrs':{g['name']:g['lr'] for g in runner.optimizer.param_groups},'scheduler':{'step_size':5000,'gamma':.5,'state':runner.scheduler.state_dict()},'diagnostics':diagnostics,'gradient_summary':gradients,'loss_component_summary':losses,'provenance':provenance}
            write_json(output/'training_report.json',report)
            with (output/'training_metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(runner.metrics[0])); writer.writeheader(); writer.writerows(runner.metrics)
    if output_sha256(source)!=initial_source_hash: raise RuntimeError('source checkpoint changed during R1')
    print(json.dumps({'source_r0_step':source_step,'r1_joint_step':runner.global_step,'output':str(output)}))


if __name__=='__main__': main()
