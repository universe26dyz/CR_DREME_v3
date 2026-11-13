#!/usr/bin/env python3
"""Read-only cardiorespiratory identity analysis before R1."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from audit_simple4d_resp_v1_checkpoints import load_inputs, load_checkpoint, output_sha256
from render_simple4d_resp_v1_dynamic_volume import verify_stage
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import diagnostic_rng_context
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import global_latent_pca
from cardioresp4d.experiments.simple4d_resp_v1_identity_stages import read_latent
from cardioresp4d.experiments.simple4d_resp_v1_dynamic_visualization import load_selected_volumes
from cardioresp4d.experiments.simple4d_resp_v1_anatomy import derive_anatomy_crop
from cardioresp4d.roi.cardiac_box import load_cardiac_box_config
from cardioresp4d.experiments.simple4d_cardioresp_stages import pure_stage, motion_stage, residual_stage
from cardioresp4d.experiments.simple4d_resp_v1_identity_stages import write_csv, json_write


def prepare_step(a,step,base):
    checkpoint=a.run_dir/f'checkpoint_step_{step}.pt'; digest=output_sha256(checkpoint); prefix=f'step_{step}/'
    latent=verify_stage(a.audit_root,'01_latent',step,digest,[prefix+'frame_latent_scores.csv',prefix+'pca_basis.npz'])
    required=[prefix+'crop_manifest.json',prefix+'selected_states.csv']
    for state in ('low','mid','high'):
        required.extend([prefix+f'volumes/SAX_{state}_pc1.nii.gz',prefix+f'volumes/SAX_{state}_pc1.json'])
    dvf=verify_stage(a.audit_root,'04_dvf',step,digest,required)
    if latent['input_artifact_sha256']!=base['provenance']['hashes'] or dvf['input_artifact_sha256']!=base['provenance']['hashes']:
        raise ValueError('stage and current input provenance differ')
    checkpoint_data=load_checkpoint(checkpoint,step,base['config'],base['provenance'],base['model'])
    rows=read_latent(a.audit_root/'01_latent'/prefix/'frame_latent_scores.csv',base['cohort'])
    by_key={(r['view'],r['slice_id'],r['dynamic_frame_id']):r for r in rows}
    means=[]; ordered=[]
    with torch.no_grad():
        for view,slice_id,frames in base['cohort']:
            group=[]
            for obs in frames:
                row=by_key[(view,str(slice_id),int(obs.dynamic_frame_id))]
                group.append(base['model'].encode_observation(obs,sample=False)['mu'].squeeze(0).cpu().numpy())
                ordered.append(row)
            means.append(np.stack(group))
    pca=global_latent_pca(means)
    scores=np.concatenate([x[:,0] for x in pca['scores']]); expected=np.asarray([r['pc1'] for r in ordered])
    if not np.allclose(scores,expected,atol=1e-4,rtol=1e-5): raise ValueError(f'recomputed PC1 differs from recorded scores at step {step}')
    basis_path=a.audit_root/'01_latent'/prefix/'pca_basis.npz'
    with np.load(basis_path) as stored:
        if not np.allclose(stored['basis'][:,0],pca['basis'][:,0],atol=1e-4,rtol=1e-5): raise ValueError('recomputed PC1 basis differs from recorded basis')
    crop,states,images=load_selected_volumes(a.audit_root/'04_dvf'/prefix,'SAX')
    for state in ('low','mid','high'):
        meta=json.loads((a.audit_root/'04_dvf'/prefix/'volumes'/f'SAX_{state}_pc1.json').read_text())
        if meta.get('checkpoint_sha256') != digest: raise ValueError('state volume checkpoint mismatch')
    if any(int(r['checkpoint_step'])!=step for r in states.values()): raise ValueError('selected state step mismatch')
    anchor=states['low']['slice_id']; index=next((i for i,(v,s,_) in enumerate(base['cohort']) if v=='SAX' and str(s)==anchor),None)
    if index is None: raise ValueError('SAX anchor is missing from cohort')
    anchor_rows=sorted((r for r in rows if r['view']=='SAX' and r['slice_id']==anchor),key=lambda r:(r['timestamp_s'],r['dynamic_frame_id']))
    frames=base['cohort'][index][2]; mapping={int(o.dynamic_frame_id):m for o,m in zip(frames,means[index])}
    anchor_means=np.stack([mapping[r['dynamic_frame_id']] for r in anchor_rows]); anchor_scores=np.array([r['pc1'] for r in anchor_rows])
    selected={k:int(v['dynamic_frame_id']) for k,v in states.items()}
    anchor_by_id={r['dynamic_frame_id']:r for r in anchor_rows}
    if any(v not in anchor_by_id for v in selected.values()): raise ValueError('selected state is missing from latent cohort')
    for state,frame_id in selected.items():
        row=anchor_by_id[frame_id]; stored=states[state]
        if abs(row['timestamp_s']-float(stored['timestamp_s']))>1e-6 or abs(row['pc1']-float(stored['pc1']))>1e-4: raise ValueError('selected state differs from latent cohort')
    return {'step':step,'checkpoint':checkpoint,'checkpoint_sha256':digest,'checkpoint_data':{'diagnostics':checkpoint_data.get('diagnostics',{})},'crop':crop,'states':states,'selected':selected,'images':images,'anchor':anchor,'anchor_rows':anchor_rows,'anchor_means':anchor_means,'anchor_scores':anchor_scores,'basis':pca['basis'][:,0],'pca':pca,'latent_score_sha256':latent['output_sha256'][prefix+'frame_latent_scores.csv'],'basis_sha256':hashlib.sha256(np.ascontiguousarray(pca['basis']).tobytes()).hexdigest()}


def load_base(a):
    bindings=json.loads((a.run_dir/'experiment_inputs.json').read_text())
    load_args=SimpleNamespace(device=a.device,**{name:Path(bindings[name]['path']) for name in ('manifest','qc_table','canonical_domain','static_manifest')})
    with diagnostic_rng_context(): config,provenance,model,cohort=load_inputs(load_args)
    static=json.loads(load_args.static_manifest.read_text()); volume=Path(static['outputs']['volume']['path'])
    if output_sha256(volume)!=provenance['hashes']['static_volume']: raise ValueError('static reference hash mismatch')
    box=load_cardiac_box_config(a.cardiac_box_config) if a.cardiac_box_config else None
    return {'config':config,'provenance':provenance,'model':model,'cohort':cohort,'static_volume':volume,'cardiac_box':box}


def report_stage(a):
    root=a.output_root; comparison=root/'comparisons'
    if comparison.exists() and not a.overwrite_stage: raise FileExistsError('report output exists; pass --overwrite-stage')
    manifests=[]
    for stage in ('pure_pc1','motion_decomposition','residual'):
        path=root/f'stage_manifest_{stage}.json'
        if not path.exists(): raise FileNotFoundError(f'run --stage {stage} before report')
        manifest=json.loads(path.read_text()); manifests.append(manifest)
        for name,digest in manifest['output_sha256'].items():
            if output_sha256(root/name)!=digest: raise ValueError('prior stage output hash mismatch: '+name)
    for manifest in manifests[1:]:
        if any(manifest[key]!=manifests[0][key] for key in ('checkpoint_sha256','input_sha256','cardiac_box_config_sha256')): raise ValueError('prior stage provenance mismatch')
    if set(manifests[0]['checkpoint_sha256']) != {str(s) for s in a.checkpoint_steps}: raise ValueError('report checkpoint list does not match prior stages')
    for step in a.checkpoint_steps:
        if output_sha256(a.run_dir/f'checkpoint_step_{step}.pt') != manifests[0]['checkpoint_sha256'][str(step)]: raise ValueError('checkpoint changed since prior stages')
    comparison.mkdir(parents=True,exist_ok=True)
    residual_path=root/'residual_audit'/'summary_by_checkpoint.json'
    if not residual_path.is_file(): raise FileNotFoundError('run --stage residual before report')
    residual=json.loads(residual_path.read_text()); rows=[]
    for step in sorted(a.checkpoint_steps):
        pure=root/f'step_{step}'/'pure_pc1'; motion=root/f'step_{step}'/'motion_decomposition'
        if not (pure/'pca_provenance.json').is_file() or not (motion/'frame_motion_decomposition.csv').is_file(): raise FileNotFoundError('run pure_pc1 and motion_decomposition for each checkpoint before report')
        pca=json.loads((pure/'pca_provenance.json').read_text()); nonpc=json.loads((pure/'non_pc1_summary.json').read_text())['anatomy_support']
        with (motion/'frame_motion_decomposition.csv').open(newline='') as f: data=[r for r in csv.DictReader(f) if r['region']=='anatomy_support']
        def mean(field): return float(np.mean([float(r[field]) for r in data]))
        def metric(model,region):
            item=next((r for r in residual if r['model']==model and r['region']==region),None)
            return None if item is None else item['mean_mse']
        static=metric('STATIC','whole_slice'); v1=metric(f'{step}_v1','whole_slice')
        stored=pca.get('stored_diagnostics',{}); baseline=stored.get('static_baseline_sampled_mse_full'); predicted=stored.get('diagnostic_sampled_reconstruction_mse_full')
        improvement=100*(baseline-predicted)/baseline if baseline and predicted is not None else None
        rows.append({'step':step,'pc1_variance_ratio':pca['explained_variance_ratios'][0],'rank95':pca['rank95'],'full_latent_reconstruction_improvement':improvement,'pure_pc1_vs_full_volume_rmse':nonpc['volume_rmse'],'pure_pc1_vs_full_dvf_rms':nonpc['dvf_rms_magnitude_mm'],'translation_rms':float(np.sqrt(np.mean([float(r['translation_magnitude_mm'])**2 for r in data]))),'nonrigid_rms':mean('residual_rms_mm'),'nonrigid_fraction':mean('nonrigid_fraction'),'jacobian_variation':mean('jacobian_mean_abs_deviation'),'V1_whole_slice_MSE':v1,'V2_whole_slice_MSE':metric(f'{step}_v2_time','whole_slice'),'V1_cardiac_box_MSE':metric(f'{step}_v1','cardiac_box'),'V2_cardiac_box_MSE':metric(f'{step}_v2_time','cardiac_box')})
    write_csv(comparison/'cardioresp_identity_summary.csv',rows); json_write(comparison/'cardioresp_identity_summary.json',{'rows':rows,'authority_checkpoint':None,'interpretation':'decision support only; no composite score'})
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,keys,title in zip(axes.flat,(('pc1_variance_ratio',),('pure_pc1_vs_full_dvf_rms','nonrigid_rms'),('translation_rms',),('V1_whole_slice_MSE','V2_whole_slice_MSE')),('PC1 variance','Non-PC1 / non-rigid (mm)','Translation RMS (mm)','Whole-slice MSE')):
        for key in keys: ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label=key)
        ax.set_title(title); ax.set_xlabel('Checkpoint step'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(comparison/'cardioresp_identity_comparison.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    (root/'audit_report.md').write_text("""# Cardiorespiratory identity audit

No authority checkpoint is automatically selected. PC1, image residuals and local non-rigid deformation are not respiratory or cardiac ground truth. V1 is target-code self-consistency; V2 is target-withheld prediction on frames seen in training.

* If pure PC1 retains cardiac-like contraction, PC1 may mix respiratory-like and cardiac-like information.
* If only full latent shows this motion, additional motion may be encoded outside PC1; this alone does not establish cardiac identity.
* Improvement in V1 but not V2 can indicate target-specific overfitting or nuisance absorption.
* Improvement in both V1 and V2, particularly in an explicitly admitted cardiac box, provides stronger evidence of reproducible dynamics but does not prove cardiac identity without an independent reference.

Cardiac box is an optional local coordinate box, not a segmentation.
""")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root',type=Path,required=True); parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-steps',type=int,nargs='+',default=[2000,5000,10000]); parser.add_argument('--device',default='cpu'); parser.add_argument('--chunk-size',type=int,default=4096)
    parser.add_argument('--stage',choices=['pure_pc1','motion_decomposition','residual','report','frequency_bands','all'],required=True); parser.add_argument('--cardiac-box-config',type=Path)
    parser.add_argument('--overwrite-stage',action='store_true')
    args=parser.parse_args(); args.device=torch.device(args.device); args.checkpoint_steps=sorted(args.checkpoint_steps)
    if args.chunk_size<=0 or not args.checkpoint_steps or len(set(args.checkpoint_steps))!=len(args.checkpoint_steps) or min(args.checkpoint_steps)<=0: parser.error('invalid chunk size or checkpoint list')
    args.output_root=args.audit_root/'07_cardioresp_identity'
    if args.stage=='frequency_bands':
        from cardioresp4d.experiments.simple4d_dual_band_periodicity import frequency_stage
        frequency_stage(args.audit_root,args.checkpoint_steps,overwrite=args.overwrite_stage)
        return
    if args.stage=='all':
        original=list(sys.argv)
        try:
            for stage in ('pure_pc1','motion_decomposition','residual','report'):
                current=list(original)
                for i,token in enumerate(current):
                    if token=='--stage': current[i+1]=stage; break
                    if token.startswith('--stage='): current[i]='--stage='+stage; break
                sys.argv=current; main()
        finally: sys.argv=original
        return
    if args.stage=='report': report_stage(args); return
    import shutil
    paths=[args.output_root/'residual_audit'] if args.stage=='residual' else [args.output_root/f'step_{step}'/args.stage for step in args.checkpoint_steps]
    if not args.overwrite_stage and any(path.exists() for path in paths): raise FileExistsError('stage output exists; pass --overwrite-stage')
    base=load_base(args)
    if base['cardiac_box'] is not None:
        coordinates=base['model'].forward_deformation.coordinates
        lower=coordinates.lower_world_mm.detach().cpu().numpy(); upper=lower+coordinates.extent_mm.detach().cpu().numpy()
        if np.any(base['cardiac_box'].min_mm < lower-1e-6) or np.any(base['cardiac_box'].max_mm > upper+1e-6): raise ValueError('user-supplied cardiac box lies outside the admitted model domain')
    hashes={step:output_sha256(args.run_dir/f'checkpoint_step_{step}.pt') for step in args.checkpoint_steps}
    infos={}
    for step in args.checkpoint_steps:
        info=prepare_step(args,step,base); infos[step]=info
        coords=base['model'].forward_deformation.coordinates
        lo=coords.lower_world_mm.detach().cpu().numpy(); hi=lo+coords.extent_mm.detach().cpu().numpy()
        computed,support,ras_to_voxel=derive_anatomy_crop(base['static_volume'],lo,hi,coords.reference_spacing_mm.detach().cpu().numpy(),source_sha256=base['provenance']['hashes']['static_volume'])
        if computed != info['crop']: raise ValueError('admitted anatomy crop differs from stored DVF crop')
        info['support']=support; info['ras_to_voxel']=ras_to_voxel
    for path in paths:
        if path.exists(): shutil.rmtree(path)
    args.output_root.mkdir(parents=True,exist_ok=True)
    if args.stage in ('pure_pc1','motion_decomposition'):
        for step in args.checkpoint_steps:
            info=infos[step]; load_checkpoint(info['checkpoint'],step,base['config'],base['provenance'],base['model'])
            (pure_stage if args.stage=='pure_pc1' else motion_stage)(args,base,info,info['support'],info['ras_to_voxel'])
    else:
        def switch(step): load_checkpoint(infos[step]['checkpoint'],step,base['config'],base['provenance'],base['model'])
        residual_stage(args,base,args.checkpoint_steps,switch)
    if any(output_sha256(infos[s]['checkpoint'])!=hashes[s] for s in args.checkpoint_steps): raise RuntimeError('checkpoint changed during audit')
    import subprocess
    files=sorted(p for path in paths for p in path.rglob('*') if p.is_file())
    manifest={'stage':args.stage,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checkpoint_sha256':{str(s):hashes[s] for s in args.checkpoint_steps},'input_sha256':base['provenance']['hashes'],'cardiac_box':base['cardiac_box'].to_dict() if base['cardiac_box'] else None,'cardiac_box_config_sha256':output_sha256(args.cardiac_box_config) if args.cardiac_box_config else None,'output_sha256':{str(p.relative_to(args.output_root)):output_sha256(p) for p in files},'analysis_policy':'read-only, posterior mean; PC1/non-rigid motion are not cardiac ground truth'}
    json_write(args.output_root/f'stage_manifest_{args.stage}.json',manifest)
    json_write(args.output_root/f'stage_report_{args.stage}.json',{'status':'complete','stage':args.stage,'cardiac_box_available':base['cardiac_box'] is not None})
    stage_files=sorted(args.output_root.glob('stage_manifest_*.json'))
    json_write(args.output_root/'stage_manifest.json',{'stages':{p.stem.removeprefix('stage_manifest_'):output_sha256(p) for p in stage_files},'analysis_policy':'no authority checkpoint automatically assigned'})
    json_write(args.output_root/'stage_report.json',{'completed_stages':[p.stem.removeprefix('stage_manifest_') for p in stage_files]})


if __name__=='__main__': main()
