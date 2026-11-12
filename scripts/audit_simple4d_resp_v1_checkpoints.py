#!/usr/bin/env python3
"""Read-only SIMPLE-4D respiratory checkpoint audit (summary and latent stages)."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from train_respiratory_s2vdreme_reference_v1 import _load_config
from train_respiratory_reference_gauge_exp2a import _build_exact_model
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import apply_corrected_static_poses, derive_effective_external_domain, load_corrected_pose_map, deterministic_diagnostic_cohort, diagnostic_rng_context
from cardioresp4d.experiments.simple4d_resp_v1_r0 import compatible_resume_provenance, resolve_static_reference_spacing, sha256, state_digest, validate_config
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import global_latent_pca
from cardioresp4d.experiments.simple4d_resp_v1_identity_stages import read_latent, surrogate_stage, coherence_stage
from cardioresp4d.experiments.simple4d_resp_v1_identity_render import dvf_stage, withheld_stage
from cardioresp4d.experiments.simple4d_resp_v1_anatomy import derive_anatomy_crop, anatomy_overlay_stage, compare_anatomy_sax
from cardioresp4d.experiments.simple4d_resp_v1_identity_report import comparison_report
from cardioresp4d.simple4d.model import Simple4DRespiratoryModel
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.visualization.simple4d_resp_v1_identity import save_summary_figures, save_latent_figures, save_periodicity_figures


SUMMARY_FIELDS = ('latent_mu_total_rms','latent_mu_static_rms','latent_mu_dynamic_rms','latent_mu_static_dynamic_ratio','latent_dynamic_pc1_explained_variance_ratio','latent_dynamic_pc2_explained_variance_ratio','latent_dynamic_rank_95pct','posterior_std_mean','kl_mean','respiratory_dvf_total_rms_mm','respiratory_dvf_static_rms_mm','respiratory_dvf_dynamic_rms_mm','respiratory_dvf_static_dynamic_ratio','respiratory_dvf_max_abs_mm','static_baseline_sampled_mse_full','diagnostic_sampled_reconstruction_mse_full','static_baseline_sampled_mae_full','diagnostic_sampled_reconstruction_mae_full','jacobian_det_p01','jacobian_det_min','jacobian_det_nonpositive_fraction','cycle_error_mean_mm','cycle_error_p95_mm','cycle_error_max_mm')


def load_inputs(args):
    config = _load_config(ROOT/'configs/simple4d_resp_v1.yaml'); validate_config(config)
    static = json.loads(args.static_manifest.read_text())
    paths = {'manifest':args.manifest,'qc_table':args.qc_table,'canonical_domain':args.canonical_domain,'static_manifest':args.static_manifest,'static_model':Path(static['outputs']['model']['path']),'static_volume':Path(static['outputs']['volume']['path']),'corrected_pose_manifest':Path(static['corrected_pose_manifest']['path']),'simple4d_vendor_manifest':ROOT/'third_party/simple4d_VENDOR.json','bootstrap_config':ROOT/'configs'/config['canonical']['bootstrap_config']}
    hashes = {key:sha256(path) for key,path in paths.items()}
    if static.get('source_manifest_sha256') != hashes['manifest'] or static.get('qc_table_sha256') != hashes['qc_table']:
        raise ValueError('manifest/QC do not match static provenance')
    for artifact, name in ((static['outputs']['model'],'static_model'),(static['outputs']['volume'],'static_volume'),(static['corrected_pose_manifest'],'corrected_pose_manifest')):
        if artifact.get('sha256') != hashes[name]: raise ValueError('static artifact hash mismatch: '+name)
    vendor = json.loads(paths['simple4d_vendor_manifest'].read_text())
    if vendor.get('commit') != 'c466b276a438b3e052ad3b44d38f398872d8aeae': raise ValueError('vendor commit mismatch')
    for name,digest in vendor['key_files'].items():
        if sha256(ROOT/'third_party/simple4d'/name) != digest: raise ValueError('vendor source hash mismatch: '+name)
    spacing = resolve_static_reference_spacing(static,config['deformation']['expected_reference_spacing_mm'])
    config_sha = hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    provenance = {'hashes':hashes,'effective_config_sha256':config_sha,'upstream_commit':vendor['commit'],'spacing':spacing,'model_seed':6,'sampler_seed':12}
    domain=json.loads(args.canonical_domain.read_text()); poses=load_corrected_pose_map(paths['corrected_pose_manifest'])
    effective=derive_effective_external_domain(poses,domain['world_min_mm'],domain['world_max_mm'])
    box=effective['model_float32_effective_domain']; model_domain={**domain,'world_min_mm':box['lower_lps_mm'],'world_max_mm':box['upper_lps_mm']}
    observations,_,_=observations_from_manifest(args.manifest,args.qc_table,args.device,normalization_mode='per_series',assign_slice_positions=True)
    observations=apply_corrected_static_poses(observations,poses)
    payload=torch.load(paths['static_model'],map_location='cpu',weights_only=False)
    source,_,_=_build_exact_model(_load_config(paths['bootstrap_config']),model_domain,payload,observations,args.device)
    model=Simple4DRespiratoryModel(source.canonical,box['lower_lps_mm'],box['upper_lps_mm'],spacing['resolved_reference_spacing_mm'],canonical_spacing_mm=spacing['static_nifti_spacing_mm'],psf_samples=8).to(args.device)
    model.freeze_r0(); del source,payload
    return config,provenance,model,deterministic_diagnostic_cohort(ViewLocationBalancedSampler(observations,seed=12))


def load_checkpoint(path, step, config, provenance, model):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('checkpoint_schema') != 1 or checkpoint.get('global_step') != step or checkpoint.get('runner_state',{}).get('global_step') != step:
        raise ValueError('checkpoint schema or step mismatch')
    if not compatible_resume_provenance(checkpoint,provenance,config): raise ValueError('checkpoint provenance mismatch')
    canonical_digest=state_digest(model.canonical)
    state=checkpoint['model']
    expected=model.state_dict()
    if state.keys() != expected.keys() or any(state[key].shape != expected[key].shape for key in state): raise ValueError('checkpoint architecture mismatch')
    if any(not torch.equal(state[key].cpu(),expected[key].cpu()) for key in expected if key.startswith('canonical.')):
        raise ValueError('checkpoint canonical differs from admitted static canonical')
    model.load_state_dict(state,strict=True)
    if state_digest(model.canonical) != canonical_digest: raise ValueError('canonical unexpectedly changed')
    return checkpoint


def output_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','canonical-domain','static-manifest','run-dir','output-dir'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint-steps',type=int,nargs='+',default=[2000,5000,10000]); parser.add_argument('--device',default='cpu'); parser.add_argument('--stage',choices=['summary','latent','surrogate','coherence','dvf','withheld','report','all'],required=True); parser.add_argument('--overwrite-stage',action='store_true'); parser.add_argument('--chunk-size',type=int,default=4096)
    args=parser.parse_args(); args.device=torch.device(args.device)
    if args.chunk_size <= 0: parser.error('chunk-size must be positive')
    if len(set(args.checkpoint_steps)) != len(args.checkpoint_steps) or any(step<=0 for step in args.checkpoint_steps): parser.error('checkpoint steps must be unique positive integers')
    if args.stage == 'report':
        comparison_report(args.output_dir, sorted(args.checkpoint_steps), overwrite=args.overwrite_stage)
        return
    if args.stage == 'all':
        original = list(sys.argv)
        try:
            for stage in ('summary','latent','surrogate','coherence','dvf','withheld','report'):
                current = list(original)
                for index, token in enumerate(current):
                    if token == '--stage': current[index+1] = stage; break
                    if token.startswith('--stage='): current[index] = '--stage='+stage; break
                sys.argv = current
                main()
        finally:
            sys.argv = original
        return
    checkpoints={step:args.run_dir/f'checkpoint_step_{step}.pt' for step in sorted(args.checkpoint_steps)}
    for path in checkpoints.values():
        if not path.is_file(): raise FileNotFoundError(path)
    checkpoint_hashes_before={step:output_sha256(path) for step,path in checkpoints.items()}
    out=args.output_dir/({'summary':'00_summary','latent':'01_latent','surrogate':'02_surrogate','coherence':'03_coherence','dvf':'04_dvf','withheld':'05_withheld'}[args.stage])
    if out.exists() and not args.overwrite_stage: raise FileExistsError(f'{out} exists; pass --overwrite-stage')
    with diagnostic_rng_context():
        config,provenance,model,cohort=load_inputs(args)
    if args.stage in ('surrogate','coherence','dvf','withheld'):
        latent_dir=args.output_dir/'01_latent'
        if not (latent_dir/'stage_manifest.json').exists():
            raise FileNotFoundError('run --stage latent first')
        latent_manifest=json.loads((latent_dir/'stage_manifest.json').read_text())
        if latent_manifest['input_artifact_sha256'] != provenance['hashes']:
            raise ValueError('latent stage input provenance mismatch')
        all_rows={}
        for step,path in checkpoints.items():
            recorded=latent_manifest['checkpoints'].get(str(step),{})
            if recorded.get('sha256') != sha256(path): raise ValueError('latent checkpoint provenance mismatch')
            load_checkpoint(path,step,config,provenance,model)
            relative=f'step_{step}/frame_latent_scores.csv'
            latent_file=latent_dir/relative
            if latent_manifest.get('output_sha256',{}).get(relative) != sha256(latent_file): raise ValueError('latent scores hash mismatch')
            all_rows[step]=read_latent(latent_file,cohort)
        crop_data=None
        if args.stage=='dvf':
            static=json.loads(args.static_manifest.read_text()); volume=Path(static['outputs']['volume']['path'])
            if output_sha256(volume) != provenance['hashes']['static_volume']: raise ValueError('static volume changed before anatomy crop')
            coords=model.forward_deformation.coordinates
            lower=coords.lower_world_mm.detach().cpu().numpy(); upper=lower+coords.extent_mm.detach().cpu().numpy()
            crop_data=derive_anatomy_crop(volume,lower,upper,coords.reference_spacing_mm.detach().cpu().numpy(),source_sha256=provenance['hashes']['static_volume'])
        if out.exists(): shutil.rmtree(out)
        out.mkdir(parents=True)
        for step,rows in all_rows.items():
            dest=out/f'step_{step}'
            if args.stage in ('surrogate','coherence'):
                (surrogate_stage if args.stage=='surrogate' else coherence_stage)(rows,dest)
            else:
                load_checkpoint(checkpoints[step],step,config,provenance,model)
                if args.stage=='dvf':
                    dvf_stage(model,rows,dest,step,checkpoint_hashes_before[step],args.chunk_size)
                    anatomy_overlay_stage(model,rows,dest,step,checkpoint_hashes_before[step],args.chunk_size,*crop_data)
                else: withheld_stage(model,rows,dest,step,args.chunk_size)
        if args.stage == 'dvf': compare_anatomy_sax(out, sorted(checkpoints))
        if any(output_sha256(path)!=checkpoint_hashes_before[step] for step,path in checkpoints.items()): raise RuntimeError('checkpoint bytes changed during analysis')
        files=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
        manifest={'stage':args.stage,'timestamp':datetime.now(timezone.utc).isoformat(),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checkpoints':{str(step):{'path':str(path.resolve()),'sha256':sha256(path)} for step,path in checkpoints.items()},'input_artifact_sha256':provenance['hashes'],'output_files':files+['stage_manifest.json','stage_report.json'],'output_sha256':{name:output_sha256(out/name) for name in files},'analysis_policy':'posterior mean; image-derived surrogate is not respiratory ground truth; mutual nearest temporal pairs'}
        (out/'stage_manifest.json').write_text(json.dumps(manifest,indent=2)); (out/'stage_report.json').write_text(json.dumps({'stage':args.stage,'policy':manifest['analysis_policy']},indent=2))
        return
    data=[]
    for step,path in checkpoints.items():
        cp=load_checkpoint(path,step,config,provenance,model)
        diag=cp['diagnostics'][f'step_{step}']
        if args.stage=='summary':
            row={'step':step,**{field:diag[field] for field in SUMMARY_FIELDS}}
            baseline=row['static_baseline_sampled_mse_full']; row['motion_vs_static_mse_improvement_percent']=100*(baseline-row['diagnostic_sampled_reconstruction_mse_full'])/baseline if baseline else None
            data.append(row)
        else:
            groups=[]
            with torch.no_grad():
                for _,_,frames in cohort:
                    groups.append(np.stack([model.encode_observation(frame,sample=False)['mu'].squeeze(0).cpu().numpy() for frame in frames]))
            result=global_latent_pca(groups); ratios=result['explained_variance_ratios']
            diffs={'pc1':float(ratios[0]-diag['latent_dynamic_pc1_explained_variance_ratio']), 'pc2':float(ratios[1]-diag['latent_dynamic_pc2_explained_variance_ratio']), 'dynamic_rms':result['latent_dynamic_rms']-diag['latent_mu_dynamic_rms'], 'rank95':result['rank95']-diag['latent_dynamic_rank_95pct']}
            if abs(diffs['pc1'])>1e-4 or abs(diffs['pc2'])>1e-4 or abs(diffs['dynamic_rms'])>1e-4 or diffs['rank95'] != 0: raise ValueError(f'PCA diagnostics mismatch at step {step}: {diffs}')
            rows=[]
            for (view,slice_id,frames),mu,scores in zip(cohort,groups,result['scores']):
                std=scores.std(axis=0)
                for frame,vector,score in zip(frames,mu,scores):
                    rows.append({'checkpoint_step':step,'view':view,'slice_id':slice_id,'dynamic_frame_id':frame.dynamic_frame_id,'timestamp_s':frame.timestamp_s,'normalized_slice_position':frame.normalized_slice_position,'pc1':float(score[0]),'pc2':float(score[1]),'pc3':float(score[2]),'pc1_location_z':float((score[0]-scores[:,0].mean())/std[0]) if std[0]>1e-12 else None,'pc2_location_z':float((score[1]-scores[:,1].mean())/std[1]) if std[1]>1e-12 else None,'pc1_low_variance':bool(std[0]<=1e-12),'pc2_low_variance':bool(std[1]<=1e-12),'mu_norm':float(np.linalg.norm(vector))})
            data.append((step,result,diffs,rows))
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    if args.stage=='summary':
        write_csv(out/'checkpoint_summary.csv',data); (out/'checkpoint_summary.json').write_text(json.dumps(data,indent=2,allow_nan=False)); save_summary_figures(data,out)
    else:
        for step,result,diffs,rows in data:
            dest=out/f'step_{step}'; dest.mkdir(); write_csv(dest/'frame_latent_scores.csv',rows)
            np.savez_compressed(dest/'pca_basis.npz',basis=result['basis'],eigenvalues=result['eigenvalues'],explained_variance_ratios=result['explained_variance_ratios'])
            (dest/'pca_summary.json').write_text(json.dumps({'rank95':result['rank95'],'eigenvalues':result['eigenvalues'].tolist(),'explained_variance_ratios':result['explained_variance_ratios'].tolist(),'latent_dynamic_rms':result['latent_dynamic_rms'],'audit_recomputed_minus_checkpoint':diffs},indent=2,allow_nan=False))
            save_latent_figures(rows,result['explained_variance_ratios'],dest)
            save_periodicity_figures(rows,dest)
    if any(output_sha256(path)!=checkpoint_hashes_before[step] for step,path in checkpoints.items()): raise RuntimeError('checkpoint bytes changed during analysis')
    files=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
    manifest={'stage':args.stage,'timestamp':datetime.now(timezone.utc).isoformat(),'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checkpoints':{str(step):{'path':str(path.resolve()),'sha256':sha256(path)} for step,path in checkpoints.items()},'input_artifact_sha256':provenance['hashes'],'output_files':files+['stage_manifest.json','stage_report.json'],'output_sha256':{name:output_sha256(out/name) for name in files},'analysis_policy':'read-only posterior mean; no respiratory ground truth'}
    (out/'stage_manifest.json').write_text(json.dumps(manifest,indent=2)); (out/'stage_report.json').write_text(json.dumps({'stage':args.stage,'checkpoints':list(checkpoints),'policy':manifest['analysis_policy']},indent=2))


if __name__=='__main__': main()
