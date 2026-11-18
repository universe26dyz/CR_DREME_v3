#!/usr/bin/env python3
"""Joint-from-scratch S1 using upstream networks and physical multi-view PSF."""
import argparse
import csv
import json
import sys
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.simple4d_resp_v1_r0 import sha256,resolve_static_reference_spacing,diagnostic_metrics
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map,apply_corrected_static_poses,derive_effective_external_domain
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiView,S1Runner,validate_s1_resume
from cardioresp4d.training.runtime_state import set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','canonical-domain','static-manifest','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--experiment-config',type=Path,default=ROOT/'configs/simple4d_core_mv_v1.yaml'); p.add_argument('--device',default='cuda:0'); p.add_argument('--backend',choices=('tcnn','pytorch_fallback'),default='tcnn'); p.add_argument('--resume',type=Path)
    target=p.add_mutually_exclusive_group(required=True)
    target.add_argument('--stop-after',choices=('smoke','diagnostic','mid','formal'))
    target.add_argument('--target-step',type=int,choices=range(1,20001),metavar='1..20000')
    return p


def write_extension_summary(output,diagnostics,metrics,scheduler):
    """Write boundary diagnostics; LR columns are before and after each update."""
    fields=['diagnostic_sampled_reconstruction_mse_full','diagnostic_sampled_reconstruction_mae_full','current_canonical_no_motion_sampled_mse_full','latent_mu_dynamic_rms','latent_dynamic_pc1_explained_variance_ratio','latent_dynamic_rank_95pct','respiratory_dvf_dynamic_rms_mm','respiratory_dvf_static_rms_mm','cycle_error_mean_mm','cycle_error_p95_mm','jacobian_det_p01','jacobian_det_nonpositive_fraction']
    groups=('hash','template','encoder','forward','inverse')
    metric_by_step={row['global_step']:row for row in metrics}
    rows=[]
    for step in (5000,10000,15000,20000):
        diag=diagnostics.get(f'step_{step}')
        if diag is None: continue
        row={'step':step,**{name:diag.get(name) for name in fields}}
        metric=metric_by_step.get(step,{})
        for group in groups:
            value=metric.get('lr_'+group)
            row['lr_'+group+'_before_update']=value
            row['lr_'+group+'_after_scheduler']=value*scheduler.gamma if value is not None and step%scheduler.step_size==0 else value
        rows.append(row)
    if rows:
        with (output/'s1_extension_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def train_to_target(runner,model,sampler,target,save_steps,diagnostics,output,provenance,cfg,normalization,groups):
    if runner.global_step>target: raise ValueError('checkpoint step exceeds target')
    while runner.global_step<target:
        runner.update()
        if runner.global_step in save_steps:
            diag=diagnostic_metrics(model,sampler)
            for key in list(diag):
                if key.startswith('static_baseline_'): diag['current_canonical_no_motion_'+key[len('static_baseline_'):]]=diag.pop(key)
            diagnostics[f'step_{runner.global_step}']=diag
            cp={'schema':'simple4d_s1_v1','provenance':provenance,'runner':runner.state_dict(),'diagnostics':diagnostics,'config':cfg,'normalization_parameters':normalization,'normalization_groups':groups}
            torch.save(cp,output/f'checkpoint_step_{runner.global_step}.pt'); torch.save(cp,output/'checkpoint_last.pt')
            (output/'training_report.json').write_text(json.dumps({'global_step':runner.global_step,'provenance':provenance,'diagnostics':diagnostics},indent=2,allow_nan=False)+'\n')
            with (output/'training_metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(runner.metrics[0])); writer.writeheader(); writer.writerows(runner.metrics)
            write_extension_summary(output,diagnostics,runner.metrics,runner.scheduler)
    write_extension_summary(output,diagnostics,runner.metrics,runner.scheduler)


def main():
    a=build_parser().parse_args(); device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    cfg=yaml.safe_load(a.experiment_config.read_text()); canonical=cfg['canonical']; data=cfg['data']; deform=cfg['deformation']
    if cfg['experiment']!={'name':'simple4d_core_mv_v1','scope':'SAX_2CH_4CH','initialization':'from_scratch'} or cfg['encoder']!={'type':'MotionCodeVAE','code_dim':32,'view_embedding':False} or cfg['boundaries']!={'smoke':200,'diagnostic':2000,'mid':5000,'formal':10000} or data!={'normalization':'per_series','observations_per_update':8,'pixel_samples_per_observation':10000}: raise ValueError('S1 protocol mismatch')
    if cfg['optimizer']!={'hash_lr':.01,'template_lr':2e-5,'encoder_lr':5e-5,'forward_lr':5e-5,'inverse_lr':5e-5,'step_size':5000,'gamma':.5} or cfg['loss']!={'photometric':'l1','kl':1e-3,'jacobian':1e-4,'magnitude':1e-2,'cycle':.1} or cfg['reproducibility']!={'model_seed':6,'sampler_seed':12}: raise ValueError('S1 optimization protocol mismatch')
    if canonical!={'type':'InstantNGPINR','ngp_levels':16,'ngp_features_per_level':2,'ngp_log2_hashmap_size':19,'ngp_base_resolution':16,'ngp_max_resolution':512,'ngp_mlp':[64,64],'required_backend':'tcnn'} or deform!={'type':'DeformCodeAdaptive','hidden_dims':[256,256,256,256],'omega':7,'bidirectional':True,'expected_reference_spacing_mm':[1.,1.,1.]} or cfg['renderer']!={'type':'nesvor_psf','psf_samples':8}: raise ValueError('S1 architecture protocol mismatch')
    static=json.loads(a.static_manifest.read_text()); pose=Path(static['corrected_pose_manifest']['path']); volume=Path(static['outputs']['volume']['path']); vendor_path=ROOT/'third_party/simple4d_VENDOR.json'; vendor=json.loads(vendor_path.read_text())
    if static.get('source_manifest_sha256')!=sha256(a.manifest) or static.get('qc_table_sha256')!=sha256(a.qc_table) or static['corrected_pose_manifest'].get('sha256')!=sha256(pose) or static['outputs']['volume'].get('sha256')!=sha256(volume): raise ValueError('static geometry provenance mismatch')
    if vendor['commit']!='c466b276a438b3e052ad3b44d38f398872d8aeae' or any(sha256(ROOT/'third_party/simple4d'/name)!=digest for name,digest in vendor['key_files'].items()): raise ValueError('vendor provenance mismatch')
    spacing=resolve_static_reference_spacing(static,deform['expected_reference_spacing_mm']); poses=load_corrected_pose_map(pose); domain=json.loads(a.canonical_domain.read_text()); effective=derive_effective_external_domain(poses,domain['world_min_mm'],domain['world_max_mm'])['model_float32_effective_domain']
    observations,normalization,groups=observations_from_manifest(a.manifest,a.qc_table,device,normalization_mode='per_series',assign_slice_positions=True); observations=apply_corrected_static_poses(observations,poses)
    if {obs.view.upper() for obs in observations if obs.qc_valid}!={'SAX','2CH','4CH'}: raise ValueError('S1 requires admitted SAX, 2CH and 4CH observations')
    set_reproducibility(6)
    model=Simple4DCoreMultiView(effective['lower_lps_mm'],effective['upper_lps_mm'],spacing['resolved_reference_spacing_mm'],use_tcnn=a.backend=='tcnn',require_tcnn=a.backend=='tcnn').to(device)
    sampler=ViewLocationBalancedSampler(observations,seed=12); runner=S1Runner(model,sampler)
    provenance={'manifest':sha256(a.manifest),'qc_table':sha256(a.qc_table),'canonical_domain':sha256(a.canonical_domain),'static_manifest':sha256(a.static_manifest),'static_volume':sha256(volume),'corrected_poses':sha256(pose),'vendor_manifest':sha256(vendor_path),'config':sha256(a.experiment_config),'backend':model.canonical.backend,'spacing':spacing,'effective_domain':effective,'initialization':'from_scratch_no_nesvor_model_weights'}
    if a.resume:
        last=a.output_dir/'checkpoint_last.pt'
        if a.output_dir.exists() and any(a.output_dir.iterdir()):
            if not last.is_file() or torch.load(last,map_location='cpu',weights_only=False).get('provenance')!=provenance: raise ValueError('existing S1 output directory provenance mismatch')
        cp=torch.load(a.resume,map_location='cpu',weights_only=False)
        validate_s1_resume(cp,provenance)
        runner.load_state_dict(cp['runner']); diagnostics=cp['diagnostics']
    else:
        if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError('fresh S1 output must be empty')
        diagnostics={}
    target=cfg['boundaries'][a.stop_after] if a.stop_after else a.target_step
    if runner.global_step>target: raise ValueError('checkpoint step exceeds target')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'effective_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (a.output_dir/'experiment_inputs.json').write_text(json.dumps({'paths':{key:str(getattr(a,key).resolve()) for key in ('manifest','qc_table','canonical_domain','static_manifest','experiment_config')},'provenance':provenance},indent=2)+'\n')
    save_steps=set(cfg['boundaries'].values())|{15000,20000,target}
    train_to_target(runner,model,sampler,target,save_steps,diagnostics,a.output_dir,provenance,cfg,normalization,groups)
    print(json.dumps({'step':runner.global_step,'backend':model.canonical.backend}))


if __name__=='__main__': main()
