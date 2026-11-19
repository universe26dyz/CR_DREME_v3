#!/usr/bin/env python3
"""Joint-from-scratch S05 using physical five-point thick-slice quadrature."""
import argparse
import csv
import math
import json
import sys
import subprocess
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.simple4d_resp_v1_r0 import sha256,resolve_static_reference_spacing,diagnostic_metrics
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map,apply_corrected_static_poses,derive_effective_external_domain
from cardioresp4d.experiments.simple4d_disentangle_freq import DualLatentThickSlice,FrequencyRunner,latent_diagnostics,intervention_diagnostics,validate_enabled_bands,band_residual
from cardioresp4d.experiments.simple4d_core_mv import validate_s1_resume
from cardioresp4d.training.runtime_state import set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler,proportional_view_counts


def main(*, sax_only=False, budget_matched=False, equal_budget=False, slice_proportional=True):
    if sum(bool(x) for x in (sax_only, budget_matched, equal_budget, slice_proportional)) > 1:
        raise ValueError('incompatible experiment modes')
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','canonical-domain','static-manifest','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--temporal-audit',type=Path,required=True); modes=p.add_mutually_exclusive_group(); modes.add_argument('--architecture-control',action='store_true'); modes.add_argument('--card-only',action='store_true'); modes.add_argument('--dual-band',action='store_true'); p.add_argument('--basis-audit',type=Path); p.add_argument('--experiment-config',type=Path,default=ROOT/('configs/simple4d_core_mv_thickslice_disentangle_freq_v1.yaml' if slice_proportional else 'configs/simple4d_core_mv_thickslice_equalbudget_v1.yaml' if equal_budget else 'configs/simple4d_core_mv_thickslice_budgetmatched_v1.yaml' if budget_matched else 'configs/simple4d_core_sax_thickslice_v1.yaml' if sax_only else 'configs/simple4d_core_mv_thickslice_v1.yaml')); p.add_argument('--device',default='cuda:0'); p.add_argument('--backend',choices=('tcnn','pytorch_fallback'),default='tcnn'); p.add_argument('--stop-after',choices=('smoke','diagnostic','mid','formal'),required=True); p.add_argument('--resume',type=Path)
    a=p.parse_args(); device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    cfg=yaml.safe_load(a.experiment_config.read_text()); canonical=cfg['canonical']; data=cfg['data']; deform=cfg['deformation']
    if cfg['experiment']!=({'name':'simple4d_disentangle_freq_v1','scope':'SAX_2CH_4CH','initialization':'from_scratch','control':'slice_proportional_fixed_budget'} if slice_proportional else {'name':'simple4d_core_mv_thickslice_equalbudget_v1','scope':'SAX_2CH_4CH','initialization':'from_scratch','control':'equal_view_budget'} if equal_budget else {'name':'simple4d_core_mv_thickslice_budgetmatched_v1','scope':'SAX_2CH_4CH','initialization':'from_scratch','control':'matched_sax_budget'} if budget_matched else {'name':'simple4d_core_sax_thickslice_v1','scope':'SAX','initialization':'from_scratch'} if sax_only else {'name':'simple4d_core_mv_v1','scope':'SAX_2CH_4CH','initialization':'from_scratch'}) or cfg['encoder']!={'type':'FrequencyDisentangledMotionCodeVAE','code_dim':32,'resp_dim':16,'card_dim':16,'view_embedding':False} or cfg['boundaries']!={'smoke':200,'diagnostic':2000,'mid':5000,'formal':10000} or data!=({'normalization':'per_series','pixel_samples_per_observation':10000,'sampling_mode':'location_proportional_fixed_budget','observations_per_update':24} if slice_proportional else {'normalization':'per_series','pixel_samples_per_observation':10000,'view_counts':{'SAX':8,'2CH':8,'4CH':8}} if equal_budget else {'normalization':'per_series','pixel_samples_per_observation':10000,'view_counts':{'SAX':8,'2CH':4,'4CH':4}} if budget_matched else {'normalization':'per_series','observations_per_update':8,'pixel_samples_per_observation':10000}): raise ValueError('S1 protocol mismatch')
    if cfg['optimizer']!={'hash_lr':.01,'template_lr':2e-5,'encoder_lr':5e-5,'forward_lr':5e-5,'inverse_lr':5e-5,'step_size':5000,'gamma':.5} or cfg['loss']!={'photometric':'l1','kl':1e-3,'jacobian':1e-4,'magnitude':1e-2,'cycle':.1} or cfg['reproducibility']!={'model_seed':6,'sampler_seed':12}: raise ValueError('S1 optimization protocol mismatch')
    if canonical!={'type':'InstantNGPINR','ngp_levels':16,'ngp_features_per_level':2,'ngp_log2_hashmap_size':19,'ngp_base_resolution':16,'ngp_max_resolution':512,'ngp_mlp':[64,64],'required_backend':'tcnn'} or deform!={'type':'DeformCodeAdaptive','hidden_dims':[256,256,256,256],'omega':7,'bidirectional':True,'expected_reference_spacing_mm':[1.,1.,1.]} or cfg['renderer']!={'type':'upstream_like_physical_thick_slice_5point'}: raise ValueError('S1 architecture protocol mismatch')
    freq=cfg['frequency_disentanglement']; audit=json.loads(a.temporal_audit.read_text())
    if audit.get('input_sha256') != {'manifest':sha256(a.manifest),'qc_table':sha256(a.qc_table)}: raise ValueError('temporal audit inputs do not match training inputs')
    if a.architecture_control: freq={**freq,'enabled':False,'lambda_resp_freq':0.,'lambda_card_freq':0.}
    if a.card_only: freq={**freq,'enabled':True,'lambda_resp_freq':0.,'lambda_card_freq':0.001}
    if a.dual_band: freq={**freq,'enabled':True,'lambda_resp_freq':0.001,'lambda_card_freq':0.001,'resp_observability_mode':'observable_basis_subset'}
    cfg['frequency_disentanglement']=freq
    validate_enabled_bands(audit,freq)
    if a.stop_after not in ('smoke','diagnostic'): raise ValueError('only smoke and 2k diagnostic permitted')
    paired=a.stop_after=='diagnostic'
    if paired and (not a.resume or not a.basis_audit or not (a.architecture_control or a.dual_band)): raise ValueError('2k requires shared parent, basis audit, and explicit control or dual band')
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError('output directory must be empty')
    allowed=Path('/data2/dengyz/dataset/CR_DREME_v3/exp/simple4d_disentangle_freq_v1').resolve()
    if not a.output_dir.resolve().is_relative_to(allowed): raise ValueError('output must be under experiment subtree')
    static=json.loads(a.static_manifest.read_text()); pose=Path(static['corrected_pose_manifest']['path']); volume=Path(static['outputs']['volume']['path']); vendor_path=ROOT/'third_party/simple4d_VENDOR.json'; vendor=json.loads(vendor_path.read_text())
    if static.get('source_manifest_sha256')!=sha256(a.manifest) or static.get('qc_table_sha256')!=sha256(a.qc_table) or static['corrected_pose_manifest'].get('sha256')!=sha256(pose) or static['outputs']['volume'].get('sha256')!=sha256(volume): raise ValueError('static geometry provenance mismatch')
    if vendor['commit']!='c466b276a438b3e052ad3b44d38f398872d8aeae' or any(sha256(ROOT/'third_party/simple4d'/name)!=digest for name,digest in vendor['key_files'].items()): raise ValueError('vendor provenance mismatch')
    spacing=resolve_static_reference_spacing(static,deform['expected_reference_spacing_mm']); poses=load_corrected_pose_map(pose); domain=json.loads(a.canonical_domain.read_text()); effective=derive_effective_external_domain(poses,domain['world_min_mm'],domain['world_max_mm'])['model_float32_effective_domain']
    observations,normalization,groups=observations_from_manifest(a.manifest,a.qc_table,device,normalization_mode='per_series',assign_slice_positions=True); observations=apply_corrected_static_poses(observations,poses)
    if {obs.view.upper() for obs in observations if obs.qc_valid}!={'SAX','2CH','4CH'}: raise ValueError('S1 requires admitted SAX, 2CH and 4CH observations')
    if sax_only:
        observations=[obs for obs in observations if obs.qc_valid and obs.view.upper()=='SAX']
        if not observations: raise ValueError('S05SAX requires admitted SAX observations')
    set_reproducibility(6)
    model=DualLatentThickSlice(effective['lower_lps_mm'],effective['upper_lps_mm'],spacing['resolved_reference_spacing_mm'],use_tcnn=a.backend=='tcnn',require_tcnn=a.backend=='tcnn').to(device)
    sampler=ViewLocationBalancedSampler(observations,seed=12,required_views=('SAX',) if sax_only else ('SAX','2CH','4CH'))
    if slice_proportional:
        location_counts={view:len(sampler._grouped[view]) for view in sampler._views}
        frame_counts={view:{str(key):len(frames) for key,frames in sorted(sampler._grouped[view].items())} for view in sampler._views}
        print(json.dumps({'location_counts':location_counts,'valid_frames_per_location':frame_counts},sort_keys=True),flush=True)
        prop_counts=proportional_view_counts(location_counts,data['observations_per_update'])
        print(json.dumps({'view_counts':prop_counts,'total_observations_per_update':sum(prop_counts.values())},sort_keys=True),flush=True)
    runner=FrequencyRunner(model,sampler,view_counts=prop_counts,frequency=freq)
    provenance={'manifest':sha256(a.manifest),'qc_table':sha256(a.qc_table),'canonical_domain':sha256(a.canonical_domain),'static_manifest':sha256(a.static_manifest),'static_volume':sha256(volume),'corrected_poses':sha256(pose),'vendor_manifest':sha256(vendor_path),'config':sha256(a.experiment_config),'backend':model.canonical.backend,'spacing':spacing,'effective_domain':effective,'initialization':'from_scratch_no_nesvor_model_weights'}
    if sax_only: provenance['scope']='SAX-only physical corrected geometry'
    if budget_matched: provenance.update({'scope':'SAX+2CH+4CH','control':'matched_sax_budget','view_counts':dict(data['view_counts']),'total_observations_per_update':16,'matched_reference':'S05SAX','matched_quantity':'SAX observation exposure per optimizer step'})
    if equal_budget: provenance.update({'scope':'SAX+2CH+4CH','control':'equal_view_budget','view_counts':dict(data['view_counts']),'total_observations_per_update':24,'matched_reference':'S05SAX','matched_quantity':'8 observations per view per optimizer step'})
    if slice_proportional: provenance.update({'scope':'SAX+2CH+4CH','control':'slice_proportional_fixed_budget','sampling_mode':'location_proportional_fixed_budget','observations_per_update':24,'location_counts':location_counts,'valid_frames_per_location':frame_counts,'view_counts':prop_counts,'total_observations_per_update':sum(prop_counts.values()),'location_sampling_within_view':'uniform_with_replacement','frame_sampling_within_location':'uniform','pixel_samples_per_observation':10000,'formal_steps':10000})
    provenance.update({'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'schema':'simple4d_disentangle_freq_v1','base_commit':'00d9c6a382e428cf77b57a8bc84eaa082d7dfb91','frequency':freq,'temporal_audit_sha256':sha256(a.temporal_audit)})
    from third_party.simple4d.simple4d.model import GL_NODES, GL_WEIGHTS
    provenance.update({'renderer':'upstream_like_physical_thick_slice_5point','gl_nodes':GL_NODES,'gl_weights':GL_WEIGHTS,'gaussian_sigma':'slice_thickness_mm / 4','vendor_commit':vendor['commit'],'upstream_thick_slice_source_sha256':sha256(ROOT/'third_party/simple4d/simple4d/model.py')})
    if paired:
        basis=json.loads(a.basis_audit.read_text())
        if basis['location_count']!=sum(location_counts.values()) or basis['source']!=str(a.temporal_audit.parent/'temporal_observability.tsv'): raise ValueError('basis audit mismatch')
        entries={(r['view'],r['slice_id']):r for r in basis['locations']}
        if len(entries)!=basis['location_count']: raise ValueError('duplicate basis locations')
        for view, locations in sampler._grouped.items():
            for loc, frames in locations.items():
                times=torch.tensor(sorted(o.timestamp_s for o in frames),device=device,dtype=torch.float64)
                entry=entries.get((view,loc))
                if entry is None or len(times)!=entry['N'] or abs(float(times[-1]-times[0])-entry['span_s'])>1e-6: raise ValueError('basis audit differs from sampled timestamps')
                span=float(times[-1]-times[0])
                for key,band,mode in [('actual_resp_basis_frequencies',(.1,.6),'observable_basis_subset'),('actual_card_basis_frequencies',(.8,2.),'full_band')]:
                    actual=[k/span for k in range(math.ceil(band[0]*span),math.floor(band[1]*span)+1)]
                    if len(actual)!=len(entry[key]) or any(abs(x-y)>1e-9 for x,y in zip(actual,entry[key])): raise ValueError('basis frequencies differ from audit')
                    band_residual(times,torch.zeros((len(times),1),device=device),band,observability_mode=mode)
        cp=torch.load(a.resume,map_location='cpu',weights_only=False)
        parent=cp.get('provenance',{})
        if cp.get('schema')!='simple4d_disentangle_freq_v1' or cp.get('runner',{}).get('global_step')!=200 or len(cp['runner']['metrics'])!=200: raise ValueError('expected complete step-200 dual latent parent')
        for key in ('manifest','qc_table','canonical_domain','static_manifest','static_volume','corrected_poses','vendor_manifest','config','backend','spacing','effective_domain','view_counts','location_counts','pixel_samples_per_observation','renderer'):
            if parent.get(key)!=provenance.get(key): raise ValueError('parent provenance mismatch: '+key)
        pf=parent.get('frequency',{})
        if pf.get('lambda_resp_freq')!=0 or pf.get('lambda_card_freq')!=0: raise ValueError('parent must be zero frequency architecture control')
        parent_config=cp['config'];
        if any(parent_config.get(k)!=v for k,v in cfg.items() if k!='frequency_disentanglement'): raise ValueError('parent protocol mismatch')
        runner.load_state_dict(cp['runner']); diagnostics=cp['diagnostics']
        provenance.update({'common_parent_run':'architecture_control_freq0','common_parent_checkpoint':str(a.resume.resolve()),'common_parent_checkpoint_sha256':sha256(a.resume),'parent_global_step':200,'basis_audit':str(a.basis_audit.resolve()),'basis_audit_sha256':sha256(a.basis_audit),'basis_frequencies_by_location':basis['locations'],'mode':'dual_band' if a.dual_band else 'architecture_control','resp_prior_description':'observable subset of nominal 0.1-0.6 Hz; full 0.1 Hz cycle unsupported'})
    elif a.resume:
        raise ValueError('resume is only supported for the paired 2k diagnostic')
    else:
        diagnostics={}
    target=cfg['boundaries'][a.stop_after]
    if runner.global_step>target: raise ValueError('checkpoint step exceeds target')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'effective_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (a.output_dir/'experiment_inputs.json').write_text(json.dumps({'paths':{key:str(getattr(a,key).resolve()) for key in ('manifest','qc_table','canonical_domain','static_manifest','experiment_config')},'provenance':provenance},indent=2)+'\n')
    if paired:
        with torch.no_grad():
            sequence=sorted(sampler._grouped['SAX'][sorted(sampler._grouped['SAX'])[0]],key=lambda o:o.timestamp_s)
            means=torch.cat([model.encode_observation(o,sample=False)['mu'] for o in sequence])
            times=torch.tensor([o.timestamp_s for o in sequence],device=device,dtype=torch.float64)
            diagnostics['step_200']['dual_latent']=latent_diagnostics(times,means)
        (a.output_dir/'training_report_step_200.json').write_text(json.dumps({'global_step':200,'provenance':provenance,'diagnostics':{'step_200':diagnostics['step_200']}},indent=2,allow_nan=False)+'\n')
    while runner.global_step<target:
        row=runner.update()
        if row['weighted_card_freq']+row['weighted_resp_freq'] > 2*row['photo_raw']: raise RuntimeError('frequency terms overwhelm photometric loss')
        if runner.global_step in (1,10,25,50,100,150,200,201,210,250,500,1000,1500,2000): print(json.dumps({'progress':row},allow_nan=False),flush=True)
        if runner.global_step in ((500,1000,2000) if paired else cfg['boundaries'].values()):
            diag=diagnostic_metrics(model,sampler)
            with torch.no_grad():
                sequence=sorted(sampler._grouped['SAX'][sorted(sampler._grouped['SAX'])[0]],key=lambda o:o.timestamp_s)
                means=torch.cat([model.encode_observation(o,sample=False)['mu'] for o in sequence])
                times=torch.tensor([o.timestamp_s for o in sequence],device=device,dtype=torch.float64)
                coords=model.forward_deformation.coordinates
                axes=[torch.linspace(float(lo),float(lo+extent),4,device=device) for lo,extent in zip(coords.lower_world_mm,coords.extent_mm)]
                grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(1,-1,3)
                diag['dual_latent']=latent_diagnostics(times,means)
                diag['branch_interventions']=intervention_diagnostics(model.forward_deformation,grid,means)
            for key in list(diag):
                if key.startswith('static_baseline_'): diag['current_canonical_no_motion_'+key[len('static_baseline_'):]]=diag.pop(key)
            diagnostics[f'step_{runner.global_step}']=diag
            cp={'schema':'simple4d_disentangle_freq_v1','provenance':provenance,'runner':runner.state_dict(),'diagnostics':diagnostics,'config':cfg,'normalization_parameters':normalization,'normalization_groups':groups}
            torch.save(cp,a.output_dir/f'checkpoint_step_{runner.global_step}.pt')
            if not paired: torch.save(cp,a.output_dir/'checkpoint_last.pt')
            (a.output_dir/(f'training_report_step_{runner.global_step}.json' if paired else 'training_report.json')).write_text(json.dumps({'global_step':runner.global_step,'provenance':provenance,'diagnostics':diagnostics},indent=2,allow_nan=False)+'\n')
            with (a.output_dir/(f'training_metrics_step_{runner.global_step}.csv' if paired else 'training_metrics.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(runner.metrics[0])); writer.writeheader(); writer.writerows(runner.metrics)
    print(json.dumps({'step':runner.global_step,'backend':model.canonical.backend}))


if __name__=='__main__': main()
