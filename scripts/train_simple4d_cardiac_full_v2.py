#!/usr/bin/env python3
"""Train Stage 2 from the verified S05PROP-2k checkpoint."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map,apply_corrected_static_poses
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
from cardioresp4d.experiments.simple4d_cardiac_full_v2 import FullCardiacModel,FullCardiacRunner
from cardioresp4d.training.runtime_state import set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.roi.cardiac_box import CardiacBox


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stop-step',type=int,required=True); p.add_argument('--resume',type=Path); p.add_argument('--output-dir',type=Path); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    root=Path('/data2/dengyz/dataset/CR_DREME_v3'); exp=root/'exp/simple4d_roi_cardiac_residual_v1'; out=a.output_dir or exp/'02_stage2_card_full_v2/continuation_5000'; ref=json.loads((exp/'01_stage1_base/base_checkpoint_reference.json').read_text()); roi=json.loads((exp/'00_roi_audit/candidate_160/selected_box.json').read_text()); cp=Path(ref['checkpoint']); digest=hashlib.sha256()
    with cp.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''): digest.update(block)
    if digest.hexdigest()!=ref['sha256']: raise ValueError('base checkpoint hash mismatch')
    base_state=torch.load(cp,map_location='cpu',weights_only=False); prov=base_state['provenance']; domain=prov['effective_domain'];
    if base_state['runner']['global_step']!=2000 or prov['view_counts']!={'SAX':8,'2CH':9,'4CH':7} or prov['pixel_samples_per_observation']!=10000: raise ValueError('base protocol mismatch')
    manifest=root/'v1/phase1/dicom_manifest.csv'; qc=manifest.parent/'acquisition_qc/acquisition_qc.csv'; static=json.loads((root/'exp1_s2vdreme_reference_v1/02_static_nesvor/static_reference_manifest.json').read_text()); device=torch.device(a.device)
    obs,_,_=observations_from_manifest(manifest,qc,device,normalization_mode='per_series',assign_slice_positions=True); obs=apply_corrected_static_poses(obs,load_corrected_pose_map(static['corrected_pose_manifest']['path']))
    set_reproducibility(6); base=Simple4DCoreMultiViewThickSlice(domain['lower_lps_mm'],domain['upper_lps_mm'],prov['spacing']['resolved_reference_spacing_mm'],use_tcnn=True,require_tcnn=True).to(device);base.load_state_dict(base_state['runner']['model'],strict=True)
    model=FullCardiacModel(base,CardiacBox(roi['center_mm'],roi['size_mm']),taper_mm=roi['taper_mm']).to(device); floor=json.loads((exp/'00_roi_audit/card_latent_energy_calibration.json').read_text())['floor']; runner=FullCardiacRunner(model,ViewLocationBalancedSampler(obs,seed=12),floor=floor);
    settings={'base_checkpoint_sha256':ref['sha256'],'roi_center_mm':roi['center_mm'],'roi_size_mm':roi['size_mm'],'taper_mm':roi['taper_mm'],'cardiac_code_dim':32,'hidden_dims':[256,256,256,256],'lr':5e-5,'lambda_roi':2.,'global_fraction':.25,'variance_floor':floor,'energy_weight':.01,'crop_size':128,'crop_margin':.1,'regularization_schedule':'500-2000 linear ramp','pixel_samples':10000,'regularizer_samples':256,'base_global_step':2000,'view_counts':{'SAX':8,'2CH':9,'4CH':7}}
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved['settings']!=settings: raise ValueError('resume settings mismatch')
        runner.load_state_dict(saved['runner'])
    else:
        if out.exists() and any(out.iterdir()): raise FileExistsError('refusing to overwrite an existing run')
        out.mkdir(parents=True,exist_ok=True); (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    while runner.step<a.stop_step:
        row=runner.update();print(json.dumps(row),flush=True)
        if runner.step in (200,500,1000,2000,3000,4000,5000) or runner.step==a.stop_step:
            torch.save({'settings':settings,'runner':runner.state_dict()},out/f'checkpoint_step_{runner.step}.pt')
    print('finished',runner.step,flush=True)
if __name__=='__main__': main()
