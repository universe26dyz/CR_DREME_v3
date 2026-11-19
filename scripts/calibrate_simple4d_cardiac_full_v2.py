#!/usr/bin/env python3
"""Calibrate the branch variance floor on the 143 QC-valid locations."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map,apply_corrected_static_poses
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
from cardioresp4d.experiments.simple4d_cardiac_full_v2 import FullCardiacModel,heart_crop,crop_rectangle
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.training.runtime_state import set_reproducibility
from cardioresp4d.roi.cardiac_box import CardiacBox
root=Path('/data2/dengyz/dataset/CR_DREME_v3');exp=root/'exp/simple4d_roi_cardiac_residual_v1';output=exp/'00_roi_audit/card_latent_energy_calibration.json'
if output.exists(): raise FileExistsError(output)
ref=json.loads((exp/'01_stage1_base/base_checkpoint_reference.json').read_text());roi=json.loads((exp/'00_roi_audit/candidate_160/selected_box.json').read_text());cp=torch.load(ref['checkpoint'],map_location='cpu',weights_only=False);p=cp['provenance'];bounds=p['effective_domain'];m=root/'v1/phase1/dicom_manifest.csv';q=m.parent/'acquisition_qc/acquisition_qc.csv';st=json.loads((root/'exp1_s2vdreme_reference_v1/02_static_nesvor/static_reference_manifest.json').read_text());device=torch.device('cuda:0')
obs,_,_=observations_from_manifest(m,q,device,normalization_mode='per_series',assign_slice_positions=True);obs=apply_corrected_static_poses(obs,load_corrected_pose_map(st['corrected_pose_manifest']['path']));sampler=ViewLocationBalancedSampler(obs,seed=12)
set_reproducibility(6);base=Simple4DCoreMultiViewThickSlice(bounds['lower_lps_mm'],bounds['upper_lps_mm'],p['spacing']['resolved_reference_spacing_mm'],use_tcnn=True,require_tcnn=True).to(device);base.load_state_dict(cp['runner']['model']);model=FullCardiacModel(base,CardiacBox(roi['center_mm'],roi['size_mm']),roi['taper_mm']).to(device);model.eval();rows=[]
with torch.no_grad():
 for v in sampler._views:
  for sid,items in sorted(sampler._grouped[v].items()):
   frames=sorted(items,key=lambda x:x.timestamp_s); frames=[frames[i] for i in np.linspace(0,len(frames)-1,min(8,len(frames))).round().astype(int)]; rect=crop_rectangle(frames[0],model.box); images=torch.stack([x.image for x in frames]); crops=torch.cat([heart_crop(x,rect) for x in frames]); pos=images.new_tensor([x.normalized_slice_position for x in frames]); b,_=base.respiratory_encoder.encode(images,pos);c,_=model.cardiac_encoder.network.encode(crops,pos); bv=float((b-b.mean(0)).square().mean());cv=float((c-c.mean(0)).square().mean());rows.append({'view':v,'slice_id':sid,'frames':len(frames),'base_variance':bv,'random_card_variance':cv})
basevals=np.array([r['base_variance'] for r in rows]);cardvals=np.array([r['random_card_variance'] for r in rows]);floor=float(.05*np.median(basevals));result={'locations':len(rows),'base_variance_quantiles':np.quantile(basevals,[0,.25,.5,.75,1]).tolist(),'random_card_variance_quantiles':np.quantile(cardvals,[0,.25,.5,.75,1]).tolist(),'floor':floor,'floor_rule':'0.05 times median frozen base temporal variance','frames_per_location':8,'model_seed':6,'rows':rows};output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
