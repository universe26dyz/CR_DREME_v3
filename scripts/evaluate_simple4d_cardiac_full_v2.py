#!/usr/bin/env python3
"""Matched central-slice structural and displacement diagnostics for stage 2."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map,apply_corrected_static_poses
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
from cardioresp4d.experiments.simple4d_cardiac_full_v2 import FullCardiacModel
from cardioresp4d.roi.cardiac_box import CardiacBox
from cardioresp4d.roi.auto_cardiac import observation_soft_roi
from cardioresp4d.training.model import SourceFirstDynamicModel


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();device=torch.device(a.device)
 root=Path('/data2/dengyz/dataset/CR_DREME_v3');exp=root/'exp/simple4d_roi_cardiac_residual_v1'; ref=json.loads((exp/'01_stage1_base/base_checkpoint_reference.json').read_text());roi=json.loads((exp/'00_roi_audit/candidate_160/selected_box.json').read_text());saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);state=torch.load(ref['checkpoint'],map_location='cpu',weights_only=False);prov=state['provenance'];domain=prov['effective_domain'];base=Simple4DCoreMultiViewThickSlice(domain['lower_lps_mm'],domain['upper_lps_mm'],prov['spacing']['resolved_reference_spacing_mm'],use_tcnn=True,require_tcnn=True).to(device);base.load_state_dict(state['runner']['model']);model=FullCardiacModel(base,CardiacBox(roi['center_mm'],roi['size_mm']),roi['taper_mm']).to(device);model.load_state_dict(saved['runner']['model']);model.eval()
 manifest=root/'v1/phase1/dicom_manifest.csv';qc=manifest.parent/'acquisition_qc/acquisition_qc.csv';static=json.loads((root/'exp1_s2vdreme_reference_v1/02_static_nesvor/static_reference_manifest.json').read_text());obs,_,_=observations_from_manifest(manifest,qc,device,normalization_mode='per_series',assign_slice_positions=True);obs=apply_corrected_static_poses(obs,load_corrected_pose_map(static['corrected_pose_manifest']['path']))
 out=exp/'04_evaluation'/f"stage2_v2/step_{saved['runner']['step']}_50frame_gt";out.mkdir(parents=True,exist_ok=False); report={}
 for view,sid in [('SAX','SAX_s018'),('2CH','2CH_s019'),('4CH','4CH_s014')]:
  group=sorted((o for o in obs if o.qc_valid and o.view.upper()==view and o.slice_id==sid),key=lambda o:o.timestamp_s); assert len(group)==50, f'{view}/{sid}: expected 50 GT frames, got {len(group)}'; frames={k:[] for k in ('base_motion','cardiac_residual_only','combined_motion')}; comparison_frames=[]; vals=[]; dvs=[]; zs=[]; temporal_base=[]; temporal_card=[]; temporal_combined=[]
  for o in group:
   with torch.no_grad():
    bc=base.encode_observation(o,sample=False)['code']; cc,_,_=model.cardiac_encoder(o,model.box,model.taper_mm); zs.append(cc.cpu().numpy()[0]);h,w=o.image.shape[-2:]; rr,cols=torch.meshgrid(torch.arange(0,h,2,device=device),torch.arange(0,w,2,device=device),indexing='ij');px=torch.stack((cols,rr),-1).reshape(-1,2);world=SourceFirstDynamicModel._pixel_world(o,px.to(o.image));mask=observation_soft_roi(o,model.box,model.taper_mm)[::2,::2].flatten(); outputs={k:[] for k in frames};basefield=[];cardfield=[]
    for start in range(0,len(px),4096):
     pts=world[start:start+4096];n=len(pts);reso=pts.new_tensor([o.pixel_spacing_mm[1],o.pixel_spacing_mm[0],o.slice_thickness_mm]).expand(n,3);norm=o.normal.expand(n,-1);row=o.row_direction.expand(n,-1);col=o.column_direction.expand(n,-1)
     samples=base.psf.sample(pts,reso,row_direction=row,column_direction=col,normal=norm);bu=base.forward_deformation(samples,bc.expand(n,-1))['displacement_mm'];cu=model.forward_residual(samples,cc.expand(n,-1),model.box,model.taper_mm)
     nodes=pts.new_tensor(__import__('third_party.simple4d.simple4d.model',fromlist=['GL_NODES']).GL_NODES); weights=pts.new_tensor(__import__('third_party.simple4d.simple4d.model',fromlist=['GL_WEIGHTS']).GL_WEIGHTS)*torch.exp(-.5*(nodes*2)**2);weights/=weights.sum()
     for name,displacement in [('base_motion',bu),('cardiac_residual_only',cu),('combined_motion',bu+cu)]: outputs[name].append((base.canonical(samples+displacement)*weights).sum(-1))
     basefield.append(bu[:,2,:]);cardfield.append(cu[:,2,:])
    images={k:torch.cat(v) for k,v in outputs.items()};b=torch.cat(basefield);c=torch.cat(cardfield);target=o.image[0,::2,::2].flatten();
    for name,x in images.items():
     arr=(x.reshape(rr.shape).clamp(0,1).cpu().numpy()*255).astype('uint8');frames[name].append(Image.fromarray(arr))
    gt_arr=(target.reshape(rr.shape).clamp(0,1).cpu().numpy()*255).astype('uint8'); panel_w=gt_arr.shape[1]; montage=Image.new('L',(panel_w*3,gt_arr.shape[0])); montage.paste(Image.fromarray(gt_arr),(0,0)); montage.paste(frames['base_motion'][-1],(panel_w,0)); montage.paste(frames['combined_motion'][-1],(panel_w*2,0)); comparison_frames.append(montage)
    for name in ('base_motion','combined_motion'):
     e=images[name]-target;vals.append({'mode':name,'mse':float(e.square().mean()),'mae':float(e.abs().mean()),'roi_mse':float((e.square()*mask).sum()/mask.sum()),'roi_mae':float((e.abs()*mask).sum()/mask.sum())})
    inside=mask>.5;outside=mask==0; temporal_base.append(b.cpu().numpy()); temporal_card.append(c.cpu().numpy()); temporal_combined.append((b+c).cpu().numpy()); dvs.append({'base_rms':float(b.square().mean().sqrt()),'residual_inside_rms':float(c[inside].square().mean().sqrt()) if inside.any() else None,'residual_outside_max':float(c[outside].abs().max()) if outside.any() else None,'combined_rms':float((b+c).square().mean().sqrt())})
  dest=out/view.lower();dest.mkdir();[images[0].save(dest/(name+'.gif'),save_all=True,append_images=images[1:],duration=100,loop=0) for name,images in frames.items()];comparison_frames[0].save(dest/'gt_base_combined_comparison.gif',save_all=True,append_images=comparison_frames[1:],duration=100,loop=0);report[view]={'slice_id':sid,'frames':len(group),'frame_metrics':vals,'displacement':dvs,'latent_temporal_variance':float(np.var(np.array(zs),axis=0).mean()),'temporal_rms_mm':{name:float(np.sqrt(np.mean((np.array(values)-np.array(values).mean(axis=0))**2))) for name,values in [('base',temporal_base),('cardiac',temporal_card),('combined',temporal_combined)]}}
 (out/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n'); print(out)
if __name__=='__main__':main()
