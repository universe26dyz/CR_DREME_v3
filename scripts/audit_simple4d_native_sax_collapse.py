#!/usr/bin/env python3
"""Read-only native S0 posterior-mean, deformation and temporal-fidelity audit."""
import argparse, csv, json, sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from third_party.simple4d.simple4d.data import SliceSeries
from third_party.simple4d.simple4d.model import SIMPLE4D
from third_party.simple4d.simple4d.coords import make_slice_coords
from cardioresp4d.experiments.simple4d_native_sax import validate_native_config
from cardioresp4d.experiments.simple4d_native_sax_collapse import latent_stats,temporal_stats,rms,ratio
from cardioresp4d.experiments.simple4d_unified_evaluation import sha256


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input-dir','checkpoint','run-dir','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/simple4d_native_sax_v1.yaml')
    p.add_argument('--device',default='cpu'); p.add_argument('--slice-id',action='append'); p.add_argument('--max-frames',type=int,default=32); p.add_argument('--grid-size',type=int,default=9)
    a=p.parse_args(); device=torch.device(a.device)
    if device.type=='cpu' and torch.cuda.is_available(): raise ValueError('upstream S0 selects CUDA automatically; hide CUDA devices for CPU')
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    if a.max_frames<2 or a.grid_size<3: raise ValueError('need at least two frames and a grid of at least 3')
    output=a.output_dir.resolve(); protected=[a.input_dir.resolve(),a.run_dir.resolve(),a.checkpoint.resolve().parent]
    if any(output==x or output in x.parents or x in output.parents for x in protected): raise ValueError('output must be separate from input and training directories')
    if output.exists() and any(output.iterdir()): raise FileExistsError('audit output must be empty')
    checkpoint_hash=sha256(a.checkpoint); state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    manifest_path=a.input_dir/'input_adapter_manifest.json'; input_path=a.input_dir/'sax_input.npz'; manifest=json.loads(manifest_path.read_text()); prov=state['provenance']
    if state.get('schema')!='simple4d_s0_v1' or prov['input_npz_sha256']!=sha256(input_path) or prov['input_manifest_sha256']!=sha256(manifest_path): raise ValueError('S0 input/checkpoint provenance mismatch')
    vendor_path=ROOT/'third_party/simple4d_VENDOR.json'; vendor=json.loads(vendor_path.read_text())
    if prov['config_sha256']!=sha256(a.config) or prov['vendor_manifest_sha256']!=sha256(vendor_path) or prov['vendor_commit']!=vendor['commit'] or any(sha256(ROOT/'third_party/simple4d'/name)!=digest for name,digest in vendor['key_files'].items()): raise ValueError('S0 config/vendor provenance mismatch')
    cfg=Namespace(**json.loads((a.run_dir/'effective_config.json').read_text())); validate_native_config(cfg)
    if cfg.slice_thickness!=manifest['slice_thickness_mm']: raise ValueError('slice thickness mismatch')
    cfg.device=device.index or 0; series=SliceSeries.load(input_path)
    model=SIMPLE4D(cfg,*series.frames.shape[1:],series.num_positions,series.spacing,series.depth_crop)
    if model.device.type!=device.type or ('tcnn' if model.feature_net.use_tcnn else 'pytorch_fallback')!=prov['backend']: raise ValueError('device or backend mismatch')
    for name,net in model.networks().items(): net.load_state_dict(state['runner']['networks'][name],strict=True)
    model.eval_mode(); mapping=sorted(manifest['frame_mapping'],key=lambda x:int(x['upstream_frame']))
    if len(mapping)!=len(series): raise ValueError('frame mapping length mismatch')
    with torch.no_grad():
        codes=np.concatenate([model.encode(torch.as_tensor(series.frames[i:i+32]),torch.as_tensor(series.slice_index[i:i+32]))[1].cpu().numpy() for i in range(0,len(series),32)])
    codes=np.asarray(codes.cpu() if torch.is_tensor(codes) else codes); groups={}
    for row in mapping: groups.setdefault(str(row['slice_id']),[]).append(row)
    selected=a.slice_id or sorted(groups)[:3]
    if any(x not in groups for x in selected): raise ValueError('unknown slice id')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'latent_codes.npz',mu=codes,frame_ids=np.array([r['dynamic_frame_id'] for r in mapping]),slice_ids=np.array([str(r['slice_id']) for r in mapping]),timestamps_s=np.array([r['timestamp_s'] for r in mapping]))
    overall=latent_stats(codes); by_slice={}; sequences={}
    for name in selected:
        rows=sorted(groups[name],key=lambda x:(x['timestamp_s'],x['dynamic_frame_id']))
        if len(rows)>a.max_frames: rows=[rows[i] for i in np.linspace(0,len(rows)-1,a.max_frames,dtype=int)]
        if len(rows)<2: raise ValueError('selected slice requires two frames')
        sequences[name]=rows; by_slice[name]=latent_stats(codes[[r['upstream_frame'] for r in rows]])
    (output/'latent_summary.json').write_text(json.dumps({'checkpoint_sha256':checkpoint_hash,'all_frames':overall,'selected_sequences':by_slice},indent=2,allow_nan=False)+'\n')
    centered=codes-codes.mean(axis=0); _,_,vt=np.linalg.svd(centered,full_matrices=False); pc=centered@vt[:2].T
    write_csv(output/'latent_pca.csv',[{'frame_id':r['dynamic_frame_id'],'slice_id':r['slice_id'],'timestamp_s':r['timestamp_s'],'pc1':float(pc[i,0]),'pc2':float(pc[i,1]) if pc.shape[1]>1 else 0} for i,r in enumerate(mapping)])
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); ax.scatter(pc[:,0],pc[:,1],s=5); ax.set(xlabel='PC1',ylabel='PC2'); fig.savefig(output/'latent_pca.png',dpi=140); plt.close(fig)
    n=a.grid_size; axes=[torch.linspace(-1+1/n,1-1/n,n,device=model.device) for _ in range(3)]; grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),dim=-1).reshape(1,-1,3)
    scale=np.array([series.num_positions*series.spacing[0],series.frames.shape[2]*series.spacing[2],series.frames.shape[1]*series.spacing[1]])/2
    dvf_rows=[]; fidelity=[]; image_pairs=[]; sequence_dvf=[]
    for name,rows in sequences.items():
        observed=[]; predicted=[]; fields=[]
        for row in rows:
            i=int(row['upstream_frame']); code=torch.as_tensor(codes[i:i+1],device=model.device); pos=torch.tensor([int(series.slice_index[i])]); coords=make_slice_coords((1,*series.frames.shape[1:]),pos,series.frames.shape[2],series.num_positions).to(model.device)
            with torch.no_grad():
                pred=torch.cat([model.predict(chunk,code)[0].reshape(-1) for chunk in coords.split(4096,dim=1)]).cpu().numpy().reshape(series.frames.shape[1:])
                field=model.dvf(grid,code).cpu().numpy().reshape(n,n,n,3)*scale
            observed.append(series.frames[i]); predicted.append(pred); fields.append(field)
            spacing=2*scale/n; deriv=np.empty((n,n,n,3,3))
            for component in range(3):
                for axis,gradient in enumerate(np.gradient(field[...,component],*spacing,edge_order=2)): deriv[...,component,axis]=gradient
            det=np.linalg.det(deriv+np.eye(3)); translation=field.mean(axis=(0,1,2))
            dvf_rows.append({'slice_id':name,'frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'dvf_rms_mm':rms(field),'max_magnitude_mm':float(np.linalg.norm(field,axis=-1).max()),'translation_d_mm':translation[0],'translation_w_mm':translation[1],'translation_h_mm':translation[2],'nonrigid_residual_rms_mm':rms(field-translation),'jacobian_mean':float(det.mean()),'jacobian_p01':float(np.percentile(det,1)),'jacobian_p05':float(np.percentile(det,5)),'jacobian_p95':float(np.percentile(det,95)),'jacobian_nonpositive_fraction':float((det<=0).mean())})
        fields=np.array(fields); static=fields.mean(axis=0); dynamic=fields-static
        sequence_dvf.append({'slice_id':name,'mean_dvf_rms_mm':rms(fields),'static_dvf_rms_mm':rms(static),'dynamic_dvf_rms_mm':rms(dynamic),'dvf_dynamic_ratio':ratio(rms(dynamic),rms(static)),'adjacent_dvf_difference_rms_mm':rms(np.diff(fields,axis=0))})
        observed=np.array(observed); predicted=np.array(predicted); fidelity.append({'slice_id':name,**temporal_stats(observed,predicted)}); image_pairs.append((name,observed.std(axis=0),predicted.std(axis=0)))
    write_csv(output/'frame_dvf_metrics.csv',dvf_rows); write_csv(output/'temporal_fidelity_by_slice.csv',fidelity)
    (output/'dvf_summary.json').write_text(json.dumps({'grid':'common cell-centred normalized (depth,width,height) grid; finite-difference Jacobian in physical mm','grid_size':n,'selected_sequences':sequence_dvf},indent=2,allow_nan=False)+'\n')
    (output/'temporal_fidelity.json').write_text(json.dumps({'selected_sequences':fidelity,'latent_dynamic_ratio':{k:v['dynamic_ratio'] for k,v in by_slice.items()},'dvf_dynamic_ratio':{x['slice_id']:x['dvf_dynamic_ratio'] for x in sequence_dvf},'roi':'whole image; no cardiac mask'},indent=2,allow_nan=False)+'\n')
    for suffix,index in [('observed',1),('predicted',2),('difference',3)]:
        fig,axs=plt.subplots(1,len(image_pairs),squeeze=False,figsize=(4*len(image_pairs),4))
        for ax,(name,obs,pred) in zip(axs[0],image_pairs):
            img=obs if index==1 else pred if index==2 else pred-obs
            im=ax.imshow(img,cmap='coolwarm' if index==3 else 'magma'); ax.set_title(name); ax.axis('off'); fig.colorbar(im,ax=ax)
        fig.tight_layout(); fig.savefig(output/f'temporal_std_{suffix}.png',dpi=140); plt.close(fig)
    if sha256(a.checkpoint)!=checkpoint_hash: raise RuntimeError('checkpoint changed during audit')
    print(json.dumps({'output':str(output),'checkpoint_sha256':checkpoint_hash,'frames':len(codes)}))

if __name__=='__main__': main()
