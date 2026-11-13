#!/usr/bin/env python3
"""Read-only matched V1/V2 residual comparison across available checkpoints."""
import argparse
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from audit_simple4d_resp_v1_checkpoints import load_inputs,load_checkpoint,output_sha256
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import representative_locations
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiView
from cardioresp4d.experiments.simple4d_unified_evaluation import R0Adapter,R1Adapter,S0NativeAdapter,S1CoreMVAdapter,evaluate_matched
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import diagnostic_rng_context
from third_party.simple4d.simple4d.data import SliceSeries
from third_party.simple4d.simple4d.model import SIMPLE4D


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','canonical-domain','static-manifest','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('r0-checkpoint','r1-checkpoint','s1-checkpoint','s0-checkpoint','s0-input-dir','s0-run-dir'): p.add_argument('--'+name,type=Path)
    p.add_argument('--dynamic-crop',type=Path); p.add_argument('--chunk-size',type=int,default=4096)
    p.add_argument('--mode',choices=('v1','v2'),default='v2'); p.add_argument('--pair',nargs=2,action='append',default=[]); p.add_argument('--device',default='cuda:0'); p.add_argument('--gif-frame-ms',type=int,default=150)
    a=p.parse_args(); a.device=torch.device(a.device)
    if not any((a.r0_checkpoint,a.r1_checkpoint,a.s1_checkpoint,a.s0_checkpoint)): p.error('select at least one checkpoint')
    if a.device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    output=a.output_dir.resolve()
    checkpoints=[x.resolve() for x in (a.r0_checkpoint,a.r1_checkpoint,a.s1_checkpoint,a.s0_checkpoint) if x]
    inputs=[x.resolve() for x in (a.manifest,a.qc_table,a.canonical_domain,a.static_manifest) if x]
    protected_dirs=[x.resolve() for x in (a.s0_input_dir,a.s0_run_dir) if x]
    if any(output==path.parent or output in path.parents or path.parent in output.parents for path in checkpoints) or any(output==path or output in path.parents for path in inputs) or any(output==directory or directory in output.parents or output in directory.parents for directory in protected_dirs): raise ValueError('evaluation output may not overlap an input directory')
    if a.chunk_size<=0 or a.gif_frame_ms<=0: raise ValueError('chunk size and GIF duration must be positive')
    with diagnostic_rng_context(): config,provenance,base_model,cohort=load_inputs(a)
    auxiliary_models={}
    if 'static_model' in provenance['hashes']:
        static_model=Path(json.loads(a.static_manifest.read_text())['outputs']['model']['path']).resolve()
        if output==static_model.parent or output in static_model.parents or static_model.parent in output.parents: raise ValueError('evaluation output may not overlap the static model directory')
        auxiliary_models[static_model]=provenance['hashes']['static_model']
    adapters=[]
    if a.r0_checkpoint:
        state=torch.load(a.r0_checkpoint,map_location='cpu',weights_only=False); step=state.get('global_step')
        load_checkpoint(a.r0_checkpoint,step,config,provenance,base_model); adapters.append(R0Adapter(base_model,a.r0_checkpoint)); adapters[-1].extra_provenance={'checkpoint_input_provenance':state['provenance']}; base_model=None
    if a.r1_checkpoint:
        if base_model is None:
            with diagnostic_rng_context(): _,_,base_model,_=load_inputs(a)
        state=torch.load(a.r1_checkpoint,map_location='cpu',weights_only=False)
        r1prov=state.get('provenance',{})
        r1config=state.get('effective_config',{}); config_digest=hashlib.sha256(json.dumps(r1config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if state.get('schema')!='simple4d_r1_v1' or r1prov.get('input_provenance')!=provenance or r1prov.get('r1_config_sha256')!=config_digest or output_sha256(Path(r1prov['source_r0_path']))!=r1prov['source_r0_sha256']: raise ValueError('R1 input/source provenance mismatch')
        source_r0=Path(r1prov['source_r0_path']).resolve()
        if output==source_r0.parent or output in source_r0.parents or source_r0.parent in output.parents: raise ValueError('evaluation output may not overlap the R1 source directory')
        auxiliary_models[source_r0]=r1prov['source_r0_sha256']
        base_model.load_state_dict(state['model'],strict=True); adapters.append(R1Adapter(base_model,a.r1_checkpoint)); adapters[-1].extra_provenance={'checkpoint_input_provenance':r1prov}; base_model=None
    if a.s1_checkpoint:
        state=torch.load(a.s1_checkpoint,map_location='cpu',weights_only=False); prov=state.get('provenance',{}); hashes=provenance['hashes']
        if state.get('schema')!='simple4d_s1_v1' or any(prov.get(k)!=hashes[v] for k,v in [('manifest','manifest'),('qc_table','qc_table'),('canonical_domain','canonical_domain'),('static_manifest','static_manifest'),('static_volume','static_volume'),('corrected_poses','corrected_pose_manifest'),('vendor_manifest','simple4d_vendor_manifest')]): raise ValueError('S1 input provenance mismatch')
        box=prov['effective_domain']; spacing=prov['spacing']['resolved_reference_spacing_mm']; backend=prov['backend']
        import yaml
        config_path=ROOT/'configs/simple4d_core_mv_v1.yaml'
        if backend not in ('tcnn','pytorch_fallback') or prov.get('config')!=output_sha256(config_path) or state.get('config')!=yaml.safe_load(config_path.read_text()): raise ValueError('unknown S1 backend or incompatible configuration')
        model=Simple4DCoreMultiView(box['lower_lps_mm'],box['upper_lps_mm'],spacing,use_tcnn=backend=='tcnn',require_tcnn=backend=='tcnn').to(a.device)
        model.load_state_dict(state['runner']['model'],strict=True); adapters.append(S1CoreMVAdapter(model,a.s1_checkpoint)); adapters[-1].extra_provenance={'checkpoint_input_provenance':prov}
    if a.s0_checkpoint:
        if not a.s0_input_dir or not a.s0_run_dir: p.error('S0 requires --s0-input-dir and --s0-run-dir')
        state=torch.load(a.s0_checkpoint,map_location='cpu',weights_only=False); manifest_path=a.s0_input_dir/'input_adapter_manifest.json'; input_path=a.s0_input_dir/'sax_input.npz'; manifest=json.loads(manifest_path.read_text()); prov=state.get('provenance',{})
        if state.get('schema')!='simple4d_s0_v1' or prov.get('input_npz_sha256')!=output_sha256(input_path) or prov.get('input_manifest_sha256')!=output_sha256(manifest_path) or manifest['source_sha256']['manifest']!=provenance['hashes']['manifest'] or manifest['source_sha256']['qc_table']!=provenance['hashes']['qc_table'] or manifest['source_sha256']['corrected_pose_manifest']!=provenance['hashes']['corrected_pose_manifest'] or manifest['source_sha256']['static_manifest']!=provenance['hashes']['static_manifest']: raise ValueError('S0 provenance mismatch')
        if prov.get('vendor_manifest_sha256')!=provenance['hashes']['simple4d_vendor_manifest'] or prov.get('config_sha256')!=output_sha256(ROOT/'configs/simple4d_native_sax_v1.yaml'): raise ValueError('S0 vendor/config provenance mismatch')
        cfg=Namespace(**json.loads((a.s0_run_dir/'effective_config.json').read_text()))
        from cardioresp4d.experiments.simple4d_native_sax import validate_native_config
        validate_native_config(cfg)
        if cfg.slice_thickness!=manifest['slice_thickness_mm']: raise ValueError('S0 slice thickness mismatch')
        cfg.device=a.device.index or 0
        if a.device.type=='cpu' and torch.cuda.is_available(): raise ValueError('upstream S0 automatically selects CUDA when available')
        series=SliceSeries.load(input_path)
        model=SIMPLE4D(cfg,series.frames.shape[1],series.frames.shape[2],series.num_positions,series.spacing,series.depth_crop)
        if ('tcnn' if model.feature_net.use_tcnn else 'pytorch_fallback')!=prov['backend']: raise ValueError('S0 backend mismatch')
        for name,net in model.networks().items(): net.load_state_dict(state['runner']['networks'][name],strict=True)
        adapters.append(S0NativeAdapter(model,series,manifest['frame_mapping'],a.s0_checkpoint)); adapters[-1].extra_provenance={'checkpoint_input_provenance':prov,'input_npz_sha256':output_sha256(input_path),'input_adapter_manifest_sha256':output_sha256(manifest_path)}
    targets=[]
    for view in ('SAX','2CH','4CH'):
        groups=[(str(slice_id),frames) for v,slice_id,frames in cohort if v==view and len(frames)>=3]
        chosen=representative_locations([(name,float(frames[0].normalized_slice_position)) for name,frames in groups])
        for name,frames in groups:
            if name not in chosen: continue
            frames=sorted(frames,key=lambda x:(x.timestamp_s,x.dynamic_frame_id))
            targets.extend((frames[i-1],frames[i],frames[i+1]) for i in range(1,len(frames)-1) if frames[i-1].timestamp_s<frames[i].timestamp_s<frames[i+1].timestamp_s)
    evaluate_matched(adapters,targets,output,mode=a.mode,pairs=a.pair,duration_ms=a.gif_frame_ms)
    if a.dynamic_crop:
        import numpy as np
        from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import interpolate_neighbor_codes
        from cardioresp4d.experiments.simple4d_resp_v1_dynamic_visualization import save_triplanar,make_gif
        crop=json.loads(a.dynamic_crop.read_text())
        if crop.get('source_sha256')!=provenance['hashes']['static_volume']: raise ValueError('dynamic crop static volume mismatch')
        physical=[adapter for adapter in adapters if adapter.evaluation_scope!='SAX-only']
        selected=[item for item in targets if item[1].view.upper()=='SAX']
        if not physical or not selected: raise ValueError('dynamic anatomy needs a physical adapter and SAX targets')
        chosen=selected[0][1].slice_id; selected=[item for item in selected if item[1].slice_id==chosen]
        for adapter in physical:
            coords=adapter.model.forward_deformation.coordinates
            lo=coords.lower_world_mm.detach().cpu().numpy(); hi=lo+coords.extent_mm.detach().cpu().numpy()
            if np.any(np.asarray(crop['lower_lps_mm'])<lo-1e-4) or np.any(np.asarray(crop['upper_lps_mm'])>hi+1e-4) or not np.allclose(crop['spacing_mm'],coords.reference_spacing_mm.detach().cpu().numpy(),atol=1e-5,rtol=0): raise ValueError('dynamic crop is incompatible with model geometry')
            dest=output/'dynamic_anatomy'/adapter.method_id; dest.mkdir(parents=True); paths=[]; indices=tuple(int(n)//2 for n in crop['shape'])
            for i,(prev,target,nxt) in enumerate(selected):
                if a.mode=='v1': code=adapter.encode_v1(target)
                else:
                    z0,z1=adapter.encode_v1(prev),adapter.encode_v1(nxt)
                    values=interpolate_neighbor_codes(z0.cpu().numpy(),z1.cpu().numpy(),prev.timestamp_s,target.timestamp_s,nxt.timestamp_s)['time']
                    code=torch.as_tensor(values,device=z0.device,dtype=z0.dtype)
                volume=adapter.render_dynamic_state(code,crop,a.chunk_size); path=dest/f'frame_{i:04d}.png'
                save_triplanar(volume,indices,path,f'{adapter.method_id} {target.view} {target.slice_id} frame {target.dynamic_frame_id} t={target.timestamp_s-selected[0][1].timestamp_s:.3f}s {a.mode}',spacing=crop['spacing_mm']); paths.append(path)
            make_gif(paths,dest/'dynamic_anatomy.gif',a.gif_frame_ms)
            (dest/'metadata.json').write_text(json.dumps({'method':adapter.method_provenance(),'frame_ids':[x[1].dynamic_frame_id for x in selected],'timestamps_s':[x[1].timestamp_s for x in selected],'crop':crop,'crop_sha256':output_sha256(a.dynamic_crop),'slice_indices_lps_xyz':indices,'intensity_window':[0,1],'mode':a.mode},indent=2)+'\n')
            adapter.verify_unchanged()
    for model_path,digest in auxiliary_models.items():
        if output_sha256(model_path)!=digest: raise RuntimeError('source model changed during evaluation')
    path=output/'evaluation_manifest.json'; manifest=json.loads(path.read_text()); manifest['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); manifest['input_sha256']=provenance['hashes']; manifest['auxiliary_model_sha256']={str(k):v for k,v in auxiliary_models.items()}; manifest['output_sha256']={str(file.relative_to(output)):output_sha256(file) for file in output.rglob('*') if file.is_file() and file!=path}; path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'targets':len(targets),'output':str(output)}))


if __name__=='__main__': main()
