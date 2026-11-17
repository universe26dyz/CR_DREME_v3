#!/usr/bin/env python3
"""Read-only S1 V2 PSF Monte-Carlo audit on matched physical slice sequences."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from audit_simple4d_resp_v1_checkpoints import load_inputs, output_sha256
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import diagnostic_rng_context
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiView
from cardioresp4d.experiments.simple4d_unified_evaluation import S1CoreMVAdapter, interpolation, residual_metrics
from cardioresp4d.experiments.simple4d_native_sax_collapse import temporal_stats
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import representative_locations
from cardioresp4d.visualization.simple4d_residual_visualization import residual_vmax, save_panel, save_gif


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':')).encode()).digest()[:8],'big') % (2**63-1)


def render(adapter, obs, pixels, code, *, mode, frame_index, chunk_size=256):
    """Use fixed full pixel ordering per sequence; chunks are stable for 128 samples."""
    if mode not in ('independent8','crn8','high128','repeat8'): raise ValueError('unknown audit mode')
    count=128 if mode=='high128' else 8
    # The 8-sample reference is exactly the existing evaluator's full-frame draw.
    chunks=[pixels] if count==8 else list(pixels.split(chunk_size))
    old=adapter.model.psf.n_samples; adapter.model.psf.n_samples=count
    try:
        arrays=[]
        for chunk_index,chunk in enumerate(chunks):
            seed=(20260823+frame_index if mode=='independent8' else
                  seed_for('S1',obs.view.upper(),str(obs.slice_id),mode,chunk_index, frame_index if mode=='repeat8' else 'sequence'))
            arrays.append(adapter.render_code(obs,chunk,code,seed=seed))
        return np.concatenate(arrays)
    finally:
        adapter.model.psf.n_samples=old


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','canonical-domain','static-manifest','checkpoint','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--slice-id',default='SAX_s018'); p.add_argument('--frame-id',type=int,default=801); p.add_argument('--expected-frames',type=int,default=48); p.add_argument('--chunk-size',type=int,default=256)
    a=p.parse_args(); a.device=torch.device(a.device); output=a.output_dir.resolve(); cp=a.checkpoint.resolve()
    if a.chunk_size<=0: p.error('chunk size must be positive')
    if a.device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    if output==cp.parent or output in cp.parents or cp.parent in output.parents: raise ValueError('output must be separate from checkpoint directory')
    for path in (a.manifest,a.qc_table,a.canonical_domain,a.static_manifest):
        path=path.resolve()
        if output==path or output in path.parents: raise ValueError('output must be separate from input files')
    if output.exists() and any(output.iterdir()): raise FileExistsError('output must be empty')
    digest=output_sha256(cp)
    with diagnostic_rng_context(): _,prov,base,cohort=load_inputs(a)
    del base
    state=torch.load(cp,map_location='cpu',weights_only=False); source=state.get('provenance',{}); hashes=prov['hashes']; config=ROOT/'configs/simple4d_core_mv_v1.yaml'
    if state.get('schema')!='simple4d_s1_v1' or state['runner']['global_step']!=2000 or source.get('config')!=output_sha256(config) or state.get('config')!=yaml.safe_load(config.read_text()) or any(source.get(k)!=hashes[v] for k,v in [('manifest','manifest'),('qc_table','qc_table'),('canonical_domain','canonical_domain'),('static_manifest','static_manifest'),('static_volume','static_volume'),('corrected_poses','corrected_pose_manifest'),('vendor_manifest','simple4d_vendor_manifest')]): raise ValueError('S1-2000 checkpoint/input provenance mismatch')
    static=json.loads(a.static_manifest.read_text())
    protected=[Path(static['outputs'][key]['path']).resolve() for key in ('model','volume')]+[Path(static['corrected_pose_manifest']['path']).resolve()]
    if any(output==path.parent or output in path.parents or path.parent in output.parents for path in protected): raise ValueError('output overlaps static artifact directory')
    box=source['effective_domain']; backend=source['backend']
    if backend not in ('tcnn','pytorch_fallback'): raise ValueError('unknown backend')
    model=Simple4DCoreMultiView(box['lower_lps_mm'],box['upper_lps_mm'],source['spacing']['resolved_reference_spacing_mm'],use_tcnn=backend=='tcnn',require_tcnn=backend=='tcnn').to(a.device)
    model.load_state_dict(state['runner']['model'],strict=True); model.eval(); adapter=S1CoreMVAdapter(model,cp)
    targets=[]
    for view in ('SAX','2CH','4CH'):
        groups=[(str(sid),list(frames)) for v,sid,frames in cohort if v==view and len(frames)>=3]
        chosen=set(representative_locations([(name,float(frames[0].normalized_slice_position)) for name,frames in groups]))
        if view=='SAX': chosen.add(a.slice_id)
        for name,frames in groups:
            if name not in chosen: continue
            frames.sort(key=lambda x:(x.timestamp_s,x.dynamic_frame_id))
            targets.extend((frames[i-1],frames[i],frames[i+1]) for i in range(1,len(frames)-1) if frames[i-1].timestamp_s<frames[i].timestamp_s<frames[i+1].timestamp_s)
    targets.sort(key=lambda x:(x[1].view.upper(),str(x[1].slice_id),x[1].timestamp_s,x[1].dynamic_frame_id))
    selected=[x for x in targets if x[1].view.upper()=='SAX' and str(x[1].slice_id)==a.slice_id]
    if len(selected)!=a.expected_frames: raise ValueError(f'{a.slice_id}: expected {a.expected_frames} V2 frames, found {len(selected)}')
    if not any(x[1].dynamic_frame_id==a.frame_id for x in selected): raise ValueError('requested repeat frame absent from V2 cohort')
    shapes={}; metrics=[]; sequences={}; repeat_targets={}; modes=('independent8','crn8','high128')
    for k,(prev,obs,nxt) in enumerate(targets):
        key=(obs.view.upper(),str(obs.slice_id)); h,w=obs.image.shape[-2:]
        if key in shapes and shapes[key]!=(h,w): raise ValueError('CRN requires identical pixel shape and ordering within each sequence')
        shapes[key]=(h,w); ids=torch.arange(h*w,device=obs.image.device); pixels=torch.stack((ids%w,torch.div(ids,w,rounding_mode='floor')),-1)
        z=interpolation(prev,obs,nxt,adapter.encode_v1(prev),adapter.encode_v1(nxt)); observed=obs.image[0].detach().cpu().numpy(); predictions={}
        for mode in modes:
            pred=render(adapter,obs,pixels,z,mode=mode,frame_index=k,chunk_size=a.chunk_size).reshape(h,w); predictions[mode]=pred
            metrics.append({'mode':mode,'view':key[0],'slice_id':key[1],'frame_id':obs.dynamic_frame_id,'timestamp_s':obs.timestamp_s,'previous_frame_id':prev.dynamic_frame_id,'next_frame_id':nxt.dynamic_frame_id,**residual_metrics(observed,pred)})
        sequences.setdefault(key,[]).append((obs,observed,predictions))
        if obs.dynamic_frame_id==a.frame_id and key==('SAX',a.slice_id): repeat_targets['requested']=(obs,pixels,z)
        repeat_targets.setdefault(key[0],(obs,pixels,z))
    for view in ('SAX','2CH','4CH'):
        if view not in repeat_targets: raise ValueError('representative V2 frame missing for '+view)
    output.mkdir(parents=True,exist_ok=True); (output/'panels').mkdir()
    by_view=[]
    for mode in modes:
        for view in ('SAX','2CH','4CH','overall'):
            rows=[r for r in metrics if r['mode']==mode and (view=='overall' or r['view']==view)]
            if rows:
                ncc=[r['ncc'] for r in rows if r['ncc'] is not None]
                by_view.append({'mode':mode,'view':view,'count':len(rows),'mean_mse':float(np.mean([r['mse'] for r in rows])),'mean_mae':float(np.mean([r['mae'] for r in rows])),'mean_ncc':float(np.mean(ncc)) if ncc else None})
    write_csv(output/'psf_stochasticity_by_view.csv',by_view)
    write_csv(output/'psf_frame_metrics.csv',metrics)
    temporal=[]
    for (view,sid),items in sequences.items():
        if len(items)<2: continue
        for mode in modes:
            temporal.append({'mode':mode,'view':view,'slice_id':sid,'frames':len(items),**temporal_stats(np.stack([x[1] for x in items]),np.stack([x[2][mode] for x in items]))})
    write_csv(output/'psf_temporal_fidelity.csv',temporal)
    items=sequences[('SAX',a.slice_id)]; vmax=residual_vmax([(obs,preds[mode]) for _,obs,preds in items for mode in modes]); paths={m:[] for m in modes}
    for i,(obs,observed,preds) in enumerate(items):
        for mode in modes:
            path=output/'panels'/f'{mode}_{i:04d}.png'; save_panel(path,observed,[preds[mode]],[mode],f'{a.slice_id} frame {obs.dynamic_frame_id} V2',vmax); paths[mode].append(path)
    for mode in modes: save_gif(paths[mode],output/f'{mode}_{a.slice_id}.gif')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    repeats={}
    for key,(obs,pixels,z) in repeat_targets.items():
        h,w=obs.image.shape[-2:]; images=np.stack([render(adapter,obs,pixels,z,mode='repeat8',frame_index=i).reshape(h,w) for i in range(32)])
        std=images.std(axis=0); target=obs.image[0].detach().cpu().numpy(); mse=np.mean((images-target)**2,axis=(1,2)); mae=np.mean(np.abs(images-target),axis=(1,2))
        repeats[key]={'view':obs.view,'slice_id':str(obs.slice_id),'frame_id':obs.dynamic_frame_id,'repeats':32,'mean_repeat_std':float(std.mean()),'p95_repeat_std':float(np.percentile(std,95)),'mse_variance':float(mse.var()),'mae_variance':float(mae.var()),'max_pairwise_pixel_difference':float((images.max(axis=0)-images.min(axis=0)).max())}
        fig,ax=plt.subplots(); im=ax.imshow(std,cmap='magma'); ax.set_title(f'{obs.view} {obs.slice_id} frame {obs.dynamic_frame_id}'); fig.colorbar(im,ax=ax); fig.savefig(output/f'same_frame_repeat_std_{key}.png',dpi=140); plt.close(fig)
    adapter.verify_unchanged()
    (output/'same_frame_repeat_summary.json').write_text(json.dumps(repeats,indent=2,allow_nan=False)+'\n')
    summary={'checkpoint_sha256':digest,'step':2000,'mode':'V2','cohort_count':len(targets),'requested_slice':a.slice_id,'requested_frame_ids':[x[1].dynamic_frame_id for x in selected],'views':by_view,'residual_vmax_all_modes':vmax,'intensity_window':[0,1],'seed_policy':{'independent8':'20260823 + sorted cohort frame index, matching unified evaluator','crn8':'SHA256(S1,view,slice_id,crn8,chunk,sequence); full image in row-major pixel order','high128':'same sequence-specific policy with high128 and fixed row-major chunks','repeat8':'32 independent seeds for fixed V2 code and observation'},'high128_chunk_size':a.chunk_size,'interpretation':'Differences between independent, CRN and high-sample estimates measure evaluation sensitivity; residual high-sample artifacts cannot uniquely distinguish trained model noise from forward-model mismatch.'}
    (output/'psf_stochasticity_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n'); print(json.dumps({'output':str(output),'checkpoint_sha256':digest,'targets':len(targets)}))

if __name__=='__main__': main()
