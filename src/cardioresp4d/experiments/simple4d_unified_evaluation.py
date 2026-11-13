"""Explicit method adapters and read-only, matched residual computations."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import interpolate_neighbor_codes


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def interpolation(previous,target,following,z_previous,z_following):
    if previous.view.upper()!=target.view.upper() or following.view.upper()!=target.view.upper() or str(previous.slice_id)!=str(target.slice_id) or str(following.slice_id)!=str(target.slice_id):
        raise ValueError('V2 neighbors must be from target location')
    result=interpolate_neighbor_codes(z_previous.detach().cpu().numpy(),z_following.detach().cpu().numpy(),previous.timestamp_s,target.timestamp_s,following.timestamp_s)
    return torch.as_tensor(result['time'],device=z_previous.device,dtype=z_previous.dtype)


def residual_metrics(observed,prediction):
    a=np.asarray(observed,dtype=float); b=np.asarray(prediction,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('observed and prediction must be aligned finite arrays')
    diff=a-b; aa=a-a.mean(); bb=b-b.mean(); denom=np.linalg.norm(aa)*np.linalg.norm(bb)
    return {'mse':float(np.mean(diff**2)),'mae':float(np.mean(np.abs(diff))),'ncc':float(np.sum(aa*bb)/denom) if denom>1e-12 else None}


class ProjectAdapter:
    evaluation_scope='SAX+2CH+4CH'
    psf_identity='NeSVoR Gaussian PSF on corrected LPS planes'
    def __init__(self,model,checkpoint_path,method_id):
        self.model=model; self.path=Path(checkpoint_path); self.method_id=method_id; self.checkpoint_or_model_sha256=sha256(self.path)
    def check_scope(self,obs):
        if obs.view.upper() not in ('SAX','2CH','4CH'): raise ValueError('unsupported view')
    def supports_target(self,previous,target,following,mode):
        return target.view.upper() in ('SAX','2CH','4CH')
    def encode_v1(self,obs):
        self.check_scope(obs)
        with torch.no_grad(): return self.model.encode_observation(obs,sample=False)['mu']
    def render_code(self,obs,pixels,code,*,seed=20260823):
        self.check_scope(obs)
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            torch.manual_seed(seed)
            return self.model.render_with_encoded(obs,pixels,{'code':code,'mu':code,'logvar':torch.zeros_like(code)})['predicted_intensity'].detach().cpu().numpy()
    def predict_v1(self,obs,pixels,*,seed=20260823): return self.render_code(obs,pixels,self.encode_v1(obs),seed=seed)
    def predict_v2(self,previous,target,following,pixels,*,seed=20260823):
        z=interpolation(previous,target,following,self.encode_v1(previous),self.encode_v1(following))
        return self.render_code(target,pixels,z,seed=seed)
    def method_provenance(self): return {'method_id':self.method_id,'scope':self.evaluation_scope,'checkpoint_path':str(self.path.resolve()),'checkpoint_sha256':self.checkpoint_or_model_sha256,'forward_model':self.psf_identity,**getattr(self,'extra_provenance',{})}
    def verify_unchanged(self):
        if sha256(self.path)!=self.checkpoint_or_model_sha256: raise RuntimeError('checkpoint changed during evaluation')
    def render_dynamic_state(self,code,crop,chunk_size=4096):
        from cardioresp4d.experiments.simple4d_resp_v1_identity_render import render_volume
        return render_volume(self.model,code,crop['lower_lps_mm'],crop['spacing_mm'],tuple(crop['shape']),chunk_size)


class R0Adapter(ProjectAdapter):
    def __init__(self,model,checkpoint_path): super().__init__(model,checkpoint_path,'R0')


class R1Adapter(ProjectAdapter):
    def __init__(self,model,checkpoint_path): super().__init__(model,checkpoint_path,'R1')


class S1CoreMVAdapter(ProjectAdapter):
    def __init__(self,model,checkpoint_path): super().__init__(model,checkpoint_path,'S1')


class S0NativeAdapter:
    evaluation_scope='SAX-only'
    psf_identity='upstream regular-stack thick-slice quadrature'
    def __init__(self,model,series,frame_mapping,checkpoint_path):
        self.model=model; self.series=series; self.mapping={int(row['dynamic_frame_id']):row for row in frame_mapping}; self.path=Path(checkpoint_path); self.method_id='S0'; self.checkpoint_or_model_sha256=sha256(self.path)
        if len(self.mapping)!=len(frame_mapping): raise ValueError('duplicate native frame IDs')
    def supports_target(self,previous,target,following,mode):
        observations=(target,) if mode=='v1' else (previous,target,following)
        return all(obs.view.upper()=='SAX' and int(obs.dynamic_frame_id) in self.mapping for obs in observations)
    def _position(self,obs):
        if obs.view.upper()!='SAX': raise ValueError('S0 is SAX-only')
        row=self.mapping.get(int(obs.dynamic_frame_id))
        if row is None or str(row['slice_id'])!=str(obs.slice_id) or abs(float(row['timestamp_s'])-obs.timestamp_s)>1e-6: raise ValueError('frame is outside admitted S0 mapping')
        index=int(row['upstream_frame'])
        if not np.allclose(self.series.frames[index],obs.image.detach().cpu().numpy()[0],rtol=0,atol=1e-6): raise ValueError('S0 target normalization mismatch')
        return int(row['slice_index'])
    def encode_v1(self,obs):
        pos=self._position(obs); self.model.eval_mode()
        with torch.no_grad(): return self.model.encode(obs.image,torch.tensor([pos]))[1]
    def render_code(self,obs,pixels,code,*,seed=20260823):
        from third_party.simple4d.simple4d.coords import make_slice_coords
        pos=self._position(obs); self.model.eval_mode(); h,w=self.series.frames.shape[1:]
        coords=make_slice_coords((1,h,w),torch.tensor([pos]),w,self.series.num_positions).to(self.model.device)
        uv=pixels.long().cpu(); indices=uv[:,1]*w+uv[:,0]
        if torch.any(uv[:,0]<0) or torch.any(uv[:,0]>=w) or torch.any(uv[:,1]<0) or torch.any(uv[:,1]>=h): raise ValueError('pixels outside slice')
        with torch.no_grad(): return self.model.predict(coords[:,indices.to(coords.device)],code.to(self.model.device))[0].reshape(-1).cpu().numpy()
    def predict_v1(self,obs,pixels,*,seed=20260823): return self.render_code(obs,pixels,self.encode_v1(obs),seed=seed)
    def predict_v2(self,previous,target,following,pixels,*,seed=20260823): return self.render_code(target,pixels,interpolation(previous,target,following,self.encode_v1(previous),self.encode_v1(following)),seed=seed)
    def method_provenance(self): return {'method_id':self.method_id,'scope':self.evaluation_scope,'checkpoint_path':str(self.path.resolve()),'checkpoint_sha256':self.checkpoint_or_model_sha256,'forward_model':self.psf_identity,**getattr(self,'extra_provenance',{})}
    def verify_unchanged(self):
        if sha256(self.path)!=self.checkpoint_or_model_sha256: raise RuntimeError('checkpoint changed during evaluation')


def evaluate_matched(adapters,targets,output,*,mode='v1',pairs=(),duration_ms=150):
    """Evaluate explicit (previous,target,next) tuples and render shared-scale GIFs."""
    import csv,json
    from cardioresp4d.visualization.simple4d_residual_visualization import residual_vmax,save_panel,save_gif
    if mode not in ('v1','v2') or not adapters or not targets: raise ValueError('adapters, targets and V1/V2 mode required')
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('evaluation output directory must be empty')
    if len({a.method_id for a in adapters})!=len(adapters): raise ValueError('method IDs must be unique')
    targets=sorted(targets,key=lambda x:(x[1].view.upper(),str(x[1].slice_id),x[1].timestamp_s,x[1].dynamic_frame_id))
    results=[]; cohort=[]
    for k,(prev,obs,nxt) in enumerate(targets):
        h,w=obs.image.shape[-2:]; idx=torch.arange(h*w,device=obs.image.device); pixels=torch.stack((idx%w,torch.div(idx,w,rounding_mode='floor')),-1)
        observed=obs.image[0].detach().cpu().numpy(); row={'view':obs.view,'slice_id':str(obs.slice_id),'frame_id':obs.dynamic_frame_id,'timestamp_s':obs.timestamp_s,'previous_frame_id':prev.dynamic_frame_id,'previous_timestamp_s':prev.timestamp_s,'next_frame_id':nxt.dynamic_frame_id,'next_timestamp_s':nxt.timestamp_s}
        predictions={}
        for adapter in adapters:
            supported=adapter.supports_target(prev,obs,nxt,mode)
            row['supports_'+adapter.method_id]=supported
            if not supported: continue
            values=adapter.predict_v1(obs,pixels,seed=20260823+k) if mode=='v1' else adapter.predict_v2(prev,obs,nxt,pixels,seed=20260823+k)
            predictions[adapter.method_id]=np.asarray(values).reshape(h,w)
        cohort.append(row)
        if predictions: results.append((obs,observed,predictions))
    if not results: raise ValueError('no targets supported by selected methods')
    for a,b in pairs:
        if a not in {x.method_id for x in adapters} or b not in {x.method_id for x in adapters}: raise ValueError('unknown pair method')
        if not any(a in r[2] and b in r[2] for r in results): raise ValueError('pair has no supported target intersection')
    for adapter in adapters: adapter.verify_unchanged()
    output.mkdir(parents=True,exist_ok=True); (output/'panels').mkdir(); (output/'gif').mkdir(); (output/'data').mkdir()
    def write_csv(path,rows):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_csv(output/'cohort.csv',cohort)
    metrics=[]
    for obs,observed,predictions in results:
        for name,prediction in predictions.items():
            adapter=next(a for a in adapters if a.method_id==name)
            metrics.append({'method':name,'checkpoint_sha256':adapter.checkpoint_or_model_sha256,'view':obs.view,'slice_id':str(obs.slice_id),'frame_id':obs.dynamic_frame_id,'timestamp_s':obs.timestamp_s,'mode':mode,'scope':adapter.evaluation_scope,**residual_metrics(observed,prediction)})
    write_csv(output/'frame_metrics.csv',metrics)
    summary=[]
    for adapter in adapters:
        for view in ('SAX','2CH','4CH'):
            group=[r for r in metrics if r['method']==adapter.method_id and r['view'].upper()==view]
            if not group: continue
            summary.append({'method':adapter.method_id,'scope':adapter.evaluation_scope,'view':view,'count':len(group),'mean_mse':float(np.mean([r['mse'] for r in group])),'mean_mae':float(np.mean([r['mae'] for r in group])),'mean_ncc':float(np.mean([r['ncc'] for r in group if r['ncc'] is not None])) if any(r['ncc'] is not None for r in group) else None})
    (output/'summary_by_view.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    method_summary=[]
    for adapter in adapters:
        group=[r for r in metrics if r['method']==adapter.method_id]
        if not group: continue
        ncc=[r['ncc'] for r in group if r['ncc'] is not None]
        method_summary.append({'method':adapter.method_id,'scope':adapter.evaluation_scope,'views':sorted({r['view'].upper() for r in group}),'count':len(group),'mean_mse':float(np.mean([r['mse'] for r in group])),'mean_mae':float(np.mean([r['mae'] for r in group])),'mean_ncc':float(np.mean(ncc)) if ncc else None})
    (output/'summary_by_method.json').write_text(json.dumps({'methods':method_summary,'scope_explicit_view_summaries':summary,'no_unlabeled_cross_scope_aggregate':True},indent=2,allow_nan=False)+'\n')
    write_csv(output/'comparison.csv',summary)
    comparisons=[(adapter.method_id,) for adapter in adapters]+[tuple(pair) for pair in pairs]
    manifest_comparisons=[]
    for names in comparisons:
        supported=[r for r in results if all(name in r[2] for name in names)]
        if not supported: continue
        vmax=residual_vmax([(observed,preds[name]) for _,observed,preds in supported for name in names])
        groups={}
        for obs,observed,preds in supported: groups.setdefault((obs.view,str(obs.slice_id)),[]).append((obs,observed,preds))
        for (view,slice_id),items in groups.items():
            items.sort(key=lambda item:(item[0].timestamp_s,item[0].dynamic_frame_id)); paths=[]
            for i,(obs,observed,preds) in enumerate(items):
                stem='_'.join(names)+f'_{view}_{slice_id}_{i:04d}'; path=output/'panels'/f'{stem}.png'
                save_panel(path,observed,[preds[name] for name in names],names,f'{view} {slice_id} frame {obs.dynamic_frame_id} t={obs.timestamp_s-items[0][0].timestamp_s:.3f}s {mode}',vmax)
                np.savez_compressed(output/'data'/f'{stem}.npz',observed=observed,**{name:preds[name] for name in names}); paths.append(path)
            gif=output/'gif'/('_'.join(names)+f'_{view}_{slice_id}_{mode}.gif'); save_gif(paths,gif,duration_ms)
            manifest_comparisons.append({'methods':list(names),'view':view,'slice_id':slice_id,'frames':[x[0].dynamic_frame_id for x in items],'timestamps_s':[x[0].timestamp_s for x in items],'gif':str(gif.relative_to(output)),'gif_frame_ms':duration_ms,'display_window':[0,1],'residual_vmin':0,'residual_vmax':vmax,'method_provenance':[a.method_provenance() for a in adapters if a.method_id in names]})
    for adapter in adapters: adapter.verify_unchanged()
    (output/'evaluation_manifest.json').write_text(json.dumps({'mode':mode,'methods':[a.method_provenance() for a in adapters],'comparisons':manifest_comparisons,'fairness_note':'S0 is SAX-only with upstream thick slice; R0/R1/S1 use project physical geometry and NeSVoR PSF.'},indent=2,allow_nan=False)+'\n')
    return metrics
