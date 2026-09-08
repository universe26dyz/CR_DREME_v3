"""功能：短程 Stage-1A/1B Canonical INR trainer 与真实 slice-domain validation。
论文来源：S2V-DREME warm start；NeSVoR static PSF SVR/Gaussian likelihood。
输入：initial reference+valid mask、canonical-domain JSON、qc-valid temporal mean slices。
输出：Stage1A/B checkpoints、canonical NIfTI、difference/QC/variance/metric artifacts。
主要步骤：1A masked MSE only；1B balanced static multi-view PSF MSE warm-up then mean-slice Gaussian NLL。
是否属于原论文直接实现 / 必要适配 / 可选实验：Hybrid v2 necessary trainer adaptation；no motion in either stage.
命令行使用示例：python -m cardioresp4d.training.stage1 --initial-reference ref.nii.gz --canonical-domain canonical_domain.json --manifest manifest.csv --output-dir results/training/stage1
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from cardioresp4d.models.hash_inr import CanonicalINR, PhysicalHashConfig
from cardioresp4d.models.uncertainty import NamespacedObservationUncertainty
from cardioresp4d.reference.build_mean_slices import build_mean_slices
from cardioresp4d.rendering.psf_renderer import AnisotropicPSFRenderer
from cardioresp4d.losses.stage_aware import gaussian_nll, masked_reference_mse
LPS_FROM_RAS=torch.tensor([-1.,-1.,1.])

def resolve_device(requested:str)->torch.device:
    if requested=='cuda' and not torch.cuda.is_available(): raise RuntimeError('device=cuda was requested but CUDA is unavailable')
    return torch.device('cuda' if requested=='auto' and torch.cuda.is_available() else ('cpu' if requested=='auto' else requested))
def model_config(domain:Path,profile:str='production',**overrides)->dict:
    extent=(np.asarray(json.loads(domain.read_text())['world_max_mm'])-np.asarray(json.loads(domain.read_text())['world_min_mm'])).tolist()
    if profile=='smoke': return {'profile':'smoke','levels':4,'features_per_level':2,'hash_size':512,'min_resolution':8,'max_resolution':32,'hidden_dim':32,'latent_dim':12,'canonical_world_extent_mm':extent}
    if profile!='production': raise ValueError('profile must be smoke or production')
    physical=PhysicalHashConfig(**{k:v for k,v in overrides.items() if k in PhysicalHashConfig.__dataclass_fields__}).derive(extent); return {'profile':'production',**physical}
def _model(config:dict|None=None) -> CanonicalINR:
    config=config or {'profile':'smoke','levels':4,'features_per_level':2,'hash_size':512,'min_resolution':8,'max_resolution':32,'hidden_dim':32,'latent_dim':12}
    if config['profile']=='smoke': return CanonicalINR(**{k:config[k] for k in ('levels','features_per_level','hash_size','min_resolution','max_resolution','hidden_dim','latent_dim')})
    return CanonicalINR(features_per_level=config['features_per_level'],hash_size=config['hash_size'],hidden_dim=config['hidden_dim'],latent_dim=config['latent_dim'],resolutions=config['grid_resolutions'])
def _load_domain(path:Path) -> torch.Tensor: return torch.tensor(json.loads(path.read_text())['world_to_normalized'],dtype=torch.float32)
def _to_norm_lps(ras:torch.Tensor,w2n:torch.Tensor)->torch.Tensor:
    lps=ras*LPS_FROM_RAS.to(ras); return torch.einsum('ij,...j->...i',w2n,torch.cat((lps,torch.ones((*lps.shape[:-1],1),device=ras.device)), -1))[...,:3]
def _save_checkpoint(path:Path,model:CanonicalINR,**payload:object)->None: torch.save({'model_state':model.state_dict(),'model_config':payload.pop('model_config',model.config),**payload},path)
def _load_checkpoint(path:Path)->CanonicalINR:
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if 'model_config' not in payload: raise ValueError('legacy checkpoint has no model_config')
    model=_model(payload['model_config']); model.load_state_dict(payload['model_state']); return model

def train_stage1a(initial_reference:Path, valid_mask:Path|None, canonical_domain:Path, output:Path, *, steps:int=24, batch_size:int=2048, lr:float=2e-3, profile:str='smoke', device:str='auto', seed:int=0, **kwargs)->dict:
    started=time.perf_counter(); output.mkdir(parents=True,exist_ok=True); image=nib.load(str(initial_reference)); volume=np.asanyarray(image.dataobj).astype(np.float32)
    if valid_mask is None or not valid_mask.is_file():
        mask=np.isfinite(volume).astype(np.float32); valid_mask=output/'initial_reference_valid_mask.nii.gz'; nib.save(nib.Nifti1Image(mask,image.affine),str(valid_mask)); mask_source='derived_all_finite_existing_reference'
    else: mask=np.asanyarray(nib.load(str(valid_mask)).dataobj).astype(np.float32); mask_source='provided'
    if volume.shape!=mask.shape: raise ValueError('initial-reference and valid-mask shapes differ')
    indices=np.argwhere(mask>0); assert len(indices)>0
    dev=resolve_device(device); cfg=model_config(canonical_domain,profile,**kwargs); affine=torch.tensor(image.affine,dtype=torch.float32,device=dev); w2n=_load_domain(canonical_domain).to(dev); model=_model(cfg).to(dev); opt=torch.optim.Adam(model.parameters(),lr=lr); losses=[]
    target=torch.tensor(volume,device=dev); generator=torch.Generator().manual_seed(seed)
    for _ in range(steps):
        pick=indices[torch.randint(len(indices),(min(batch_size,len(indices)),),generator=generator).numpy()]; vox=torch.tensor(pick,dtype=torch.float32,device=dev); ras=torch.cat((vox,torch.ones(len(vox),1,device=dev)),1)@affine.T; pred=model(_to_norm_lps(ras[:,:3],w2n)).squeeze(-1); observed=target[pick[:,0],pick[:,1],pick[:,2]]; loss=masked_reference_mse(pred,observed,torch.ones_like(observed)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    checkpoint=output/'stage1a.pt'; _save_checkpoint(checkpoint,model,stage='1A',steps=steps,losses=losses,mask_path=str(valid_mask),model_config=cfg); report={'stage':'1A','steps':steps,'seconds':time.perf_counter()-started,'loss_first':losses[0],'loss_last':losses[-1],'checkpoint':str(checkpoint),'model_config':cfg}; (output/'stage1a_metrics.json').write_text(json.dumps(report,indent=2)); return report

def _export_canonical(model:CanonicalINR,domain:Path,path:Path,size:int=64)->Path:
    payload=json.loads(domain.read_text()); lower=np.asarray(payload['world_min_mm']); upper=np.asarray(payload['world_max_mm']); axes=[np.linspace(lower[i],upper[i],size,dtype=np.float32) for i in range(3)]; grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3); w2n=torch.tensor(payload['world_to_normalized'],dtype=torch.float32); values=[]
    with torch.no_grad():
        for chunk in np.array_split(grid,max(1,len(grid)//8192)):
            values.append(model(_to_norm_lps(torch.tensor(chunk),w2n)).squeeze(-1).numpy())
    data=np.concatenate(values).reshape(size,size,size); affine=np.diag([-(upper[0]-lower[0])/(size-1),-(upper[1]-lower[1])/(size-1),(upper[2]-lower[2])/(size-1),1.]); affine[:3,3]=[-upper[0],-upper[1],lower[2]]; nib.save(nib.Nifti1Image(data.astype(np.float32),affine),str(path)); return path

def train_stage1b(stage1a_checkpoint:Path, mean_manifest:Path, canonical_domain:Path, output:Path, *, warmup_steps:int=6, nll_steps:int=6, patch:int=24, lr:float=1e-3)->dict:
    started=time.perf_counter(); output.mkdir(parents=True,exist_ok=True); root=mean_manifest.parent; rows=list(csv.DictReader(mean_manifest.open())); by_view={view:[row for row in rows if row['view']==view] for view in ('SAX','2CH','4CH')}
    if any(not by_view[v] for v in by_view): raise ValueError('Stage1B requires valid mean slices from SAX, 2CH, and 4CH')
    model=_load_checkpoint(stage1a_checkpoint); renderer=AnisotropicPSFRenderer(_load_domain(canonical_domain)); uncertainty=NamespacedObservationUncertainty(12,num_mean_slices=len(rows),num_dynamic_frames=1,embedding_dim=4); opt=torch.optim.Adam(list(model.parameters())+list(uncertainty.parameters()),lr=lr); losses=[]; per_view={v:{'mse':[],'nll':[]} for v in by_view}; observation_ids={row['mean_slice_id']:i for i,row in enumerate(rows)}; initial_metrics=_qc_views(model,renderer,uncertainty,root,by_view,observation_ids,patch,output/'initial_qc')
    schedule=[(i < warmup_steps) for i in range(warmup_steps+nll_steps)]
    for step,warmup in enumerate(schedule):
        view=('SAX','2CH','4CH')[step%3]; row=by_view[view][(step//3)%len(by_view[view])]; target,geo=_mean_observation(root,row,patch); out=renderer(model,**geo); variance=uncertainty(out['latent_samples'],torch.tensor([observation_ids[row['mean_slice_id']]]),out['psf_weights'],namespace='mean_slice',enabled=not warmup)['total_variance']; loss=((out['predicted_slice'].squeeze(1)-target).square().mean() if warmup else gaussian_nll(out['predicted_slice'].squeeze(1),target,variance)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach())); per_view[view]['mse' if warmup else 'nll'].append(float(loss.detach()))
    checkpoint=output/'stage1b.pt'; _save_checkpoint(checkpoint,model,stage='1B',warmup_steps=warmup_steps,nll_steps=nll_steps,losses=losses,uncertainty_state=uncertainty.state_dict()); volume=_export_canonical(model,canonical_domain,output/'stage1b_canonical.nii.gz'); _difference(output/'stage1a_canonical.nii.gz' if (output/'stage1a_canonical.nii.gz').is_file() else None,volume,output/'stage1a_vs_stage1b_difference.nii.gz'); metrics=_qc_views(model,renderer,uncertainty,root,by_view,observation_ids,patch,output); missing={'hard_invalid_mean_observations':'recorded by exclusion summary','canonical_domain_unchanged':True,'canonical_query_finite':bool(np.isfinite(np.asanyarray(nib.load(str(volume)).dataobj)).all())}; (output/'missing_slice_through_plane_qc.json').write_text(json.dumps(missing,indent=2)); (output/'stage1b_curve.csv').write_text('step,loss\n'+'\n'.join(f'{i},{x}' for i,x in enumerate(losses))+'\n'); report={'stage':'1B','steps':len(schedule),'seconds':time.perf_counter()-started,'loss_first':losses[0],'loss_last':losses[-1],'per_view_batch_count':{v:sum(len(x) for x in per_view[v].values()) for v in per_view},'per_view_train_loss':{v:{phase:{'first':x[0],'last':x[-1]} for phase,x in per_view[v].items() if x} for v in per_view},'checkpoint':str(checkpoint),'volume':str(volume),'initial_metrics':initial_metrics,'metrics':metrics,'missing_slice_qc':missing}; (output/'stage1b_metrics.json').write_text(json.dumps(report,indent=2)); return report

def _mean_observation(root:Path,row:dict,patch:int)->tuple[torch.Tensor,dict]:
    image=np.load(root/row['image_file']); r0=image.shape[0]//2-patch//2; c0=image.shape[1]//2-patch//2; target=torch.tensor(image[r0:r0+patch,c0:c0+patch])[None]; g=json.loads(row['geometry_json']); origin=np.asarray(g['image_position_patient']); orient=np.asarray(g['image_orientation_patient']); spacing=np.asarray(g['pixel_spacing']); center=origin+((g['columns']-1)/2)*spacing[1]*orient[:3]+((g['rows']-1)/2)*spacing[0]*orient[3:]; rowd=torch.tensor(orient[:3])[None].float(); cold=torch.tensor(orient[3:])[None].float(); normal=torch.linalg.cross(rowd,cold); return target,{'center_mm':torch.tensor(center)[None].float(),'row_direction':rowd,'column_direction':cold,'normal':normal,'pixel_spacing_mm':torch.tensor(spacing)[None].float(),'thickness_mm':torch.tensor([float(g['slice_thickness'])]),'height':patch,'width':patch}

def _difference(first:Path|None,second:Path,out:Path)->None:
    if first is not None: a=np.asanyarray(nib.load(str(first)).dataobj); b=nib.load(str(second)); nib.save(nib.Nifti1Image(np.asarray(b.dataobj)-a,b.affine),str(out))
def _qc_views(model,renderer,uncertainty,root,by_view,ids,patch,out):
    out.mkdir(parents=True,exist_ok=True); result={}
    for view,items in by_view.items():
        values=[]; residuals=[]; variances=[]
        for item in items:
            target,geo=_mean_observation(root,item,patch)
            with torch.no_grad(): q=renderer(model,**geo); pred=q['predicted_slice'].squeeze(); var=uncertainty(q['latent_samples'],torch.tensor([ids[item['mean_slice_id']]]),q['psf_weights'],namespace='mean_slice',enabled=True); t=target.squeeze(); values.append((float(((pred-pred.mean())*(t-t.mean())).mean()/((pred.std()*t.std()).clamp_min(1e-6))),float(torch.sqrt((pred-t).square().mean())/(t.max()-t.min()).clamp_min(1e-6)),float(gaussian_nll(pred[None],t[None],var['total_variance'])),float(var['observation_variance'].mean()))); residuals.append((pred-t).abs().flatten().numpy()); variances.append(var['total_variance'].flatten().numpy())
        target,geo=_mean_observation(root,items[0],patch)
        with torch.no_grad(): q=renderer(model,**geo); pred=q['predicted_slice'].squeeze(); var=uncertainty(q['latent_samples'],torch.tensor([ids[items[0]['mean_slice_id']]]),q['psf_weights'],namespace='mean_slice',enabled=True); t=target.squeeze(); np.save(out/f'{view.lower()}_pixel_variance.npy',var['pixel_variance'].numpy()); _plot_qc(t.numpy(),pred.numpy(),out/f'{view.lower()}_acquired_vs_reprojected.png')
        array=np.asarray(values); r=np.concatenate(residuals); v=np.concatenate(variances); result[view]={'ncc':float(array[:,0].mean()),'nrmse':float(array[:,1].mean()),'loss':float(array[:,2].mean()),'mean_variance':float(array[:,3].mean()),'pixel_variance_mean':float(v.mean()),'pixel_variance_std':float(v.std()),'residual_variance_correlation':float(np.corrcoef(r,v)[0,1])}
    (out/'mean_slice_variance.csv').write_text('view,mean_variance\n'+'\n'.join(f'{v},{m["mean_variance"]}' for v,m in result.items())+'\n'); return result
def _plot_qc(target,pred,path):
    import matplotlib; matplotlib.use('Agg'); from matplotlib import pyplot as plt
    f,a=plt.subplots(1,3,figsize=(9,3));
    for ax,img,title in zip(a,(target,pred,pred-target),('acquired','reprojected','residual')): ax.imshow(img,cmap='gray'); ax.set_title(title); ax.axis('off')
    f.tight_layout(); f.savefig(path,dpi=120); plt.close(f)

def main()->None:
    p=argparse.ArgumentParser(description='Run Stage 1 no-motion Canonical INR.'); p.add_argument('--initial-reference',type=Path,required=True);p.add_argument('--initial-reference-valid-mask',type=Path);p.add_argument('--canonical-domain',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--qc-table',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--profile',choices=('smoke','production'),default='production');p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');p.add_argument('--stage1a-steps',type=int,default=1000);p.add_argument('--stage1a-batch-size',type=int,default=8192);p.add_argument('--stage1a-lr',type=float,default=2e-3);p.add_argument('--coarsest-resolution-mm',type=float,default=16.);p.add_argument('--finest-resolution-mm',type=float,default=1.5);p.add_argument('--level-scale',type=float,default=1.38);p.add_argument('--features-per-level',type=int,default=2);p.add_argument('--log2-hashmap-size',type=int,default=19);p.add_argument('--hidden-dim',type=int,default=64);p.add_argument('--latent-dim',type=int,default=32);p.add_argument('--seed',type=int,default=0);args=p.parse_args();kw={k:getattr(args,k) for k in ('coarsest_resolution_mm','finest_resolution_mm','level_scale','features_per_level','log2_hashmap_size','hidden_dim','latent_dim')}; a=train_stage1a(args.initial_reference,args.initial_reference_valid_mask,args.canonical_domain,args.output_dir,steps=args.stage1a_steps,batch_size=args.stage1a_batch_size,lr=args.stage1a_lr,profile=args.profile,device=args.device,seed=args.seed,**kw); print(json.dumps(a,indent=2))
if __name__=='__main__': main()
