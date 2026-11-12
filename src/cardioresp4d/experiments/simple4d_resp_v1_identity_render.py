"""Read-only, chunked world-coordinate rendering for identity audit."""
from pathlib import Path
import json
import numpy as np
import torch
import nibabel as nib
from third_party.simple4d.simple4d.losses import jacobian_matrix
from .simple4d_resp_v1_identity_audit import interpolate_neighbor_codes


def choose_anchor_states(locations, view):
    candidates=[loc for loc in locations if loc['view']==view and len(loc['frames'])>=5]
    if not candidates: return None
    loc=min(candidates,key=lambda l:(abs(l['frames'][0]['normalized_slice_position']),str(l['slice_id'])))
    frames=loc['frames']; scores=np.array([r['pc1'] for r in frames])
    return {state:min(frames,key=lambda row:(abs(row['pc1']-np.quantile(scores,q)),row['timestamp_s'],row['dynamic_frame_id'])) for state,q in (('low',.1),('mid',.5),('high',.9))}


def fixed_pixels(image, count=256):
    h,w=image.shape[-2:]; idx=torch.linspace(0,h*w-1,count,device=image.device).round().long()
    return torch.stack((idx%w,torch.div(idx,w,rounding_mode='floor')),-1)


def dense_field(model, code, points, chunk_size):
    """Evaluate forward DVF, unclamped Jacobian and inverse cycle at points."""
    coords=model.forward_deformation.coordinates
    fields=[]; dets=[]; cycles=[]
    for chunk in points.split(chunk_size):
        x=coords.world_to_normalized(chunk[None]).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            _,dvf=model.forward_deformation.forward_normalized(x,code.detach())
            determinant=torch.linalg.det(jacobian_matrix(x,dvf))
            xc=(x+dvf).detach()
            _,inv=model.inverse_deformation.forward_normalized(xc,code.detach())
            error=torch.linalg.vector_norm(coords.normalized_deformation_to_world_mm(xc+inv-x),dim=-1)
            mm=coords.normalized_deformation_to_world_mm(dvf)
        fields.append(mm.detach().cpu()); dets.append(determinant.detach().cpu()); cycles.append(error.detach().cpu())
    return torch.cat(fields,1)[0].numpy(),torch.cat(dets,1)[0].numpy(),torch.cat(cycles,1)[0].numpy()


def volume_grid(bounds_lower,bounds_upper,spacing):
    lower=np.asarray(bounds_lower,dtype=float); upper=np.asarray(bounds_upper,dtype=float); spacing=np.asarray(spacing,dtype=float)
    if np.any(spacing<=0) or np.any(upper<=lower): raise ValueError('invalid volume domain or spacing')
    shape=np.floor((upper-lower)/spacing+1e-6).astype(int)+1
    return lower,spacing,tuple(int(x) for x in shape)


def render_volume(model,code,lower,spacing,shape,chunk_size):
    device=code.device; total=int(np.prod(shape)); output=np.empty(total,dtype=np.float32)
    lo=torch.as_tensor(lower,dtype=code.dtype,device=device); sp=torch.as_tensor(spacing,dtype=code.dtype,device=device)
    with torch.no_grad():
        for start in range(0,total,chunk_size):
            index=torch.arange(start,min(start+chunk_size,total),device=device)
            xyz=torch.stack((index//(shape[1]*shape[2]),(index//shape[2])%shape[1],index%shape[2]),-1)
            world=lo+xyz*sp; reference=model.forward_deformation(world[None],code)['reference_points_mm']
            output[start:start+len(index)]=model.canonical(reference,return_features=True)[0].reshape(-1).detach().cpu().numpy()
    return output.reshape(shape)


def save_lps_nifti(path,volume,lower,spacing):
    """NIfTI affine maps voxel indices to RAS; tensors remain external LPS xyz."""
    flip=np.diag([-1.,-1.,1.,1.]); lps=np.eye(4); lps[:3,:3]=np.diag(spacing); lps[:3,3]=lower
    nib.save(nib.Nifti1Image(volume,flip@lps),str(path))


def withheld_codes(prev,next_,prev_t,target_t,next_t):
    """Compute V2 solely from neighboring posterior means."""
    return interpolate_neighbor_codes(prev,next_,prev_t,target_t,next_t)


def dvf_stage(model,rows,dest,step,checkpoint_sha,chunk_size):
    from .simple4d_resp_v1_identity_stages import latent_locations,write_csv,json_write
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True); locations=latent_locations(rows); selected=[]; state_stats=[]; arrays={}; volumes={}
    coords=model.forward_deformation.coordinates; lower=coords.lower_world_mm.detach().cpu().numpy(); upper=lower+coords.extent_mm.detach().cpu().numpy(); spacing=coords.reference_spacing_mm.detach().cpu().numpy()
    device=coords.lower_world_mm.device
    axes=[torch.linspace(float(a),float(b),16,device=device) for a,b in zip(lower,upper)]
    world=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    for view in ('SAX','2CH','4CH'):
        states=choose_anchor_states(locations,view)
        if states is None: continue
        for state,row in states.items():
            obs=row['observation']
            with torch.no_grad(): code=model.encode_observation(obs,sample=False)['mu'].detach()
            field,det,cycle=dense_field(model,code,world,chunk_size)
            field=field.reshape(16,16,16,3); magnitude=np.linalg.norm(field,axis=-1)
            arrays[(view,state)]=field
            selected.append({'checkpoint_step':step,'view':view,'slice_id':row['slice_id'],'state':state,'dynamic_frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'pc1':row['pc1'],'pc1_percentile_target':{'low':.1,'mid':.5,'high':.9}[state]})
            state_stats.append({'checkpoint_step':step,'view':view,'slice_id':row['slice_id'],'state':state,'timestamp_s':row['timestamp_s'],'pc1':row['pc1'],'mean_dvf_mm':float(magnitude.mean()),'rms_dvf_mm':float(np.sqrt(np.mean(magnitude**2))),'p95_dvf_mm':float(np.quantile(magnitude,.95)),'max_dvf_mm':float(magnitude.max()),'jacobian_mean':float(det.mean()),'jacobian_p01':float(np.quantile(det,.01)),'jacobian_min':float(det.min()),'folding_fraction':float(np.mean(det<=0)),'cycle_mean_mm':float(cycle.mean()),'cycle_p95_mm':float(np.quantile(cycle,.95)),'cycle_max_mm':float(cycle.max())})
            if view=='SAX':
                origin,sp,shape=volume_grid(lower,upper,spacing); vol=render_volume(model,code,origin,sp,shape,chunk_size); volumes[state]=vol
                vdir=dest/'volumes'; vdir.mkdir(exist_ok=True); path=vdir/f'SAX_{state}_pc1.nii.gz'; save_lps_nifti(path,vol,origin,sp)
                json_write(vdir/f'SAX_{state}_pc1.json',{'checkpoint':step,'checkpoint_sha256':checkpoint_sha,'view':view,'slice_id':row['slice_id'],'dynamic_frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'pc1':row['pc1'],'world_bounds_lps_mm':[lower.tolist(),upper.tolist()],'spacing_mm':sp.tolist(),'axis_convention':'array xyz in external patient LPS; NIfTI affine maps indices to RAS','rendering_convention':'external observation world -> forward deformation -> frozen canonical'})
    if not selected: raise ValueError('DVF stage requires a location with at least five frames')
    write_csv(dest/'selected_states.csv',selected); write_csv(dest/'dvf_state_summary.csv',state_stats)
    np.savez_compressed(dest/'dvf_plane_data.npz',**{f'{v}_{s}':a for (v,s),a in arrays.items()},lower_lps_mm=lower,upper_lps_mm=upper)
    vmax=max(float(np.linalg.norm(a,axis=-1).max()) for a in arrays.values()) or 1.
    for view in ('SAX','2CH','4CH'):
        if (view,'low') not in arrays: continue
        fig,axes_plot=plt.subplots(3,3,figsize=(12,11))
        for i,state in enumerate(('low','mid','high')):
            a=arrays[(view,state)]; info=next(r for r in state_stats if r['view']==view and r['state']==state)
            for j,(plane,components) in enumerate(((a[:,:,8,:],(0,1)),(a[:,8,:,:],(0,2)),(a[8,:,:,:],(1,2)))):
                ax=axes_plot[i,j]; mag=np.linalg.norm(plane,axis=-1); ax.imshow(mag.T,origin='lower',vmin=0,vmax=vmax,cmap='viridis'); q=np.arange(0,16,3); x,y=np.meshgrid(q,q,indexing='ij'); ax.quiver(x,y,plane[x,y,components[0]],plane[x,y,components[1]],color='white',scale=1); ax.set_title(f"{step} {view}/{info['slice_id']} {state} t={info['timestamp_s']:.2f} PC1={info['pc1']:.2f}\nRMS={info['rms_dvf_mm']:.2f} max={info['max_dvf_mm']:.2f} mm")
        fig.savefig(dest/f'{view}_dvf_states.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    if len(volumes)==3:
        fig,panels=plt.subplots(3,3,figsize=(10,10))
        for i,state in enumerate(('low','mid','high')):
            vol=volumes[state]; mid=np.array(vol.shape)//2
            for j,plane in enumerate((vol[:,:,mid[2]],vol[:,mid[1],:],vol[mid[0],:,:])):
                panels[i,j].imshow(plane.T,origin='lower',cmap='gray',vmin=0,vmax=1); panels[i,j].set_title(f'{state} {("axial","coronal","sagittal")[j]} (LPS axes)')
        fig.savefig(dest/'01_dynamic_volume_low_mid_high.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        diff=volumes['high']-volumes['low']; mid=np.array(diff.shape)//2; vmax_diff=float(np.max(np.abs(diff))) or 1.
        fig,panels=plt.subplots(1,3,figsize=(10,3))
        for ax,plane in zip(panels,(diff[:,:,mid[2]],diff[:,mid[1],:],diff[mid[0],:,:])): ax.imshow(plane.T,origin='lower',cmap='coolwarm',vmin=-vmax_diff,vmax=vmax_diff)
        fig.savefig(dest/'02_dynamic_volume_high_minus_low.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    return state_stats


def predict_modes(model,obs,pixels,codes,chunk_size):
    from cardioresp4d.training.model import SourceFirstDynamicModel
    device=obs.image.device; outputs=[[] for _ in range(4)]
    with torch.no_grad():
        for uv in pixels.split(chunk_size):
            world=SourceFirstDynamicModel._pixel_world(obs,uv.to(obs.image)); n=len(world)
            res=obs.image.new_tensor([obs.pixel_spacing_mm[1],obs.pixel_spacing_mm[0],obs.slice_thickness_mm]).expand(n,3)
            samples=model.psf.sample(world,res,row_direction=obs.row_direction.to(obs.image).expand(n,-1),column_direction=obs.column_direction.to(obs.image).expand(n,-1),normal=obs.normal.to(obs.image).expand(n,-1))
            for i,code in enumerate((None,*codes)):
                reference=samples if code is None else model.forward_deformation(samples,code.expand(n,-1))['reference_points_mm']
                pred=model.canonical(reference,return_features=True)[0].mean(-1)
                outputs[i].append(pred.cpu())
    return [torch.cat(x).numpy() for x in outputs]


def withheld_stage(model,rows,dest,step,chunk_size):
    from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import diagnostic_rng_context
    from .simple4d_resp_v1_identity_stages import latent_locations,write_csv,json_write
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True); locations=latent_locations(rows); records=[]; examples={}
    with diagnostic_rng_context():
        for loc in locations:
            frames=loc['frames']
            if len(frames)<3: continue
            with torch.no_grad(): codes=[model.encode_observation(f['observation'],sample=False)['mu'].detach() for f in frames]
            candidate=[]
            for i in range(1,len(frames)-1):
                prev,cur,nxt=frames[i-1:i+2]
                if not prev['timestamp_s'] < cur['timestamp_s'] < nxt['timestamp_s'] or nxt['timestamp_s']-prev['timestamp_s'] <= 1e-9: continue
                interpolated=withheld_codes(codes[i-1].cpu().numpy(),codes[i+1].cpu().numpy(),prev['timestamp_s'],cur['timestamp_s'],nxt['timestamp_s'])
                alpha=interpolated['alpha']
                simple=torch.as_tensor(interpolated['simple'],device=codes[i].device,dtype=codes[i].dtype); timed=torch.as_tensor(interpolated['time'],device=codes[i].device,dtype=codes[i].dtype)
                obs=cur['observation']; uv=fixed_pixels(obs.image); pred=predict_modes(model,obs,uv,(codes[i],simple,timed),chunk_size)
                target=obs.image[0,uv[:,1],uv[:,0]].detach().cpu().numpy()
                metrics={}
                for label,array in zip(('static','v1','v2_simple','v2_time'),pred):
                    error=target-array; metrics[label+'_mse']=float(np.mean(error**2)); metrics[label+'_mae']=float(np.mean(np.abs(error)))
                def improvement(a,b): return 100*(a-b)/a if a else None
                r={'checkpoint_step':step,'view':loc['view'],'slice_id':loc['slice_id'],'dynamic_frame_id':cur['dynamic_frame_id'],'timestamp_s':cur['timestamp_s'],'prev_timestamp_s':prev['timestamp_s'],'next_timestamp_s':nxt['timestamp_s'],'alpha':alpha,**metrics,'v1_vs_static_improvement_percent':improvement(metrics['static_mse'],metrics['v1_mse']),'v2_time_vs_static_improvement_percent':improvement(metrics['static_mse'],metrics['v2_time_mse']),'v2_time_vs_v1_change_percent':improvement(metrics['v1_mse'],metrics['v2_time_mse'])}
                records.append(r); candidate.append((i,cur,codes[i],simple,timed))
            if candidate: examples[(loc['view'],loc['slice_id'])]=(loc,candidate)
        if not records: raise ValueError('no targets with strictly increasing neighbor timestamps')
        write_csv(dest/'target_withheld_per_frame.csv',records)
        summary={}
        for view in ('SAX','2CH','4CH','ALL'):
            subset=[r for r in records if view=='ALL' or r['view']==view]; result={'frame_count':len(subset)}
            for label in ('static','v1','v2_time'):
                for metric in ('mse','mae'):
                    values=[r[label+'_'+metric] for r in subset]
                    result[f'mean_{label}_{metric}']=float(np.mean(values)) if values else None; result[f'median_{label}_{metric}']=float(np.median(values)) if values else None
            for key,condition in [('fraction_v2_time_better_than_static',lambda r:r['v2_time_mse']<r['static_mse']),('fraction_v2_time_better_than_v1',lambda r:r['v2_time_mse']<r['v1_mse'])]: result[key]=sum(condition(r) for r in subset)/len(subset) if subset else None
            for key in ('v2_time_vs_static_improvement_percent','v2_time_vs_v1_change_percent'):
                values=[r[key] for r in subset if r[key] is not None]; result['median_'+key]=float(np.median(values)) if values else None
            summary[view]=result
        json_write(dest/'summary.json',summary); write_csv(dest/'summary.csv',[{'view':view,**vals} for view,vals in summary.items()])
        fig,ax=plt.subplots(); views=('SAX','2CH','4CH'); x=np.arange(3)
        for i,label in enumerate(('static','v1','v2_time')): ax.bar(x+(i-1)*.25,[summary[v]['mean_'+label+'_mse'] if summary[v]['mean_'+label+'_mse'] is not None else np.nan for v in views],width=.25,label=label)
        ax.set(xticks=x,xticklabels=views,ylabel='Mean sampled MSE'); ax.legend(); fig.savefig(dest/'01_metric_comparison_by_view.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots(); values=[r['v2_time_vs_static_improvement_percent'] for r in records if r['v2_time_vs_static_improvement_percent'] is not None]; ax.hist(values,bins=30); ax.set(xlabel='V2 time vs static improvement (%)',ylabel='Frames'); fig.savefig(dest/'02_per_frame_v2_improvement_distribution.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots(); ax.scatter([r['v1_mse'] for r in records],[r['v2_time_mse'] for r in records],s=8); ax.set(xlabel='V1 MSE',ylabel='V2 time MSE'); fig.savefig(dest/'03_v1_vs_v2_scatter.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        exdir=dest/'examples'; exdir.mkdir()
        for view in ('SAX','2CH','4CH'):
            options=[value for (v,_),value in examples.items() if v==view]
            if not options: continue
            loc,candidates=min(options,key=lambda item:(abs(item[0]['frames'][0]['normalized_slice_position']),str(item[0]['slice_id'])))
            chosen=[candidates[round((len(candidates)-1)*q)] for q in (.25,.5,.75)]
            all_panels=[]
            for _,cur,self_code,simple,timed in chosen:
                obs=cur['observation']; h,w=obs.image.shape[-2:]; index=torch.arange(h*w,device=obs.image.device); uv=torch.stack((index%w,torch.div(index,w,rounding_mode='floor')),-1)
                preds=predict_modes(model,obs,uv,(self_code,simple,timed),chunk_size); target=obs.image[0].detach().cpu().numpy(); static,v1,_,v2=[x.reshape(h,w) for x in preds]
                all_panels.append((target,static,v1,v2,np.abs(target-static),np.abs(target-v1),np.abs(target-v2)))
            vmax=max(float(panel.max()) for group in all_panels for panel in group[4:]) or 1.
            np.savez_compressed(exdir/f'{view}_examples_data.npz',panels=np.asarray(all_panels),frame_ids=np.asarray([c[1]['dynamic_frame_id'] for c in chosen]),timestamps=np.asarray([c[1]['timestamp_s'] for c in chosen]))
            fig,axes=plt.subplots(3,7,figsize=(18,8))
            for i,group in enumerate(all_panels):
                for j,panel in enumerate(group):
                    axes[i,j].imshow(panel,cmap='gray' if j<4 else 'magma',vmin=0,vmax=1 if j<4 else vmax); axes[i,j].axis('off')
                    label=('Target','Static','V1','V2 time','|static error|','|V1 error|','|V2 error|')[j]
                    if 1 <= j <= 3:
                        err=group[0]-panel; label+=f'\nMSE={np.mean(err**2):.3g} MAE={np.mean(np.abs(err)):.3g}'
                    axes[i,j].set_title(label,fontsize=8)
            fig.savefig(exdir/f'{view}_examples.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    return summary


def compare_sax_fields(dest,steps):
    import matplotlib.pyplot as plt
    data={}
    for step in steps:
        path=Path(dest)/f'step_{step}'/'dvf_plane_data.npz'
        if path.exists():
            with np.load(path) as a:
                if all(f'SAX_{state}' in a for state in ('low','mid','high')):
                    data[step]={state:a[f'SAX_{state}'] for state in ('low','mid','high')}
    if not data: return
    vmax=max(float(np.linalg.norm(field,axis=-1).max()) for states in data.values() for field in states.values()) or 1.
    fig,axes=plt.subplots(len(data),3,figsize=(10,3*len(data)),squeeze=False)
    for i,(step,states) in enumerate(sorted(data.items())):
        for j,state in enumerate(('low','mid','high')):
            axes[i,j].imshow(np.linalg.norm(states[state][:,:,8,:],axis=-1).T,origin='lower',vmin=0,vmax=vmax,cmap='viridis'); axes[i,j].set_title(f'{step} SAX {state} |DVF| mm')
    fig.savefig(Path(dest)/'comparison_SAX_low_mid_high.png',dpi=180,bbox_inches='tight'); plt.close(fig)
