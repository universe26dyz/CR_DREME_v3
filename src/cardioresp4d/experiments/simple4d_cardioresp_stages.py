"""Read-only execution helpers for cardiorespiratory identity comparisons."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from .simple4d_cardioresp_identity import pure_pc1_codes, difference_summary, vector_difference_summary, decompose_motion
from .simple4d_resp_v1_identity_render import render_volume, save_lps_nifti, dense_field
from .simple4d_resp_v1_identity_audit import periodicity_analysis
from .simple4d_resp_v1_anatomy import crop_points, support_at_world
from .simple4d_resp_v1_identity_stages import write_csv, json_write
from .simple4d_resp_v1_dynamic_visualization import save_triplanar, make_gif


def crop_masks(crop,support,ras_to_voxel,device,chunk_size,cardiac_box=None):
    shape=tuple(crop['shape']); n=int(np.prod(shape)); anatomy=np.empty(n,dtype=bool); cardiac=np.empty(n,dtype=bool) if cardiac_box else None
    for begin in range(0,n,chunk_size):
        end=min(begin+chunk_size,n); world=crop_points(crop['lower_lps_mm'],crop['spacing_mm'],shape,begin,end,device).cpu().numpy()
        anatomy[begin:end]=support_at_world(world,support,ras_to_voxel)
        if cardiac is not None: cardiac[begin:end]=np.all((world>=cardiac_box.min_mm)&(world<=cardiac_box.max_mm),axis=1)
    if not anatomy.any(): raise ValueError('no anatomy support on audit grid')
    return {'full_crop':np.ones(n,dtype=bool),'anatomy_support':anatomy,'cardiac_box':cardiac}


def all_fields(model,code,crop,chunk_size,*,jacobian=False):
    shape=tuple(crop['shape']); n=int(np.prod(shape)); field=np.empty((n,3),dtype=np.float32); det=np.empty(n,dtype=np.float32) if jacobian else None
    for begin in range(0,n,chunk_size):
        end=min(n,begin+chunk_size); points=crop_points(crop['lower_lps_mm'],crop['spacing_mm'],shape,begin,end,code.device)
        if jacobian:
            f,d,_=dense_field(model,code,points,chunk_size); field[begin:end]=f; det[begin:end]=d
        else:
            with torch.no_grad(): field[begin:end]=model.forward_deformation(points[None],code)['displacement_mm'].reshape(-1,3).cpu().numpy()
    return field,det


def field_difference_magnitudes(model,full_code,pure_code,crop,chunk_size):
    shape=tuple(crop['shape']); n=int(np.prod(shape)); magnitude=np.empty(n,dtype=np.float32)
    with torch.no_grad():
        for begin in range(0,n,chunk_size):
            end=min(n,begin+chunk_size); points=crop_points(crop['lower_lps_mm'],crop['spacing_mm'],shape,begin,end,full_code.device)[None]
            full=model.forward_deformation(points,full_code)['displacement_mm']
            pure=model.forward_deformation(points,pure_code)['displacement_mm']
            magnitude[begin:end]=torch.linalg.vector_norm(full-pure,dim=-1).reshape(-1).cpu().numpy()
    return magnitude


def magnitude_summary(values,mask):
    selected=values[mask]
    if not len(selected): return None
    return {'mean_magnitude_mm':float(selected.mean()),'rms_magnitude_mm':float(np.sqrt(np.mean(selected.astype(np.float64)**2))),'p95_magnitude_mm':float(np.quantile(selected,.95)),'max_magnitude_mm':float(selected.max())}

def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def dynamic_contract(a,info):
    path=a.audit_root/'04_dvf'/f"step_{info['step']}"/'dynamic_visualization'; manifest=json.loads((path/'visualization_manifest.json').read_text())
    if manifest['checkpoint_sha256']!=info['checkpoint_sha256'] or manifest['crop']!=info['crop'] or manifest['anchor_slice_id']!=info['anchor'] or manifest['intensity_window']!=[0.,1.]: raise ValueError('full-latent dynamic visualization provenance mismatch')
    if info.get('latent_score_sha256') is not None and manifest.get('sequence',{}).get('latent_score_sha256') != info['latent_score_sha256']: raise ValueError('dynamic visualization latent-score hash mismatch')
    with (path/'sequence_metadata.csv').open(newline='') as f: metadata=list(csv.DictReader(f))
    rows=info['anchor_rows']
    if len(metadata)!=len(rows) or any(int(x['dynamic_frame_id'])!=y['dynamic_frame_id'] or abs(float(x['timestamp_s'])-y['timestamp_s'])>1e-6 for x,y in zip(metadata,rows)):
        raise ValueError('full-latent sequence does not match anchor frames')
    source_paths=[path/'visualization_manifest.json',path/'sequence_metadata.csv']+[path/'sequence_frames'/f'frame_{i:04d}.png' for i in range(len(metadata))]+[path/'selected_states'/f'SAX_{state}.png' for state in ('low','mid','high')]
    return manifest,metadata,{str(p):file_hash(p) for p in source_paths}


def pair_png(left,right,path):
    with Image.open(left) as a, Image.open(right) as b:
        if a.size!=b.size: raise ValueError('full and pure frame sizes differ')
        canvas=Image.new('RGB',(a.width+b.width,a.height),'white'); canvas.paste(a.convert('RGB'),(0,0)); canvas.paste(b.convert('RGB'),(a.width,0)); draw=ImageDraw.Draw(canvas); draw.text((8,5),'FULL LATENT',fill='red'); draw.text((a.width+8,5),'PURE PC1',fill='red'); canvas.save(path)


def pure_stage(a,base,info,support,ras_to_voxel):
    step=info['step']; crop=info['crop']; model=base['model']; rows=info['anchor_rows']; scores=info['anchor_scores']; means=info['anchor_means']
    vis,_,source_hashes=dynamic_contract(a,info); indices=tuple(vis['slice_indices_lps_xyz']); spacing=crop['spacing_mm']; gif_ms=vis['gif_frame_ms']
    pure=pure_pc1_codes(means,scores,info['basis']); shape=tuple(crop['shape']); device=next(model.parameters()).device
    masks=crop_masks(crop,support,ras_to_voxel,device,a.chunk_size,base['cardiac_box'])
    out=a.output_root/f'step_{step}'/'pure_pc1'; selected_dir=out/'selected_states'; frames_dir=out/'sequence_frames'; gif_dir=out/'gif'; volumes_dir=out/'volumes'; pair_dir=out/'comparison_frames'
    for directory in (selected_dir,frames_dir,gif_dir,volumes_dir,pair_dir): directory.mkdir(parents=True,exist_ok=True)
    meta=[]; contribution=[]; pngs=[]; pair_pngs=[]; selected_paths={}
    for index,(row,full_code,pure_code) in enumerate(zip(rows,means,pure)):
        full_tensor=torch.as_tensor(full_code,dtype=torch.float32,device=device)[None]; pure_tensor=torch.as_tensor(pure_code,dtype=torch.float32,device=device)[None]
        full_volume=render_volume(model,full_tensor,crop['lower_lps_mm'],spacing,shape,a.chunk_size)
        pure_volume=render_volume(model,pure_tensor,crop['lower_lps_mm'],spacing,shape,a.chunk_size)
        difference_magnitude=field_difference_magnitudes(model,full_tensor,pure_tensor,crop,a.chunk_size)
        for region,mask in masks.items():
            if mask is None: continue
            v=difference_summary(full_volume.reshape(-1),pure_volume.reshape(-1),mask); d=magnitude_summary(difference_magnitude,mask)
            if v is not None: contribution.append({'checkpoint_step':step,'dynamic_frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'region':region,**{'volume_'+k:v1 for k,v1 in v.items()},**{'dvf_'+k:v1 for k,v1 in d.items()}})
        rel=row['timestamp_s']-rows[0]['timestamp_s']; png=frames_dir/f'frame_{index:04d}.png'
        title=f"step {step} | SAX {info['anchor']} | frame {index:04d} | t={rel:.3f}s | PC1={row['pc1']:.3f}"
        save_triplanar(pure_volume,indices,png,'PURE PC1 | '+title,spacing=spacing); pngs.append(png)
        full_path=a.audit_root/'04_dvf'/f'step_{step}'/'dynamic_visualization'/'sequence_frames'/f'frame_{index:04d}.png'
        pair=pair_dir/f'frame_{index:04d}.png'; pair_png(full_path,png,pair); pair_pngs.append(pair)
        meta.append({'frame_index':index,'checkpoint_step':step,'view':'SAX','slice_id':info['anchor'],'dynamic_frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'relative_time_s':rel,'pc1':row['pc1']})
        for state,frame_id in info['selected'].items():
            if row['dynamic_frame_id']==frame_id:
                selected=selected_dir/f'SAX_{state}_pure_pc1.png'; save_triplanar(pure_volume,indices,selected,f'step {step} | {state} | PURE PC1 | t={row["timestamp_s"]:.3f}s | PC1={row["pc1"]:.3f}',spacing=spacing)
                selected_paths[state]=selected
                save_lps_nifti(volumes_dir/f'SAX_{state}_pure_pc1.nii.gz',pure_volume,crop['lower_lps_mm'],spacing)
    if set(selected_paths)!={'low','mid','high'}: raise ValueError('missing selected pure-PC1 states')
    write_csv(out/'sequence_metadata.csv',meta); write_csv(out/'non_pc1_contribution.csv',contribution)
    write_csv(out/'pure_pc1_selected_states.csv',[{'state':state,**next(r for r in meta if r['dynamic_frame_id']==fid)} for state,fid in info['selected'].items()])
    json_write(out/'non_pc1_summary.json',{region:{key:float(np.mean([r[key] for r in contribution if r['region']==region])) for key in ('volume_mae','volume_rmse','volume_p95_abs_difference','dvf_mean_magnitude_mm','dvf_rms_magnitude_mm','dvf_p95_magnitude_mm','dvf_max_magnitude_mm')} for region in sorted({r['region'] for r in contribution})})
    make_gif(pngs,gif_dir/'SAX_pure_pc1_temporal_sequence.gif',gif_ms)
    make_gif(pair_pngs,gif_dir/'SAX_full_vs_pure_pc1_temporal.gif',gif_ms)
    make_gif([selected_paths[x] for x in ('low','mid','high','mid','low')],gif_dir/'SAX_pure_pc1_selected_states_pingpong.gif',max(gif_ms,300))
    selected_images=[Image.open(selected_paths[state]).convert('RGB') for state in ('low','mid','high')]
    canvas=Image.new('RGB',(selected_images[0].width,selected_images[0].height*3),'white')
    for i,image in enumerate(selected_images): canvas.paste(image,(0,i*image.height))
    canvas.save(selected_dir/'SAX_low_mid_high_pure_pc1_comparison.png')
    full_dir=a.audit_root/'04_dvf'/f'step_{step}'/'dynamic_visualization'/'selected_states'
    pairs=[]
    for state in ('low','mid','high'):
        path=selected_dir/f'_full_vs_pure_{state}.png'; pair_png(full_dir/f'SAX_{state}.png',selected_paths[state],path); pairs.append(path)
    ims=[Image.open(p).convert('RGB') for p in pairs]; comp=Image.new('RGB',(ims[0].width,ims[0].height*3),'white')
    for i,image in enumerate(ims): comp.paste(image,(0,i*image.height))
    comp.save(selected_dir/'SAX_full_vs_pure_pc1_selected_states.png')
    for path in pairs: path.unlink()
    if any(file_hash(path)!=digest for path,digest in source_hashes.items()): raise RuntimeError('existing dynamic visualization changed during audit')
    json_write(out/'pca_provenance.json',{'full_visualization_source_sha256':source_hashes,'basis_array_sha256':info['basis_sha256'],'explained_variance_ratios':info['pca']['explained_variance_ratios'].tolist(),'rank95':info['pca']['rank95'],'sign_convention':'largest-magnitude loading positive','stored_diagnostics':info.get('checkpoint_data',{}).get('diagnostics',{}).get(f'step_{step}',{}),'checkpoint_sha256':info['checkpoint_sha256'],'anchor_mean':means.mean(0).tolist()})


def motion_stage(a,base,info,support,ras_to_voxel):
    step=info['step']; crop=info['crop']; model=base['model']; device=next(model.parameters()).device
    masks=crop_masks(crop,support,ras_to_voxel,device,a.chunk_size,base['cardiac_box']); rows=[]
    for frame,code in zip(info['anchor_rows'],info['anchor_means']):
        field,det=all_fields(model,torch.as_tensor(code,dtype=torch.float32,device=device)[None],crop,a.chunk_size,jacobian=True)
        for region in ('anatomy_support','cardiac_box'):
            mask=masks[region]
            if mask is None: continue
            result=decompose_motion(field,det,mask)
            if result is not None: rows.append({'checkpoint_step':step,'slice_id':info['anchor'],'dynamic_frame_id':frame['dynamic_frame_id'],'timestamp_s':frame['timestamp_s'],'pc1':frame['pc1'],'region':region,**result})
    out=a.output_root/f'step_{step}'/'motion_decomposition'; out.mkdir(parents=True,exist_ok=True); write_csv(out/'frame_motion_decomposition.csv',rows)
    import matplotlib.pyplot as plt
    for name,keys,title in [('translation_xyz_vs_time.png',('tx_mm','ty_mm','tz_mm'),'Translation LPS xyz (mm)'),('translation_magnitude_vs_time.png',('translation_magnitude_mm',),'Translation magnitude (mm)'),('nonrigid_rms_vs_time.png',('residual_rms_mm',),'Non-rigid residual RMS (mm)'),('jacobian_mean_vs_time.png',('jacobian_mean',),'Mean Jacobian'),('jacobian_deviation_vs_time.png',('jacobian_mean_abs_deviation',),'Mean |J−1|')]:
        fig,ax=plt.subplots()
        for region in ('anatomy_support','cardiac_box'):
            series=[r for r in rows if r['region']==region]
            for key in keys:
                if series: ax.plot([r['timestamp_s'] for r in series],[r[key] for r in series],label=f'{region} {key}')
        ax.set(xlabel='Timestamp (s)',ylabel=title); ax.legend(fontsize=7); fig.savefig(out/name,dpi=180,bbox_inches='tight'); plt.close(fig)
    spectra={}
    for region in ('anatomy_support','cardiac_box'):
        series=[r for r in rows if r['region']==region]
        if not series: continue
        spectra[region]={}
        for key in ('tz_mm','translation_magnitude_mm','residual_rms_mm','jacobian_mean_abs_deviation','pc1'):
            info_spectrum,_=periodicity_analysis([r['timestamp_s'] for r in series],[r[key] for r in series]); spectra[region][key]=info_spectrum
    json_write(out/'frequency_summary.json',spectra)


def residual_stage(a,base,steps,load_step):
    """Matched full-slice PSF samples, static/V1/V2 predictions for fixed cohort."""
    from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import representative_locations
    from cardioresp4d.experiments.simple4d_cardioresp_identity import residual_metrics
    from cardioresp4d.training.model import SourceFirstDynamicModel
    from .simple4d_resp_v1_identity_render import fixed_pixels
    from .simple4d_resp_v1_identity_audit import interpolate_neighbor_codes
    from .respiratory_s2vdreme_reference_v1 import diagnostic_rng_context
    import matplotlib.pyplot as plt
    model=base['model']; targets=[]
    for view in ('SAX','2CH','4CH'):
        locations=[(slice_id,frames) for v,slice_id,frames in base['cohort'] if v==view and len(frames)>=3]
        choices=representative_locations([(str(loc),float(frames[0].normalized_slice_position)) for loc,frames in locations])
        for slice_id,frames in locations:
            if str(slice_id) not in choices: continue
            frames=sorted(frames,key=lambda f:(f.timestamp_s,f.dynamic_frame_id))
            candidates=[i for i in range(1,len(frames)-1) if frames[i-1].timestamp_s<frames[i].timestamp_s<frames[i+1].timestamp_s and frames[i+1].timestamp_s-frames[i-1].timestamp_s>1e-9]
            if not candidates: continue
            chosen=sorted({candidates[round((len(candidates)-1)*q)] for q in (.25,.5,.75)})
            for index in chosen: targets.append({'view':view,'slice_id':str(slice_id),'frames':frames,'index':index})
    if not targets: raise ValueError('no representative frames with strictly ordered temporal neighbors')
    out=a.output_root/'residual_audit'; out.mkdir(parents=True,exist_ok=True); (out/'residuals').mkdir()
    cohort=[{'view':r['view'],'slice_id':r['slice_id'],'dynamic_frame_id':r['frames'][r['index']].dynamic_frame_id,'timestamp_s':r['frames'][r['index']].timestamp_s} for r in targets]
    write_csv(out/'cohort.csv',cohort); predictions=[{} for _ in targets]; result=[]
    device=next(model.parameters()).device
    for step in steps:
        load_step(step)
        for k,target_info in enumerate(targets):
            frames=target_info['frames']; i=target_info['index']; previous,obs,following=frames[i-1:i+2]
            with torch.no_grad():
                codes=[model.encode_observation(f,sample=False)['mu'] for f in (previous,obs,following)]
                interpolated=interpolate_neighbor_codes(codes[0].cpu().numpy(),codes[2].cpu().numpy(),previous.timestamp_s,obs.timestamp_s,following.timestamp_s)
                code_simple=torch.as_tensor(interpolated['simple'],device=device,dtype=codes[1].dtype); code_time=torch.as_tensor(interpolated['time'],device=device,dtype=codes[1].dtype)
                h,w=obs.image.shape[-2:]; linear=torch.arange(h*w,device=device); uv=torch.stack((linear%w,torch.div(linear,w,rounding_mode='floor')),-1)
                centers=SourceFirstDynamicModel._pixel_world(obs,uv.to(obs.image)); n=len(centers)
                resolution=obs.image.new_tensor([obs.pixel_spacing_mm[1],obs.pixel_spacing_mm[0],obs.slice_thickness_mm]).expand(n,3)
                with diagnostic_rng_context(7261+k):
                    samples=model.psf.sample(centers,resolution,row_direction=obs.row_direction.to(obs.image).expand(n,-1),column_direction=obs.column_direction.to(obs.image).expand(n,-1),normal=obs.normal.to(obs.image).expand(n,-1))
                def render(code):
                    batches=[]
                    for chunk in samples.split(a.chunk_size):
                        points=chunk if code is None else model.forward_deformation(chunk,code.expand(len(chunk),-1))['reference_points_mm']
                        batches.append(model.canonical(points,return_features=True)[0].mean(-1).cpu())
                    return torch.cat(batches).numpy()
                if 'static' not in predictions[k]: predictions[k]['static']=render(None)
                predictions[k][(step,'v1')]=render(codes[1]); predictions[k][(step,'v2_simple')]=render(code_simple); predictions[k][(step,'v2_time')]=render(code_time)
                actual=obs.image[0].reshape(-1).cpu().numpy(); predictions[k]['target']=actual; predictions[k]['shape']=(h,w)
                box=base['cardiac_box']; cardiac=None if box is None else np.all((centers.cpu().numpy()>=box.min_mm)&(centers.cpu().numpy()<=box.max_mm),axis=1)
                for label,pred in [('STATIC',predictions[k]['static'])]+[(str(step)+'_'+mode,predictions[k][(step,mode)]) for mode in ('v1','v2_simple','v2_time')]:
                    if label=='STATIC' and step!=steps[0]: continue
                    for region,mask in [('whole_slice',None),('cardiac_box',cardiac)]:
                        if region=='cardiac_box' and mask is None: continue
                        metrics=residual_metrics(actual,pred,mask)
                        if metrics is not None: result.append({'view':target_info['view'],'slice_id':target_info['slice_id'],'dynamic_frame_id':obs.dynamic_frame_id,'timestamp_s':obs.timestamp_s,'model':label,'region':region,**metrics})
    write_csv(out/'frame_metrics.csv',result)
    groups={}
    for row in result: groups.setdefault((row['model'],row['region'],row['view']),[]).append(row)
    summary=[]
    for (label,region,view),group in sorted(groups.items()):
        entry={'model':label,'region':region,'view':view,'frame_count':len(group)}
        for metric in ('mse','mae','ncc'):
            values=[r[metric] for r in group if r[metric] is not None]; entry['mean_'+metric]=float(np.mean(values)) if values else None; entry['median_'+metric]=float(np.median(values)) if values else None
        summary.append(entry)
    json_write(out/'summary_by_view.json',summary)
    grouped={}
    for row in result: grouped.setdefault((row['model'],row['region']),[]).append(row)
    by_checkpoint=[]
    for (label,region),group in sorted(grouped.items()):
        entry={'model':label,'region':region,'frame_count':len(group)}
        for metric in ('mse','mae','ncc'):
            values=[r[metric] for r in group if r[metric] is not None]; entry['mean_'+metric]=float(np.mean(values)) if values else None; entry['median_'+metric]=float(np.median(values)) if values else None
        by_checkpoint.append(entry)
    json_write(out/'summary_by_checkpoint.json',by_checkpoint)
    gap=[]
    for step in steps:
        for region in ('whole_slice','cardiac_box'):
            v1=next((r['mean_mse'] for r in by_checkpoint if r['model']==f'{step}_v1' and r['region']==region),None)
            v2=next((r['mean_mse'] for r in by_checkpoint if r['model']==f'{step}_v2_time' and r['region']==region),None)
            gap.append({'checkpoint_step':step,'region':region,'v1_mean_mse':v1,'v2_time_mean_mse':v2,'v2_minus_v1_mse':v2-v1 if v1 is not None and v2 is not None else None})
    write_csv(out/'v1_v2_gap.csv',gap); json_write(out/'v1_v2_gap.json',gap)
    for k,target_info in enumerate(targets):
        p=predictions[k]; h,w=p['shape']; obs=p['target'].reshape(h,w)
        for mode in ('v1','v2_time','v2_simple'):
            arrays=[p['static'].reshape(h,w)]+[p[(step,mode)].reshape(h,w) for step in steps]
            vmax=max(float(np.max(np.abs(obs-pred))) for pred in arrays) or 1.
            fig,axes=plt.subplots(2,len(arrays)+1,figsize=(3*(len(arrays)+1),6),squeeze=False)
            axes[0,0].imshow(obs,cmap='gray',vmin=0,vmax=1); axes[0,0].set_title('Observed'); axes[1,0].axis('off')
            for j,(label,pred) in enumerate(zip(['Static']+[str(step) for step in steps],arrays),1):
                axes[0,j].imshow(pred,cmap='gray',vmin=0,vmax=1); axes[0,j].set_title(label); axes[1,j].imshow(np.abs(obs-pred),cmap='magma',vmin=0,vmax=vmax)
            fig.suptitle(f"{target_info['view']} {target_info['slice_id']} frame {cohort[k]['dynamic_frame_id']} t={cohort[k]['timestamp_s']:.3f} {mode}; target-code self-consistency or target-withheld prediction")
            fig.savefig(out/'residuals'/f"{target_info['view']}_{target_info['slice_id']}_{cohort[k]['dynamic_frame_id']}_{mode}.png",dpi=180,bbox_inches='tight'); plt.close(fig)
            box=base['cardiac_box']
            if box is not None:
                frame=target_info['frames'][target_info['index']]
                linear=torch.arange(h*w,device=device); uv=torch.stack((linear%w,torch.div(linear,w,rounding_mode='floor')),-1)
                centers=SourceFirstDynamicModel._pixel_world(frame,uv.to(frame.image)).detach().cpu().numpy()
                mask=np.all((centers>=box.min_mm)&(centers<=box.max_mm),axis=1).reshape(h,w)
                if mask.any():
                    ys,xs=np.nonzero(mask); y0,y1=max(int(ys.min())-3,0),min(int(ys.max())+4,h); x0,x1=max(int(xs.min())-3,0),min(int(xs.max())+4,w)
                    fig,axes=plt.subplots(2,len(arrays)+1,figsize=(3*(len(arrays)+1),6),squeeze=False)
                    axes[0,0].imshow(obs[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1); axes[0,0].set_title('Observed'); axes[1,0].axis('off')
                    for j,(label,pred) in enumerate(zip(['Static']+[str(step) for step in steps],arrays),1):
                        axes[0,j].imshow(pred[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1); axes[0,j].set_title(label); axes[1,j].imshow(np.abs(obs-pred)[y0:y1,x0:x1],cmap='magma',vmin=0,vmax=vmax)
                    fig.suptitle(f"User-provided cardiac box (not segmentation): {target_info['view']} {target_info['slice_id']} frame {cohort[k]['dynamic_frame_id']} {mode}")
                    fig.savefig(out/'residuals'/f"{target_info['view']}_{target_info['slice_id']}_{cohort[k]['dynamic_frame_id']}_{mode}_cardiac_box.png",dpi=180,bbox_inches='tight'); plt.close(fig)
        np.savez_compressed(out/'residuals'/f"{target_info['view']}_{target_info['slice_id']}_{cohort[k]['dynamic_frame_id']}_data.npz",**{'target':p['target'],'static':p['static'],**{f'{step}_{mode}':p[(step,mode)] for step in steps for mode in ('v1','v2_simple','v2_time')}})
    fig,ax=plt.subplots(figsize=(8,4))
    for mode in ('v1','v2_time'):
        values=[]
        for step in steps:
            row=next((r for r in by_checkpoint if r['model']==f'{step}_{mode}' and r['region']=='whole_slice'),None); values.append(row['mean_mse'] if row else np.nan)
        ax.plot(steps,values,marker='o',label=mode)
    ax.set(xlabel='Checkpoint step',ylabel='Mean whole-slice MSE'); ax.legend(); fig.savefig(out/'comparison_metrics.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    return by_checkpoint
