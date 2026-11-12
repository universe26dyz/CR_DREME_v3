"""Static-support crop and anatomy-aware DVF visualization in external LPS mm."""
from pathlib import Path
import itertools
import tempfile
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import map_coordinates
from .simple4d_resp_v1_identity_render import dense_field, volume_grid, save_lps_nifti

# Nonzero static intensity support: > 5% of the 99th percentile of positive voxels.
SUPPORT_FRACTION = .05
SUPPORT_PERCENTILE = 99.
CROP_PADDING_MM = 10.


def derive_anatomy_crop(static_volume, lower_lps, upper_lps, spacing_mm, *, source_sha256, padding_mm=CROP_PADDING_MM):
    img=nib.load(str(static_volume)); volume=np.asarray(img.dataobj,dtype=np.float32)
    if volume.ndim!=3 or not np.isfinite(volume).all() or not np.isfinite(img.affine).all(): raise ValueError('static NIfTI must be a finite 3D image with finite affine')
    positive=volume[volume>0]
    if not len(positive): raise ValueError('cannot derive anatomy crop: static volume has no positive support')
    threshold=float(np.percentile(positive,SUPPORT_PERCENTILE)*SUPPORT_FRACTION)
    support=(volume>threshold).astype(np.uint8)
    if not support.any(): raise ValueError('cannot derive anatomy crop: empty support')
    indices=[np.flatnonzero(support.any(axis=tuple(j for j in range(3) if j!=i))) for i in range(3)]
    corners=np.array(list(itertools.product(*[(axis[0],axis[-1]) for axis in indices])),dtype=float)
    ras=nib.affines.apply_affine(img.affine,corners); lps=ras*np.array([-1.,-1.,1.])
    domain_lo=np.asarray(lower_lps,dtype=float); domain_hi=np.asarray(upper_lps,dtype=float); spacing=np.asarray(spacing_mm,dtype=float)
    if domain_lo.shape!=(3,) or domain_hi.shape!=(3,) or spacing.shape!=(3,) or not np.isfinite(np.r_[domain_lo,domain_hi,spacing,padding_mm]).all() or np.any(domain_hi<=domain_lo) or np.any(spacing<=0) or padding_mm<0: raise ValueError('invalid domain or spacing')
    raw_lo=np.maximum(domain_lo,lps.min(0)-padding_mm); raw_hi=np.minimum(domain_hi,lps.max(0)+padding_mm)
    if np.any(raw_hi<=raw_lo): raise ValueError('static anatomy support does not intersect model domain')
    # Align the crop to the model-domain grid so every checkpoint uses identical voxels.
    origin=domain_lo+np.ceil((raw_lo-domain_lo)/spacing-1e-7)*spacing
    last=domain_lo+np.floor((raw_hi-domain_lo)/spacing+1e-7)*spacing
    if np.any(last<=origin): raise ValueError('anatomy crop contains fewer than two grid points per axis')
    _,_,shape=volume_grid(origin,last,spacing)
    manifest={'source':'admitted_static_reference_nifti','source_path':str(Path(static_volume).resolve()),'source_sha256':source_sha256,'support_rule':'static voxel intensity > 0.05 * positive-voxel 99th percentile','support_percentile':SUPPORT_PERCENTILE,'support_fraction':SUPPORT_FRACTION,'support_threshold':threshold,'padding_mm':padding_mm,'lower_lps_mm':origin.tolist(),'upper_lps_mm':last.tolist(),'spacing_mm':spacing.tolist(),'shape':list(shape),'crop_covers_full_model_domain':bool(np.allclose(origin,domain_lo) and np.allclose(last,domain_hi)),'crop_interpretation':'static intensity support; not a heart-specific segmentation','axis_convention':'array xyz in external patient LPS mm; NIfTI affine uses RAS'}
    return manifest,support,np.linalg.inv(img.affine)


def crop_points(origin,spacing,shape,start,stop,device):
    index=torch.arange(start,stop,device=device)
    xyz=torch.stack((index//(shape[1]*shape[2]),(index//shape[2])%shape[1],index%shape[2]),-1)
    return torch.as_tensor(origin,dtype=torch.float32,device=device)+xyz*torch.as_tensor(spacing,dtype=torch.float32,device=device)


def support_at_world(points_lps, support, ras_to_voxel):
    points=np.asarray(points_lps,dtype=float); ras=points*np.array([-1.,-1.,1.]); vox=nib.affines.apply_affine(ras_to_voxel,ras)
    return map_coordinates(support.astype(np.uint8,copy=False),vox.T,order=0,mode='constant',cval=0,prefilter=False).astype(bool)


def supported_stats(field, det, mask):
    mag=np.linalg.norm(field,axis=-1); mask=np.asarray(mask,dtype=bool)
    if not mask.any(): raise ValueError('static support is empty on visualization grid')
    def describe(which):
        v=mag[which]; f=field[which]; j=det[which]
        return {'voxel_count':int(len(v)),'mean_magnitude_mm':float(v.mean()),'rms_magnitude_mm':float(np.sqrt(np.mean(v**2))),'p95_magnitude_mm':float(np.quantile(v,.95)),'max_magnitude_mm':float(v.max()),'mean_abs_dx_mm':float(np.abs(f[:,0]).mean()),'mean_abs_dy_mm':float(np.abs(f[:,1]).mean()),'mean_abs_dz_mm':float(np.abs(f[:,2]).mean()),'jacobian_p01':float(np.quantile(j,.01)),'folding_fraction':float(np.mean(j<=0))}
    return {'all_crop':describe(np.ones(len(mask),dtype=bool)), 'anatomical_support':describe(mask)}


def planes(array,shape):
    a=array.reshape(*shape,*array.shape[1:]); mid=np.asarray(shape)//2
    return (np.array(a[:,:,mid[2]]),np.array(a[:,mid[1],:]),np.array(a[mid[0],:,:]))


def render_anatomy_state(model, code, manifest, support, ras_to_voxel, chunk_size, workdir):
    """Evaluate crop in bounded GPU chunks; temporary arrays are disk-backed."""
    lower=np.asarray(manifest['lower_lps_mm']); spacing=np.asarray(manifest['spacing_mm']); shape=tuple(manifest['shape']); total=int(np.prod(shape)); device=code.device
    field=np.memmap(Path(workdir)/'field.dat',mode='w+',dtype=np.float32,shape=(total,3))
    intensity=np.memmap(Path(workdir)/'image.dat',mode='w+',dtype=np.float32,shape=(total,))
    determinant=np.memmap(Path(workdir)/'jac.dat',mode='w+',dtype=np.float32,shape=(total,))
    mask=np.memmap(Path(workdir)/'mask.dat',mode='w+',dtype=np.bool_,shape=(total,))
    for start in range(0,total,chunk_size):
        stop=min(start+chunk_size,total); world=crop_points(lower,spacing,shape,start,stop,device)
        dvf,det,_=dense_field(model,code,world,chunk_size); field[start:stop]=dvf; determinant[start:stop]=det
        mask[start:stop]=support_at_world(world.detach().cpu().numpy(),support,ras_to_voxel)
        with torch.no_grad():
            ref=model.forward_deformation(world[None],code)['reference_points_mm']
            intensity[start:stop]=model.canonical(ref,return_features=True)[0].reshape(-1).detach().cpu().numpy()
    stats=supported_stats(field,determinant,mask)
    return {'field_planes':planes(field,shape),'image_planes':planes(intensity,shape),'mask_planes':planes(mask,shape),'statistics':stats,'intensity':intensity}


def outline_support(ax, mask, extent):
    """Cyan contour marks static support; heatmaps cover the entire crop."""
    mask=np.asarray(mask,dtype=bool)
    if mask.any() and not mask.all():
        ax.contour(mask.T.astype(float),levels=[.5],colors=['cyan'],linewidths=.6,origin='lower',extent=extent)


def anatomy_overlay_stage(model,rows,dest,step,checkpoint_sha,chunk_size,crop,support,ras_to_voxel):
    from .simple4d_resp_v1_identity_render import choose_anchor_states
    from .simple4d_resp_v1_identity_stages import latent_locations,write_csv,json_write
    import matplotlib.pyplot as plt
    dest=Path(dest); locations=latent_locations(rows); json_write(dest/'crop_manifest.json',crop)
    records=[]; saved={}
    lo=np.asarray(crop['lower_lps_mm']); hi=np.asarray(crop['upper_lps_mm'])
    extents=[(lo[0],hi[0],lo[1],hi[1]),(lo[0],hi[0],lo[2],hi[2]),(lo[1],hi[1],lo[2],hi[2])]
    for view in ('SAX','2CH','4CH'):
        states=choose_anchor_states(locations,view)
        if states is None: continue
        figures={}
        for state,row in states.items():
            with torch.no_grad(): code=model.encode_observation(row['observation'],sample=False)['mu'].detach()
            with tempfile.TemporaryDirectory(prefix='anatomy-',dir=dest) as workdir:
                result=render_anatomy_state(model,code,crop,support,ras_to_voxel,chunk_size,workdir)
                figures[state]={k:result[k] for k in ('field_planes','image_planes','mask_planes','statistics')}
                for region,stats in result['statistics'].items():
                    records.append({'checkpoint_step':step,'view':view,'slice_id':row['slice_id'],'state':state,'dynamic_frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'pc1':row['pc1'],'region':region,**stats})
                if view=='SAX':
                    vdir=dest/'volumes'; vdir.mkdir(exist_ok=True)
                    volume=np.asarray(result['intensity']).reshape(crop['shape'])
                    save_lps_nifti(vdir/f'SAX_{state}_pc1.nii.gz',volume,crop['lower_lps_mm'],crop['spacing_mm'])
                    json_write(vdir/f'SAX_{state}_pc1.json',{'checkpoint':step,'checkpoint_sha256':checkpoint_sha,'view':view,'slice_id':row['slice_id'],'dynamic_frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'pc1':row['pc1'],'crop':crop,'rendering_convention':'external observation LPS -> forward deformation -> frozen canonical'})
                del result
        if not figures: continue
        saved[view]=figures
        arrays={}
        for state,result in figures.items():
            for kind in ('field_planes','image_planes','mask_planes'):
                for plane,data in zip(('axial','coronal','sagittal'),result[kind]): arrays[f'{state}_{kind}_{plane}']=data
        np.savez_compressed(dest/f'{view}_anatomy_overlay_data.npz',**arrays)
        vmax=max(float(np.linalg.norm(p,axis=-1).max()) for r in figures.values() for p in r['field_planes']) or 1.
        fig,axs=plt.subplots(3,3,figsize=(13,11))
        for i,state in enumerate(('low','mid','high')):
            r=figures[state]; row=states[state]; stats=r['statistics']['anatomical_support']
            for j,(image,field) in enumerate(zip(r['image_planes'],r['field_planes'])):
                ax=axs[i,j]; ax.imshow(image.T,origin='lower',cmap='gray',vmin=0,vmax=1,extent=extents[j]); heat=ax.imshow(np.linalg.norm(field,axis=-1).T,origin='lower',cmap='magma',vmin=0,vmax=vmax,alpha=.45,extent=extents[j])
                outline_support(ax,r['mask_planes'][j],extents[j])
                ax.set_title(f"{step} {view}/{row['slice_id']} {state} t={row['timestamp_s']:.2f}s PC1={row['pc1']:.2f}\nsupport RMS={stats['rms_magnitude_mm']:.2f} P95={stats['p95_magnitude_mm']:.2f} max={stats['max_magnitude_mm']:.2f} mm",fontsize=8)
                ax.set(xlabel=('LPS x (mm)','LPS x (mm)','LPS y (mm)')[j],ylabel=('LPS y (mm)','LPS z (mm)','LPS z (mm)')[j])
        fig.suptitle('Heatmap: full crop; cyan contour: static anatomical support'); fig.colorbar(heat,ax=axs.ravel().tolist(),label='|DVF| mm'); fig.savefig(dest/f'{view}_dvf_magnitude_overlay.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        for state in ('low','mid','high'):
            r=figures[state]; component_max=max(float(np.max(np.abs(f))) for f in r['field_planes']) or 1.
            fig,axs=plt.subplots(3,3,figsize=(12,10))
            for j,(image,field) in enumerate(zip(r['image_planes'],r['field_planes'])):
                for i in range(3):
                    ax=axs[i,j]; ax.imshow(image.T,origin='lower',cmap='gray',vmin=0,vmax=1,extent=extents[j]); heat=ax.imshow(field[...,i].T,origin='lower',cmap='coolwarm',vmin=-component_max,vmax=component_max,alpha=.45,extent=extents[j]); outline_support(ax,r['mask_planes'][j],extents[j]); ax.set(xlabel=('LPS x (mm)','LPS x (mm)','LPS y (mm)')[j],ylabel=('LPS y (mm)','LPS z (mm)','LPS z (mm)')[j]); ax.set_title(f'{view} {state} LPS {"xyz"[i]} displacement (mm) — {("axial","coronal","sagittal")[j]}')
            fig.suptitle('Heatmap: full crop; cyan contour: static anatomical support'); fig.colorbar(heat,ax=axs.ravel().tolist(),label='Signed mm'); fig.savefig(dest/f'{view}_dvf_components_{state}.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(12,4)); diffs=[h-l for h,l in zip(figures['high']['field_planes'],figures['low']['field_planes'])]; maxdiff=max(float(np.linalg.norm(d,axis=-1).max()) for d in diffs) or 1.
        for j,(ax,image,diff) in enumerate(zip(axs,figures['mid']['image_planes'],diffs)):
            ax.imshow(image.T,origin='lower',cmap='gray',vmin=0,vmax=1,extent=extents[j]); heat=ax.imshow(np.linalg.norm(diff,axis=-1).T,origin='lower',cmap='magma',vmin=0,vmax=maxdiff,alpha=.45,extent=extents[j]); outline_support(ax,figures['mid']['mask_planes'][j],extents[j]); ax.set(xlabel=('LPS x (mm)','LPS x (mm)','LPS y (mm)')[j],ylabel=('LPS y (mm)','LPS z (mm)','LPS z (mm)')[j]); ax.set_title(f'{view} |DVF high−low| mm — {("axial","coronal","sagittal")[j]}')
        fig.suptitle('Heatmap: full crop; cyan contour: static anatomical support'); fig.colorbar(heat,ax=axs.ravel().tolist(),label='Difference magnitude mm'); fig.savefig(dest/f'{view}_dvf_high_minus_low_overlay.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    if not records: raise ValueError('no eligible states for anatomy visualization')
    if 'SAX' in saved:
        fig,axs=plt.subplots(3,3,figsize=(10,10))
        for i,state in enumerate(('low','mid','high')):
            for j,image in enumerate(saved['SAX'][state]['image_planes']):
                axs[i,j].imshow(image.T,origin='lower',cmap='gray',vmin=0,vmax=1); axs[i,j].set_title(f'{state} {("axial","coronal","sagittal")[j]} (LPS axes)')
        fig.savefig(dest/'01_dynamic_volume_low_mid_high.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        diffs=[h-l for h,l in zip(saved['SAX']['high']['image_planes'],saved['SAX']['low']['image_planes'])]
        limit=max(float(np.max(np.abs(d))) for d in diffs) or 1.
        fig,axs=plt.subplots(1,3,figsize=(10,3))
        for ax,d in zip(axs,diffs): ax.imshow(d.T,origin='lower',cmap='coolwarm',vmin=-limit,vmax=limit)
        fig.savefig(dest/'02_dynamic_volume_high_minus_low.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    write_csv(dest/'dvf_anatomy_summary.csv',records)


def compare_anatomy_sax(dest,steps):
    import json
    import matplotlib.pyplot as plt
    data={}; crops=[]
    for step in steps:
        directory=Path(dest)/f'step_{step}'; path=directory/'SAX_anatomy_overlay_data.npz'
        if path.exists():
            crops.append(json.loads((directory/'crop_manifest.json').read_text()))
            with np.load(path) as f: data[step]={key:np.array(f[key]) for key in f.files}
    if not data: return
    geometry=('lower_lps_mm','upper_lps_mm','spacing_mm','shape','source_sha256','support_threshold')
    if any(any(c[key]!=crops[0][key] for key in geometry) for c in crops[1:]): raise ValueError('checkpoint visualization crops differ')
    lo=np.asarray(crops[0]['lower_lps_mm']); hi=np.asarray(crops[0]['upper_lps_mm']); extent=(lo[0],hi[0],lo[1],hi[1])
    vmax=max(float(np.linalg.norm(d[f'{state}_field_planes_axial'],axis=-1).max()) for d in data.values() for state in ('low','mid','high')) or 1.
    fig,axes=plt.subplots(len(data),3,figsize=(10,len(data)*3),squeeze=False)
    for i,(step,d) in enumerate(sorted(data.items())):
        for j,state in enumerate(('low','mid','high')):
            ax=axes[i,j]; ax.imshow(d[f'{state}_image_planes_axial'].T,origin='lower',cmap='gray',vmin=0,vmax=1,extent=extent); im=ax.imshow(np.linalg.norm(d[f'{state}_field_planes_axial'],axis=-1).T,origin='lower',cmap='magma',vmin=0,vmax=vmax,alpha=.45,extent=extent); outline_support(ax,d[f'{state}_mask_planes_axial'],extent); ax.set_title(f'{step} SAX {state} |DVF| mm'); ax.set(xlabel='LPS x (mm)',ylabel='LPS y (mm)')
    fig.suptitle('Heatmap: full crop; cyan contour: static anatomical support'); fig.colorbar(im,ax=axes.ravel().tolist(),label='|DVF| mm'); fig.savefig(Path(dest)/'comparison_SAX_dvf_magnitude_overlay.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    vmax=max(float(np.linalg.norm(d['high_field_planes_axial']-d['low_field_planes_axial'],axis=-1).max()) for d in data.values()) or 1.
    fig,axes=plt.subplots(1,len(data),figsize=(4*len(data),4),squeeze=False)
    for ax,(step,d) in zip(axes[0],sorted(data.items())):
        ax.imshow(d['mid_image_planes_axial'].T,origin='lower',cmap='gray',vmin=0,vmax=1,extent=extent); im=ax.imshow(np.linalg.norm(d['high_field_planes_axial']-d['low_field_planes_axial'],axis=-1).T,origin='lower',cmap='magma',vmin=0,vmax=vmax,alpha=.45,extent=extent); outline_support(ax,d['mid_mask_planes_axial'],extent); ax.set_title(f'{step} |DVF high−low| mm'); ax.set(xlabel='LPS x (mm)',ylabel='LPS y (mm)')
    fig.suptitle('Heatmap: full crop; cyan contour: static anatomical support'); fig.colorbar(im,ax=axes.ravel().tolist(),label='Difference magnitude mm'); fig.savefig(Path(dest)/'comparison_SAX_high_minus_low.png',dpi=180,bbox_inches='tight'); plt.close(fig)
