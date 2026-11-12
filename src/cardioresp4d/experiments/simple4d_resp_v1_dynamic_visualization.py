"""Read-only fixed-geometry visualization of existing SIMPLE-4D DVF outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image


def load_selected_volumes(step_dir, view='SAX'):
    """Load existing state volumes, checking every geometry and sidecar contract."""
    step_dir = Path(step_dir)
    crop = json.loads((step_dir/'crop_manifest.json').read_text())
    with (step_dir/'selected_states.csv').open(newline='') as file:
        selected = [row for row in csv.DictReader(file) if row['view'] == view]
    if len(selected) != 3 or {row['state'] for row in selected} != {'low','mid','high'} or len({row['slice_id'] for row in selected}) != 1:
        raise ValueError('selected states must have one shared anchor and low/mid/high')
    records = {row['state']: row for row in selected}
    shape = tuple(int(n) for n in crop['shape'])
    spacing = np.asarray(crop['spacing_mm'],dtype=float)
    lower = np.asarray(crop['lower_lps_mm'],dtype=float)
    upper = np.asarray(crop['upper_lps_mm'],dtype=float)
    if len(shape) != 3 or min(shape) < 2 or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0) or lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(np.r_[lower,upper]).all() or not np.allclose(lower+(np.array(shape)-1)*spacing,upper,atol=1e-4,rtol=0):
        raise ValueError('invalid crop grid geometry')
    lps = np.eye(4); lps[:3,:3] = np.diag(spacing); lps[:3,3] = lower
    expected_affine = np.diag([-1.,-1.,1.,1.]) @ lps
    images = {}
    for state in ('low','mid','high'):
        base = step_dir/'volumes'/f'{view}_{state}_pc1'
        image = nib.load(str(base)+'.nii.gz')
        metadata = json.loads(Path(str(base)+'.json').read_text())
        if image.shape != shape or not np.allclose(image.affine,expected_affine,atol=1e-4,rtol=0) or not np.allclose(image.header.get_zooms()[:3],spacing,atol=1e-5,rtol=0):
            raise ValueError(f'{state} volume geometry does not match crop')
        row = records[state]
        if metadata.get('crop') != crop or metadata.get('view') != view or str(metadata.get('slice_id')) != row['slice_id'] or str(metadata.get('dynamic_frame_id')) != row['dynamic_frame_id'] or int(metadata.get('checkpoint',-1)) != int(row['checkpoint_step']):
            raise ValueError(f'{state} volume sidecar does not match selected state/crop')
        for name in ('timestamp_s','pc1'):
            sidecar=float(metadata[name]); selected_value=float(row[name])
            if not np.isfinite([sidecar,selected_value]).all() or abs(sidecar-selected_value)>1e-6:
                raise ValueError(f'{state} volume sidecar {name} mismatch')
        images[state] = image
    return crop, records, images


def fixed_planes(volume, indices):
    arr = np.asanyarray(volume)
    if arr.ndim != 3 or not np.isfinite(arr).all():
        raise ValueError('volume must be finite and three-dimensional')
    x,y,z = indices
    return [arr[:,:,z].T, arr[:,y,:].T, arr[x,:,:].T]


def save_triplanar(volume, indices, output, title, *, window=(0.,1.), spacing=(1.,1.,1.)):
    planes = fixed_planes(volume,indices)
    fig,axes = plt.subplots(1,3,figsize=(12,4),dpi=100)
    spacings=((spacing[0],spacing[1]),(spacing[0],spacing[2]),(spacing[1],spacing[2]))
    for ax,plane,label,sp in zip(axes,planes,('Axial (LPS x/y)','Coronal (LPS x/z)','Sagittal (LPS y/z)'),spacings):
        extent=(0,plane.shape[1]*sp[0],0,plane.shape[0]*sp[1])
        ax.imshow(plane,origin='lower',cmap='gray',vmin=window[0],vmax=window[1],aspect='equal',interpolation='nearest',extent=extent)
        ax.set_title(label); ax.set_axis_off()
    fig.suptitle(title); fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.85,wspace=.04)
    fig.savefig(output,dpi=100); plt.close(fig)


def make_gif(paths, target, duration_ms):
    if duration_ms <= 0 or not paths:
        raise ValueError('GIF requires frames and a positive duration')
    images=[]
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    if len({image.size for image in images}) != 1:
        raise ValueError('all GIF frames must have identical dimensions')
    images[0].save(target,save_all=True,append_images=images[1:],duration=duration_ms,loop=0,optimize=False,disposal=2)


def selected_visualization(output, images, records, indices, step, view, duration_ms, spacing):
    selected_dir=output/'selected_states'; gif_dir=output/'gif'
    selected_dir.mkdir(parents=True,exist_ok=True); gif_dir.mkdir(parents=True,exist_ok=True)
    paths={}
    for state in ('low','mid','high'):
        row=records[state]; path=selected_dir/f'{view}_{state}.png'
        save_triplanar(np.asanyarray(images[state].dataobj),indices,path,f"step {step} | {view} {row['slice_id']} | {state} | t={float(row['timestamp_s']):.3f}s | PC1={float(row['pc1']):.3f}",spacing=spacing)
        paths[state]=path
    fig,axes=plt.subplots(3,3,figsize=(12,10),dpi=100)
    for i,state in enumerate(('low','mid','high')):
        for ax,plane,label,sp in zip(axes[i],fixed_planes(np.asanyarray(images[state].dataobj),indices),('Axial','Coronal','Sagittal'),((spacing[0],spacing[1]),(spacing[0],spacing[2]),(spacing[1],spacing[2]))):
            ax.imshow(plane,origin='lower',cmap='gray',vmin=0,vmax=1,aspect='equal',interpolation='nearest',extent=(0,plane.shape[1]*sp[0],0,plane.shape[0]*sp[1])); ax.set_title(f'{state} | {label}'); ax.set_axis_off()
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.97,wspace=.04,hspace=.16)
    fig.savefig(selected_dir/f'{view}_low_mid_high_comparison.png',dpi=100); plt.close(fig)
    make_gif([paths[x] for x in ('low','mid','high','mid','low')],gif_dir/f'{view}_selected_states_pingpong.gif',duration_ms)


def write_metadata(path, rows):
    fields=['frame_index','dynamic_frame_id','timestamp_s','relative_time_s','pc1','pc2','pc3','slice_id','checkpoint_step']
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
