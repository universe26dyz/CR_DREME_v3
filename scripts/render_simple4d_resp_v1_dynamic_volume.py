#!/usr/bin/env python3
"""Render existing SIMPLE-4D states and observation-driven temporal anatomy."""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from audit_simple4d_resp_v1_checkpoints import load_inputs, load_checkpoint, output_sha256
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import diagnostic_rng_context
from cardioresp4d.experiments.simple4d_resp_v1_identity_render import render_volume
from cardioresp4d.experiments.simple4d_resp_v1_identity_stages import read_latent
from cardioresp4d.experiments.simple4d_resp_v1_dynamic_visualization import load_selected_volumes, save_triplanar, selected_visualization, make_gif, write_metadata


def verify_stage(root, stage, step, checkpoint_hash, names):
    directory=root/stage
    manifest=json.loads((directory/'stage_manifest.json').read_text())
    if manifest['checkpoints'].get(str(step),{}).get('sha256') != checkpoint_hash:
        raise ValueError(f'{stage} checkpoint hash mismatch')
    for name in names:
        path=directory/name
        if manifest.get('output_sha256',{}).get(name) != output_sha256(path):
            raise ValueError(f'{stage} artifact hash mismatch: {name}')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root',type=Path,required=True); parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-step',type=int,required=True); parser.add_argument('--view',choices=['SAX'],default='SAX')
    parser.add_argument('--device',default='cpu'); parser.add_argument('--chunk-size',type=int,default=4096)
    parser.add_argument('--mode',choices=('selected','sequence','all'),default='all'); parser.add_argument('--gif-frame-ms',type=int,default=150)
    parser.add_argument('--save-4d-volume',action='store_true'); parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(); args.device=torch.device(args.device)
    if args.chunk_size<=0 or args.gif_frame_ms<=0 or args.checkpoint_step<=0: parser.error('step, chunk-size and gif-frame-ms must be positive')
    if args.save_4d_volume and args.mode=='selected': parser.error('--save-4d-volume requires sequence mode')
    step=args.checkpoint_step; view=args.view; checkpoint=args.run_dir/f'checkpoint_step_{step}.pt'
    checkpoint_hash=output_sha256(checkpoint)
    prefix=f'step_{step}/'; dvf_dir=args.audit_root/'04_dvf'/f'step_{step}'
    dvf_names=[prefix+'crop_manifest.json',prefix+'selected_states.csv']
    for state in ('low','mid','high'):
        dvf_names.extend((prefix+f'volumes/{view}_{state}_pc1.nii.gz',prefix+f'volumes/{view}_{state}_pc1.json'))
    dvf_manifest=verify_stage(args.audit_root,'04_dvf',step,checkpoint_hash,dvf_names)
    crop,records,images=load_selected_volumes(dvf_dir,view)
    for state in ('low','mid','high'):
        metadata=json.loads((dvf_dir/'volumes'/f'{view}_{state}_pc1.json').read_text())
        if metadata.get('checkpoint_sha256')!=checkpoint_hash or int(records[state]['checkpoint_step'])!=step:
            raise ValueError('selected state checkpoint identity mismatch')
    indices=tuple(int(n)//2 for n in crop['shape'])
    output=dvf_dir/'dynamic_visualization'; selected_dir=output/'selected_states'; sequence_dir=output/'sequence_frames'; gif_dir=output/'gif'
    if args.mode in ('selected','all') and (selected_dir.exists() or (gif_dir/f'{view}_selected_states_pingpong.gif').exists()) and not args.overwrite:
        raise FileExistsError('selected visualization exists; use --overwrite')
    if args.mode in ('sequence','all') and (sequence_dir.exists() or (gif_dir/f'{view}_temporal_sequence.gif').exists() or (output/'sequence_metadata.csv').exists() or (args.save_4d_volume and (output/'volumes'/f'{view}_temporal_sequence_4d.nii.gz').exists())) and not args.overwrite:
        raise FileExistsError('sequence visualization exists; use --overwrite')
    previous=json.loads((output/'visualization_manifest.json').read_text()) if (output/'visualization_manifest.json').exists() else {}
    if previous and (previous.get('checkpoint_sha256')!=checkpoint_hash or previous.get('crop')!=crop): raise ValueError('existing visualization provenance differs')
    sequence=[]; model=None; latent_manifest=None
    if args.mode in ('sequence','all'):
        latent_name=prefix+'frame_latent_scores.csv'
        latent_manifest=verify_stage(args.audit_root,'01_latent',step,checkpoint_hash,[latent_name])
        if latent_manifest['input_artifact_sha256'] != dvf_manifest['input_artifact_sha256']:
            raise ValueError('latent and DVF input provenance mismatch')
        bindings=json.loads((args.run_dir/'experiment_inputs.json').read_text())
        load_args=SimpleNamespace(device=args.device,**{name:Path(bindings[name]['path']) for name in ('manifest','qc_table','canonical_domain','static_manifest')})
        with diagnostic_rng_context():
            config,provenance,model,cohort=load_inputs(load_args)
        if provenance['hashes'] != dvf_manifest['input_artifact_sha256']:
            raise ValueError('current inputs do not match audit provenance')
        load_checkpoint(checkpoint,step,config,provenance,model)
        coords=model.forward_deformation.coordinates
        spacing=coords.reference_spacing_mm.detach().cpu().numpy()
        if not np.allclose(spacing,crop['spacing_mm'],atol=1e-5,rtol=0): raise ValueError('model and crop spacing mismatch')
        lower=coords.lower_world_mm.detach().cpu().numpy(); upper=lower+coords.extent_mm.detach().cpu().numpy()
        if np.any(np.array(crop['lower_lps_mm']) < lower-1e-4) or np.any(np.array(crop['upper_lps_mm']) > upper+1e-4): raise ValueError('crop lies outside model domain')
        latent_rows=read_latent(args.audit_root/'01_latent'/latent_name,cohort)
        anchor=records['low']['slice_id']
        sequence=sorted((r for r in latent_rows if r['view']==view and r['slice_id']==anchor),key=lambda r:(r['timestamp_s'],r['dynamic_frame_id']))
        if not sequence or any(b['timestamp_s'] < a['timestamp_s'] for a,b in zip(sequence,sequence[1:])):
            raise ValueError('anchor location has no valid timestamp sequence')
        by_id={str(row['dynamic_frame_id']):row for row in sequence}
        for record in records.values():
            row=by_id.get(record['dynamic_frame_id'])
            if row is None or abs(float(record['timestamp_s'])-row['timestamp_s'])>1e-6 or abs(float(record['pc1'])-row['pc1'])>1e-6:
                raise ValueError('selected anchor state is inconsistent with latent frame scores')
    output.mkdir(parents=True,exist_ok=True); gif_dir.mkdir(exist_ok=True)
    if args.mode in ('selected','all'):
        if selected_dir.exists(): shutil.rmtree(selected_dir)
        selected_visualization(output,images,records,indices,step,view,max(args.gif_frame_ms,300),crop['spacing_mm'])
    metadata_rows=[]
    if args.mode in ('sequence','all'):
        if sequence_dir.exists(): shutil.rmtree(sequence_dir)
        sequence_dir.mkdir(); paths=[]; start=sequence[0]['timestamp_s']; shape=tuple(crop['shape'])
        volume_4d=None; temp_path=None
        if args.save_4d_volume:
            with tempfile.NamedTemporaryFile(prefix='sequence_4d_',suffix='.dat',dir=output,delete=False) as temporary:
                temp_path=Path(temporary.name)
            volume_4d=np.memmap(temp_path,dtype=np.float32,mode='w+',shape=(*shape,len(sequence)))
        try:
            for index,row in enumerate(sequence):
                with torch.no_grad():
                    code=model.encode_observation(row['observation'],sample=False)['mu']
                    volume=render_volume(model,code,crop['lower_lps_mm'],crop['spacing_mm'],shape,args.chunk_size)
                relative=row['timestamp_s']-start; path=sequence_dir/f'frame_{index:04d}.png'
                save_triplanar(volume,indices,path,f"step {step} | {view} {row['slice_id']} | frame {index:04d} | t={relative:.3f}s | PC1={row['pc1']:.3f}",spacing=crop['spacing_mm'])
                paths.append(path)
                metadata_rows.append({'frame_index':index,'dynamic_frame_id':row['dynamic_frame_id'],'timestamp_s':row['timestamp_s'],'relative_time_s':relative,'pc1':row['pc1'],'pc2':float(row['pc2']),'pc3':float(row['pc3']),'slice_id':row['slice_id'],'checkpoint_step':step})
                if volume_4d is not None: volume_4d[...,index]=volume
                del volume
            write_metadata(output/'sequence_metadata.csv',metadata_rows)
            make_gif(paths,gif_dir/f'{view}_temporal_sequence.gif',args.gif_frame_ms)
            if volume_4d is not None:
                volume_4d.flush(); volumes_dir=output/'volumes'; volumes_dir.mkdir(exist_ok=True)
                nifti=nib.Nifti1Image(volume_4d,images['low'].affine); nib.save(nifti,str(volumes_dir/f'{view}_temporal_sequence_4d.nii.gz'))
                (volumes_dir/f'{view}_temporal_sequence_4d.json').write_text(json.dumps({'timestamps_s':[r['timestamp_s'] for r in metadata_rows],'time_axis':'last; irregular timestamps are authoritative','checkpoint_sha256':checkpoint_hash,'crop':crop},indent=2)+'\n')
        finally:
            if volume_4d is not None:
                del volume_4d
                if temp_path.exists(): temp_path.unlink()
    if output_sha256(checkpoint)!=checkpoint_hash: raise RuntimeError('checkpoint changed during visualization')
    previous.update({'checkpoint_step':step,'checkpoint_sha256':checkpoint_hash,'checkpoint_path':str(checkpoint.resolve()),'view':view,'anchor_slice_id':records['low']['slice_id'],'crop':crop,'slice_indices_lps_xyz':list(indices),'intensity_window':[0.,1.],'axis_convention':'external LPS xyz; NIfTI affine maps voxel indices to RAS','frame_dimensions_px':[1200,400],'gif_frame_ms':args.gif_frame_ms,'dvf_stage_input_hashes':dvf_manifest['input_artifact_sha256'],'dvf_artifact_sha256':{name:dvf_manifest['output_sha256'][name] for name in dvf_names},'selected_gif_frame_ms':max(args.gif_frame_ms,300)})
    if metadata_rows: previous['sequence']={'frame_count':len(metadata_rows),'timestamp_range_s':[metadata_rows[0]['timestamp_s'],metadata_rows[-1]['timestamp_s']],'pc1_range':[min(r['pc1'] for r in metadata_rows),max(r['pc1'] for r in metadata_rows)],'latent_score_sha256':latent_manifest['output_sha256'][prefix+'frame_latent_scores.csv'],'posterior_policy':'same checkpoint encode_observation(sample=False); full 32D mean','saved_4d_volume':args.save_4d_volume}
    if args.mode in ('selected','all'): previous['selected']={'source':'existing DVF stage NIfTI volumes','states':['low','mid','high'],'gif_sequence':['low','mid','high','mid','low']}
    (output/'visualization_manifest.json').write_text(json.dumps(previous,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'mode':args.mode,'frame_count':len(metadata_rows),'checkpoint_sha256':checkpoint_hash}))


if __name__=='__main__': main()
