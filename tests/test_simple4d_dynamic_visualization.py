import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
from PIL import Image, ImageSequence

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dynamic_visualization_cli',ROOT/'scripts/render_simple4d_resp_v1_dynamic_volume.py')
cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def setup_fixture(tmp_path):
    root=tmp_path/'audit'; run=tmp_path/'run'; run.mkdir(); checkpoint=run/'checkpoint_step_2000.pt'; checkpoint.write_bytes(b'checkpoint immutable')
    digest=cli.output_sha256(checkpoint); dvf=root/'04_dvf'; folder=dvf/'step_2000'; vols=folder/'volumes'; vols.mkdir(parents=True)
    crop={'shape':[12,12,12],'spacing_mm':[1.,1.,1.],'lower_lps_mm':[-6.,-6.,-6.],'upper_lps_mm':[5.,5.,5.]}
    (folder/'crop_manifest.json').write_text(json.dumps(crop)); affine=np.diag([-1.,-1.,1.,1.]); affine[:3,3]=[6.,6.,-6.]
    times=[3.,1.,2.,4.,5.]; ids=[12,10,11,13,14]
    values=[.2,-.8,-.2,.6,.9]
    rows=[]
    for i in range(5):
        rows.append({'checkpoint_step':2000,'view':'SAX','slice_id':'s01','dynamic_frame_id':ids[i],'timestamp_s':times[i],'normalized_slice_position':0.,'pc1':values[i],'pc2':.1,'pc3':.2,'pc1_location_z':values[i]})
    chosen=[('low',1),('mid',0),('high',4)]
    states=[]
    for state,index in chosen:
        r=rows[index]; states.append({**{k:r[k] for k in ('checkpoint_step','view','slice_id','dynamic_frame_id','timestamp_s','pc1')},'state':state,'pc1_percentile_target':0})
        grid=np.zeros((12,12,12),np.float32); grid[2+index:5+index,4:8,4:8]=.9
        nib.save(nib.Nifti1Image(grid,affine),vols/f'SAX_{state}_pc1.nii.gz')
        (vols/f'SAX_{state}_pc1.json').write_text(json.dumps({'crop':crop,'view':'SAX','slice_id':'s01','dynamic_frame_id':ids[index],'checkpoint':2000,'checkpoint_sha256':digest,'timestamp_s':r['timestamp_s'],'pc1':r['pc1']}))
    write_csv(folder/'selected_states.csv',states)
    files={str(p.relative_to(dvf)):cli.output_sha256(p) for p in folder.rglob('*') if p.is_file()}
    manifest={'checkpoints':{'2000':{'sha256':digest}},'input_artifact_sha256':{'fixture':'hash'},'output_sha256':files}
    (dvf/'stage_manifest.json').write_text(json.dumps(manifest))
    latent=root/'01_latent'; latent_step=latent/'step_2000'; latent_step.mkdir(parents=True)
    write_csv(latent_step/'frame_latent_scores.csv',rows)
    latent_manifest={'checkpoints':{'2000':{'sha256':digest}},'input_artifact_sha256':{'fixture':'hash'},'output_sha256':{'step_2000/frame_latent_scores.csv':cli.output_sha256(latent_step/'frame_latent_scores.csv')}}
    (latent/'stage_manifest.json').write_text(json.dumps(latent_manifest))
    (run/'experiment_inputs.json').write_text(json.dumps({name:{'path':str(tmp_path/name)} for name in ('manifest','qc_table','canonical_domain','static_manifest')}))
    return root,run,checkpoint,rows,crop


class FakeModel:
    def __init__(self):
        self.forward_deformation=SimpleNamespace(coordinates=SimpleNamespace(reference_spacing_mm=torch.ones(3),lower_world_mm=torch.tensor([-6.,-6.,-6.]),extent_mm=torch.tensor([11.,11.,11.])))
    def encode_observation(self,obs,*,sample):
        assert not sample
        return {'mu':torch.tensor([[obs.code]])}


def test_all_outputs_and_read_only(tmp_path,monkeypatch):
    root,run,checkpoint,rows,crop=setup_fixture(tmp_path); before=checkpoint.read_bytes()
    frames=tuple(SimpleNamespace(dynamic_frame_id=r['dynamic_frame_id'],timestamp_s=r['timestamp_s'],code=r['timestamp_s']) for r in rows)
    cohort=(('SAX','s01',frames),)
    monkeypatch.setattr(cli,'load_inputs',lambda args: ({},{'hashes':{'fixture':'hash'}},FakeModel(),cohort))
    monkeypatch.setattr(cli,'load_checkpoint',lambda *args: None)
    def render(model,code,lower,spacing,shape,chunk):
        array=np.zeros(shape,np.float32); index=int(code.item()); array[index:index+2,4:8,4:8]=.9
        return array
    monkeypatch.setattr(cli,'render_volume',render)
    monkeypatch.setattr(sys,'argv',['render','--audit-root',str(root),'--run-dir',str(run),'--checkpoint-step','2000','--mode','all','--chunk-size','8'])
    cli.main(); output=root/'04_dvf/step_2000/dynamic_visualization'
    for name in ('SAX_low.png','SAX_mid.png','SAX_high.png','SAX_low_mid_high_comparison.png'):
        assert (output/'selected_states'/name).stat().st_size>0
    frames=sorted((output/'sequence_frames').glob('*.png')); assert len(frames)==5
    rendered=[np.array(Image.open(p).convert('RGB')) for p in frames]
    assert len({a.shape for a in rendered})==1 and not np.array_equal(rendered[0][120:,:],rendered[-1][120:,:])
    for name in ('SAX_selected_states_pingpong.gif','SAX_temporal_sequence.gif'):
        with Image.open(output/'gif'/name) as gif:
            assert gif.n_frames>1 and gif.info['loop']==0
    with (output/'sequence_metadata.csv').open() as f: metadata=list(csv.DictReader(f))
    assert [float(r['timestamp_s']) for r in metadata]==[1.,2.,3.,4.,5.]
    assert [float(r['relative_time_s']) for r in metadata]==[0.,1.,2.,3.,4.]
    manifest=json.loads((output/'visualization_manifest.json').read_text())
    assert manifest['slice_indices_lps_xyz']==[6,6,6] and manifest['intensity_window']==[0.,1.] and manifest['crop']==crop
    assert checkpoint.read_bytes()==before
    assert not (output/'volumes').exists()


def test_geometry_mismatch_rejected(tmp_path):
    root,run,checkpoint,rows,crop=setup_fixture(tmp_path)
    folder=root/'04_dvf/step_2000'; path=folder/'volumes/SAX_high_pc1.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((12,12,12),np.float32),np.eye(4)),path)
    with pytest.raises(ValueError,match='geometry'): cli.load_selected_volumes(folder)


def test_selected_mode_needs_no_model_and_checks_hash(tmp_path,monkeypatch):
    root,run,checkpoint,_,_=setup_fixture(tmp_path)
    monkeypatch.setattr(cli,'load_inputs',lambda args: pytest.fail('selected mode must not build a model'))
    monkeypatch.setattr(sys,'argv',['render','--audit-root',str(root),'--run-dir',str(run),'--checkpoint-step','2000','--mode','selected'])
    cli.main()
    output=root/'04_dvf/step_2000/dynamic_visualization'
    assert (output/'gif/SAX_selected_states_pingpong.gif').exists()
    assert not (output/'sequence_frames').exists()
    (root/'04_dvf/step_2000/selected_states.csv').write_text('tampered')
    with pytest.raises(ValueError,match='artifact hash mismatch'): cli.main()


def test_optional_4d_output(tmp_path,monkeypatch):
    root,run,checkpoint,rows,crop=setup_fixture(tmp_path)
    frames=tuple(SimpleNamespace(dynamic_frame_id=r['dynamic_frame_id'],timestamp_s=r['timestamp_s'],code=r['timestamp_s']) for r in rows)
    monkeypatch.setattr(cli,'load_inputs',lambda args: ({},{'hashes':{'fixture':'hash'}},FakeModel(),(('SAX','s01',frames),)))
    monkeypatch.setattr(cli,'load_checkpoint',lambda *args: None)
    monkeypatch.setattr(cli,'render_volume',lambda model,code,lower,spacing,shape,chunk: np.full(shape,code.item(),dtype=np.float32))
    monkeypatch.setattr(sys,'argv',['render','--audit-root',str(root),'--run-dir',str(run),'--checkpoint-step','2000','--mode','sequence','--save-4d-volume'])
    cli.main(); output=root/'04_dvf/step_2000/dynamic_visualization'
    path=output/'volumes/SAX_temporal_sequence_4d.nii.gz'
    nii=nib.load(path)
    assert nii.shape==(12,12,12,5)
    assert np.allclose(np.asarray(nii.dataobj)[0,0,0,:],[1.,2.,3.,4.,5.])
    reference=nib.load(root/'04_dvf/step_2000/volumes/SAX_low_pc1.nii.gz')
    assert np.allclose(nii.affine,reference.affine)
    assert json.loads((output/'volumes/SAX_temporal_sequence_4d.json').read_text())['timestamps_s']==[1.,2.,3.,4.,5.]
    assert not list(output.glob('sequence_4d_*.dat'))
