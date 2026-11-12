import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pytest
from torch import nn
from PIL import Image
from cardioresp4d.simple4d.coordinates import Simple4DWorldCoordinateAdapter

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cardioresp_cli',ROOT/'scripts/audit_simple4d_cardioresp_identity.py')
cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


class Field(nn.Module):
    def __init__(self,coords,sign): super().__init__(); self.coordinates=coords; self.sign=sign
    def forward_normalized(self,x,code):
        dvf=x*0+code[:,None,:3]*.01*self.sign
        return dvf,dvf
    def forward(self,points,code):
        _,dvf=self.forward_normalized(self.coordinates.world_to_normalized(points),code)
        mm=self.coordinates.normalized_deformation_to_world_mm(dvf)
        return {'displacement_mm':mm,'reference_points_mm':points+mm}


class FakePSF(nn.Module):
    def __init__(self): super().__init__(); self.calls=[]
    def sample(self,points,resolution_mm,**kw):
        self.calls.append(points.detach().clone()); return points[:,None,:].expand(-1,2,-1)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.tensor(1.)); coords=Simple4DWorldCoordinateAdapter([-2]*3,[2]*3,[2]*3)
        self.forward_deformation=Field(coords,1); self.inverse_deformation=Field(coords,-1); self.psf=FakePSF()
    def canonical(self,x,return_features=True): return torch.sigmoid(x[...,0]),x
    def encode_observation(self,obs,*,sample): assert not sample; return {'mu':obs.code[None]}


@pytest.mark.parametrize("with_box", [False, True])
def test_all_stages_synthetic(tmp_path,monkeypatch,with_box):
    audit=tmp_path/'audit'; run=tmp_path/'run'; run.mkdir(); steps=[2000,5000,10000]
    for step in steps: (run/f'checkpoint_step_{step}.pt').write_bytes(f'checkpoint{step}'.encode())
    frames=[]; means=[]; rows=[]
    for i in range(5):
        code=np.zeros(32,np.float32); code[0]=i*.25; code[1]=.1*np.sin(i)
        frame=SimpleNamespace(code=torch.tensor(code),image=torch.rand(1,4,4),timestamp_s=float(i),dynamic_frame_id=i,normalized_slice_position=0.,center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1]),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=1.)
        frames.append(frame); means.append(code); rows.append({'view':'SAX','slice_id':'s01','dynamic_frame_id':i,'timestamp_s':float(i),'pc1':float(code[0]-.5),'observation':frame})
    crop={'lower_lps_mm':[-2.,-2.,-2.],'upper_lps_mm':[2.,2.,2.],'spacing_mm':[2.,2.,2.],'shape':[3,3,3]}
    infos={}; basis=np.zeros(32); basis[0]=1.
    for step in steps:
        checkpoint=run/f'checkpoint_step_{step}.pt'; digest=cli.output_sha256(checkpoint)
        info={'step':step,'checkpoint':checkpoint,'checkpoint_sha256':digest,'crop':crop,'anchor':'s01','anchor_rows':rows,'anchor_means':np.stack(means),'anchor_scores':np.array([r['pc1'] for r in rows]),'basis':basis,'pca':{'explained_variance_ratios':np.array([.9,.1]),'rank95':2},'basis_sha256':'fixture','selected':{'low':0,'mid':2,'high':4}}
        infos[step]=info; dynamic=audit/'04_dvf'/f'step_{step}'/'dynamic_visualization'; (dynamic/'sequence_frames').mkdir(parents=True); (dynamic/'selected_states').mkdir()
        (dynamic/'visualization_manifest.json').write_text(json.dumps({'checkpoint_sha256':digest,'crop':crop,'anchor_slice_id':'s01','intensity_window':[0.,1.],'slice_indices_lps_xyz':[1,1,1],'gif_frame_ms':150}))
        with (dynamic/'sequence_metadata.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['dynamic_frame_id','timestamp_s']); writer.writeheader(); writer.writerows([{'dynamic_frame_id':i,'timestamp_s':i} for i in range(5)])
        for i in range(5): Image.new('RGB',(1200,400),(i*10,0,0)).save(dynamic/'sequence_frames'/f'frame_{i:04d}.png')
        for state in ('low','mid','high'): Image.new('RGB',(1200,400),'white').save(dynamic/'selected_states'/f'SAX_{state}.png')
    from cardioresp4d.roi.cardiac_box import CardiacBox
    box=CardiacBox(np.zeros(3),np.array([2.,2.,2.])) if with_box else None
    box_path=tmp_path/'box.yaml'; box_path.write_text('roi: {}\n')
    model=FakeModel(); base={'config':{},'provenance':{'hashes':{'static_volume':'fixture'}},'model':model,'cohort':(('SAX','s01',tuple(frames)),),'static_volume':tmp_path/'unused','cardiac_box':box}
    monkeypatch.setattr(cli,'load_base',lambda args:base); monkeypatch.setattr(cli,'prepare_step',lambda args,step,base: dict(infos[step]))
    monkeypatch.setattr(cli,'load_checkpoint',lambda *args:None)
    monkeypatch.setattr(cli,'derive_anatomy_crop',lambda *args,**kwargs:(crop,np.ones((3,3,3),dtype=np.uint8),np.diag([-0.5,-0.5,.5,1.])))
    arguments=['audit','--audit-root',str(audit),'--run-dir',str(run),'--checkpoint-steps',*map(str,steps),'--stage','all','--chunk-size','8']
    if with_box: arguments += ['--cardiac-box-config',str(box_path)]
    monkeypatch.setattr(sys,'argv',arguments)
    cli.main(); out=audit/'07_cardioresp_identity'
    for step in steps:
        pure=out/f'step_{step}'/'pure_pc1'; motion=out/f'step_{step}'/'motion_decomposition'
        assert (pure/'gif/SAX_pure_pc1_temporal_sequence.gif').exists()
        assert (pure/'gif/SAX_full_vs_pure_pc1_temporal.gif').exists()
        assert (pure/'non_pc1_contribution.csv').exists()
        assert (motion/'frame_motion_decomposition.csv').exists()
        assert (run/f'checkpoint_step_{step}.pt').read_bytes()==f'checkpoint{step}'.encode()
    assert (out/'residual_audit/frame_metrics.csv').exists() and (out/'comparisons/cardioresp_identity_summary.csv').exists()
    with (out/'residual_audit/frame_metrics.csv').open(newline='') as f: metrics=list(csv.DictReader(f))
    labels={r['model'] for r in metrics}
    assert 'STATIC' in labels and all(f'{step}_v1' in labels and f'{step}_v2_time' in labels and f'{step}_v2_simple' in labels for step in steps)
    calls=model.psf.calls; assert len(calls)%3==0
    count=len(calls)//3
    assert all(torch.equal(calls[i],calls[count+i]) and torch.equal(calls[i],calls[2*count+i]) for i in range(count))
    assert (out/'stage_manifest_pure_pc1.json').exists() and (out/'stage_manifest_residual.json').exists()
    assert json.loads((out/'comparisons/cardioresp_identity_summary.json').read_text())['authority_checkpoint'] is None
    assert any(r['region']=='cardiac_box' for r in metrics) is with_box
    assert bool(list((out/'residual_audit/residuals').glob('*_cardiac_box.png'))) is with_box


def test_recomputed_pca_score_and_provenance(tmp_path,monkeypatch):
    from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import global_latent_pca
    root=tmp_path/'audit'; run=tmp_path/'run'; run.mkdir(); checkpoint=run/'checkpoint_step_2000.pt'; checkpoint.write_bytes(b'unchanged')
    digest=cli.output_sha256(checkpoint); latent=root/'01_latent'; latstep=latent/'step_2000'; latstep.mkdir(parents=True)
    frames=[]; means=[]
    for i in range(6):
        code=np.zeros(32,dtype=np.float32); code[0]=np.sin(i*.7); code[1]=.1*np.cos(i*.7)
        means.append(code); frames.append(SimpleNamespace(code=torch.tensor(code),dynamic_frame_id=i,timestamp_s=float(i)))
    pca=global_latent_pca([np.stack(means)])
    rows=[{'view':'SAX','slice_id':'s01','dynamic_frame_id':i,'timestamp_s':float(i),'normalized_slice_position':0,'pc1':float(pca['scores'][0][i,0]),'pc1_location_z':0} for i in range(6)]
    def save_rows():
        with (latstep/'frame_latent_scores.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_rows(); np.savez(latstep/'pca_basis.npz',basis=pca['basis'])
    def latent_manifest():
        files={f'step_2000/{name}':cli.output_sha256(latstep/name) for name in ('frame_latent_scores.csv','pca_basis.npz')}
        (latent/'stage_manifest.json').write_text(json.dumps({'checkpoints':{'2000':{'sha256':digest}},'input_artifact_sha256':{'static_volume':'bound'},'output_sha256':files}))
    latent_manifest(); dvf=root/'04_dvf'; step=dvf/'step_2000'; (step/'volumes').mkdir(parents=True)
    (step/'crop_manifest.json').write_text('{}'); (step/'selected_states.csv').write_text('fixture')
    for state in ('low','mid','high'):
        (step/'volumes'/f'SAX_{state}_pc1.nii.gz').write_bytes(b'fixture')
        (step/'volumes'/f'SAX_{state}_pc1.json').write_text(json.dumps({'checkpoint_sha256':digest}))
    paths={str(p.relative_to(dvf)):cli.output_sha256(p) for p in step.rglob('*') if p.is_file()}
    (dvf/'stage_manifest.json').write_text(json.dumps({'checkpoints':{'2000':{'sha256':digest}},'input_artifact_sha256':{'static_volume':'bound'},'output_sha256':paths}))
    selected={name:{'checkpoint_step':'2000','slice_id':'s01','dynamic_frame_id':str(i),'timestamp_s':str(i),'pc1':str(rows[i]['pc1'])} for name,i in (('low',0),('mid',2),('high',4))}
    monkeypatch.setattr(cli,'load_selected_volumes',lambda *args: ({},selected,{}))
    monkeypatch.setattr(cli,'load_checkpoint',lambda *args: {'diagnostics':{}})
    base={'config':{},'provenance':{'hashes':{'static_volume':'bound'}},'model':FakeModel(),'cohort':(('SAX','s01',tuple(frames)),)}
    args=SimpleNamespace(audit_root=root,run_dir=run)
    out=cli.prepare_step(args,2000,base)
    assert np.allclose(out['basis'],pca['basis'][:,0]) and checkpoint.read_bytes()==b'unchanged'
    rows[0]['pc1']+=1; save_rows(); latent_manifest()
    with pytest.raises(ValueError,match='recomputed PC1'): cli.prepare_step(args,2000,base)


def test_known_motion_improves_matched_residual():
    from cardioresp4d.experiments.simple4d_resp_v1_identity_render import predict_modes
    from cardioresp4d.experiments.simple4d_cardioresp_identity import residual_metrics, timestamp_interpolated_code
    model=FakeModel()
    obs=SimpleNamespace(image=torch.zeros(1,6,6),center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1]),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=1.)
    index=torch.arange(36); uv=torch.stack((index%6,index//6),-1)
    previous=np.zeros((1,32),dtype=np.float32); following=np.zeros_like(previous); following[0,0]=4
    interpolated=timestamp_interpolated_code(previous,following,0.,.5,1.)
    code=torch.from_numpy(interpolated['time'])
    static,v1,v2_simple,v2_time=predict_modes(model,obs,uv,(code,torch.from_numpy(interpolated['simple']),code),8)
    observed=v1.copy()
    assert residual_metrics(observed,static)['mse']>0
    assert residual_metrics(observed,v1)['mse']==0
    assert residual_metrics(observed,v2_simple)['mse']==0 and residual_metrics(observed,v2_time)['mse']==0
