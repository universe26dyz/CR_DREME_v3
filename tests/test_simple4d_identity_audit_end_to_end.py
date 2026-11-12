import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import nibabel as nib
from cardioresp4d.simple4d.coordinates import Simple4DWorldCoordinateAdapter
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import global_latent_pca
from cardioresp4d.experiments.simple4d_resp_v1_identity_render import dense_field,save_lps_nifti,volume_grid,withheld_codes

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('identity_end_to_end_cli',ROOT/'scripts/audit_simple4d_resp_v1_checkpoints.py')
cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


class Field(nn.Module):
    def __init__(self, coords, sign=1): super().__init__(); self.coordinates=coords; self.sign=sign
    def forward_normalized(self,x,code):
        dvf=x*0 + code[:,None,:3]*.005*self.sign
        return dvf,dvf
    def forward(self,world,code):
        _,dvf=self.forward_normalized(self.coordinates.world_to_normalized(world),code)
        mm=self.coordinates.normalized_deformation_to_world_mm(dvf)
        return {'displacement_mm':mm,'reference_points_mm':world+mm}


class FakePSF(nn.Module):
    def sample(self,centers,resolution_mm,**kwargs): return centers[:,None,:].expand(-1,2,-1)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__(); self.canonical_weight=nn.Parameter(torch.tensor(1.))
        coords=Simple4DWorldCoordinateAdapter([-2]*3,[2]*3,[2]*3)
        self.forward_deformation=Field(coords); self.inverse_deformation=Field(coords,-1); self.psf=FakePSF()
    def canonical(self,points,return_features=False): return torch.sigmoid(points[...,0]/4),points
    def encode_observation(self,obs,*,sample): assert not sample; return {'mu':obs.vector[None,:]}


def test_chunked_translation_and_nifti(tmp_path):
    model=FakeModel(); x=torch.randn(11,3); code=torch.ones(1,32)
    field,det,cycle=dense_field(model,code,x,3)
    assert np.allclose(field,.01) and np.allclose(det,1) and np.max(cycle)<1e-6
    lower,spacing,shape=volume_grid([-2]*3,[2]*3,[2]*3)
    path=tmp_path/'v.nii.gz'; save_lps_nifti(path,np.zeros(shape,dtype=np.float32),lower,spacing)
    nii=nib.load(path); assert nii.shape==(3,3,3) and np.allclose(nii.affine@np.array([0,0,0,1]),[2,2,-2,1])
    v=withheld_codes([0],[4],0,.25,1); assert v['alpha']==.25 and np.allclose(v['time'],[1])


def test_synthetic_all_stages(tmp_path,monkeypatch):
    model=FakeModel(); cohort=[]; groups=[]
    for vi,view in enumerate(('SAX','2CH','4CH')):
        for li in range(2):
            frames=[]; group=[]
            for i in range(6):
                signal=float(np.sin(i*.7)); vec=np.zeros(32,dtype=np.float32); vec[0]=signal; vec[1]=np.cos(i*.7)*.1; vec+=li
                image=torch.full((1,4,4),.5+.1*signal)
                frame=SimpleNamespace(vector=torch.tensor(vec),image=image,view=view,slice_id=str(li),dynamic_frame_id=vi*100+li*10+i,timestamp_s=i*.5+vi*.05,normalized_slice_position=li-.5,center_mm=torch.zeros(3),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=1.,row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1]))
                frames.append(frame); group.append(vec)
            cohort.append((view,str(li),tuple(frames))); groups.append(np.stack(group))
    pca=global_latent_pca(groups); diag={k:.2 for k in cli.SUMMARY_FIELDS}
    diag.update({'latent_dynamic_pc1_explained_variance_ratio':float(pca['explained_variance_ratios'][0]),'latent_dynamic_pc2_explained_variance_ratio':float(pca['explained_variance_ratios'][1]),'latent_dynamic_rank_95pct':pca['rank95'],'latent_mu_dynamic_rms':pca['latent_dynamic_rms'],'static_baseline_sampled_mse_full':1.,'diagnostic_sampled_reconstruction_mse_full':.8})
    static_path=tmp_path/'static.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((3,3,3),dtype=np.float32),np.diag([-2.,-2.,2.,1.])),static_path)
    static_manifest=tmp_path/'static.json'; static_manifest.write_text(json.dumps({'outputs':{'volume':{'path':str(static_path)}}}))
    monkeypatch.setattr(cli,'load_inputs',lambda args: ({},{'hashes':{'static_volume':cli.output_sha256(static_path)}},model,cohort))
    monkeypatch.setattr(cli,'load_checkpoint',lambda path,step,config,prov,model: {'diagnostics':{f'step_{step}':diag}})
    run=tmp_path/'run'; run.mkdir(); (run/'checkpoint_step_2000.pt').write_bytes(b'unchanged'); out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['audit','--manifest','m','--qc-table','q','--canonical-domain','d','--static-manifest',str(static_manifest),'--run-dir',str(run),'--output-dir',str(out),'--checkpoint-steps','2000','--stage','all','--chunk-size','1024'])
    cli.main()
    for name in ('00_summary','01_latent','02_surrogate','03_coherence','04_dvf','05_withheld'):
        assert (out/name/'stage_manifest.json').exists()
        assert list((out/name).rglob('*.png'))
    latent=out/'01_latent/step_2000'
    for name in ('06_pc1_relative_time_SAX.png','09_pc1_periodogram_SAX.png','12_pc1_autocorrelation_SAX.png','periodicity_by_location.csv','periodicity_summary.json'):
        assert (latent/name).stat().st_size > 0
    latent_manifest=json.loads((out/'01_latent/stage_manifest.json').read_text())
    for name,digest in latent_manifest['output_sha256'].items():
        if name.endswith(('.csv','.json')): assert cli.output_sha256(out/'01_latent'/name)==digest
    assert (out/'audit_report.md').exists() and (out/'comparison/checkpoint_comparison.json').exists()
    assert json.loads((out/'audit_manifest.json').read_text())['recommended_authority_checkpoint'] is None
    assert (out/'04_dvf/step_2000/volumes/SAX_low_pc1.nii.gz').exists()
    dvf=out/'04_dvf/step_2000'
    for view in ('SAX','2CH','4CH'):
        assert (dvf/f'{view}_dvf_magnitude_overlay.png').stat().st_size > 0
        assert (dvf/f'{view}_dvf_high_minus_low_overlay.png').stat().st_size > 0
        for state in ('low','mid','high'):
            assert (dvf/f'{view}_dvf_components_{state}.png').stat().st_size > 0
    assert (out/'04_dvf/comparison_SAX_dvf_magnitude_overlay.png').exists()
    assert (out/'04_dvf/comparison_SAX_high_minus_low.png').exists()
    manifest=json.loads((out/'04_dvf/stage_manifest.json').read_text())
    for name,digest in manifest['output_sha256'].items():
        if name.endswith(('.csv','.json')): assert cli.output_sha256(out/'04_dvf'/name) == digest
    with np.load(dvf/'SAX_anatomy_overlay_data.npz') as data:
        low=data['low_field_planes_axial']; high=data['high_field_planes_axial']
        assert low.shape == high.shape and low.shape[:2] != (16,16)
        assert low.shape[:2] == data['low_image_planes_axial'].shape == data['low_mask_planes_axial'].shape
        assert np.allclose(np.linalg.norm(high-low,axis=-1),np.linalg.norm((high-low)[0,0]),atol=1e-7)
    assert (dvf/'dvf_anatomy_summary.csv').exists() and (dvf/'crop_manifest.json').exists()
    assert (run/'checkpoint_step_2000.pt').read_bytes()==b'unchanged'
