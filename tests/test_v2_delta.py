"""Focused deterministic tests for the v2 NeSVoR PSF, SINR, mean-slice, FiLM and loss delta."""
from __future__ import annotations
import csv, json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cardioresp4d.reference.build_mean_slices import build_mean_slices
from cardioresp4d.rendering.psf_renderer import AnisotropicPSFRenderer, psf_sigma_mm, psf_covariance_mm2
from cardioresp4d.models.hash_inr import CanonicalINR
from cardioresp4d.models.sinr_mbc import RespiratorySINRMBC, CardiacSINRMBC
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.models.uncertainty import NamespacedObservationUncertainty
from cardioresp4d.models.film_motion_encoder import GeometryFiLMMotionEncoder
from cardioresp4d.losses.stage_aware import gaussian_nll, masked_reference_mse

class _ConstantINR(nn.Module):
    def forward(self,x,return_features=False):
        value=x.new_ones((*x.shape[:-1],1)); return (value,x) if return_features else value
class _ZeroMotion(nn.Module):
    def forward(self,x): return {'reference_points_mm':x}

class V2DeltaTest(unittest.TestCase):
    def test_mean_slices_average_only_valid_and_omit_whole_invalid_location(self):
        class Fake:
            def __init__(self,*args,**kw): self._rows=[{'view':'SAX','slice_id':'a','qc_valid':'1','source_file_token':'a0'},{'view':'SAX','slice_id':'a','qc_valid':'1','source_file_token':'a1'},{'view':'2CH','slice_id':'gone','qc_valid':'0','source_file_token':'b0'}]
            def __getitem__(self,i): return {'image':torch.tensor([[float(i+1)]]).numpy(),'timestamp_s':float(i),'geometry':{'rows':1},'source_file_token':self._rows[i]['source_file_token']}
        with tempfile.TemporaryDirectory() as tmp, patch('cardioresp4d.reference.build_mean_slices.CardioRespDataset',Fake):
            out=Path(tmp); manifest=build_mean_slices(out/'manifest.csv',out)
            with manifest.open(newline='', encoding='utf-8') as handle: rows=list(csv.DictReader(handle))
            self.assertEqual(1,len(rows)); self.assertEqual('SAX',rows[0]['view']); self.assertEqual(2,int(rows[0]['n_valid'])); self.assertAlmostEqual(1.5,float(__import__('numpy').load(out/rows[0]['image_file'])[0,0])); self.assertNotIn('gone',manifest.read_text())
    def test_psf_constant_identity_thickness_orientation_and_gradient(self):
        renderer=AnisotropicPSFRenderer(torch.eye(4)); geometry=dict(center_mm=torch.zeros(1,3),row_direction=torch.tensor([[1.,0,0]]),column_direction=torch.tensor([[0.,1,0]]),normal=torch.tensor([[0.,0,1.]]),pixel_spacing_mm=torch.tensor([[2.,3.]]),thickness_mm=torch.tensor([8.]),height=3,width=4)
        plain=renderer(_ConstantINR(),**geometry); warped=renderer(_ConstantINR(),motion=_ZeroMotion(),**geometry); torch.testing.assert_close(plain['predicted_slice'],torch.ones(1,1,3,4)); torch.testing.assert_close(plain['predicted_slice'],warped['predicted_slice']); expected=torch.tensor([[2.4,3.6,8.]])/(2*torch.sqrt(2*torch.log(torch.tensor(2.)))); torch.testing.assert_close(psf_sigma_mm(geometry['pixel_spacing_mm'],geometry['thickness_mm']),expected)
        oblique_row=torch.tensor([[2**-.5,2**-.5,0.]]); oblique_column=torch.tensor([[0.,0.,1.]]); oblique_normal=torch.cross(oblique_row,oblique_column,dim=-1); covariance=psf_covariance_mm2(oblique_row,oblique_column,oblique_normal,geometry['pixel_spacing_mm'],geometry['thickness_mm']); torch.testing.assert_close(covariance,covariance.transpose(-1,-2)); self.assertGreater(abs(float(covariance[0,0,1])),0.)
        inr=CanonicalINR(levels=2,features_per_level=2,hash_size=32,min_resolution=4,max_resolution=8,hidden_dim=8,latent_dim=4); renderer(inr,**geometry)['predicted_slice'].mean().backward(); self.assertTrue(torch.isfinite(inr.intensity_head.weight.grad).all())
    def test_sinr_mbc_zero_mm_scale_boundary_and_backward(self):
        lo,hi=torch.tensor([-10.,-20.,-30.]),torch.tensor([10.,20.,30.]); resp=RespiratorySINRMBC(lo,hi,resolutions=(4,4,4)); card=CardiacSINRMBC(lo,hi,resolution=4); point=torch.zeros(1,2,3,requires_grad=True); self.assertTrue(torch.equal(resp(point),torch.zeros(1,3,2,3))); self.assertTrue(torch.equal(card(torch.tensor([[[-10.,0,0]]])),torch.zeros(1,1,3)))
        with torch.no_grad(): resp.levels[0].sinr[-1].bias[0]=2.
        value=resp(point); self.assertGreater(float(value[0,0,0,0]),1.9); self.assertEqual([20.,40.,60.],resp.levels[0].grid_metadata()['actual_control_point_spacing_mm']); value.sum().backward(); self.assertTrue(torch.isfinite(point.grad).all())
    def test_uncertainty_namespaces_film_and_nll_oracle(self):
        latent=torch.randn(2,3,2,2,4,requires_grad=True); u=NamespacedObservationUncertainty(4,num_mean_slices=2,num_dynamic_frames=3,embedding_dim=2); mean=u(latent,torch.tensor([0,1]),torch.ones(3)/3,namespace='mean_slice',enabled=True); dyn=u(latent,torch.tensor([0,1]),torch.ones(3)/3,namespace='dynamic_frame',enabled=True); self.assertTrue(torch.all(mean['total_variance']>0)); self.assertFalse(u(latent,torch.tensor([0,1]),torch.ones(3)/3,namespace='mean_slice',enabled=False)['enabled']); (mean['total_variance'].mean()+dyn['total_variance'].mean()).backward(); self.assertIsNotNone(u.embeddings['mean_slice'].weight.grad); self.assertIsNotNone(u.embeddings['dynamic_frame'].weight.grad)
        encoder=GeometryFiLMMotionEncoder(channels=8); result=encoder(torch.randn(2,1,8,8),center_mm=torch.zeros(2,3),row_direction=torch.tensor([[1.,0,0],[1.,0,0]]),column_direction=torch.tensor([[0.,1,0],[0.,1,0]]),normal=torch.tensor([[0.,0,1],[0.,0,1.]]),pixel_spacing_mm=torch.ones(2,2),slice_thickness_mm=torch.ones(2,1)*8); self.assertEqual((2,3,3),tuple(result['resp_scores'].shape)); self.assertEqual((2,1,3),tuple(result['card_scores'].shape)); result['resp_scores'].sum().backward(); self.assertTrue(torch.isfinite(encoder.resp.weight.grad).all())
        p=torch.tensor([1.],requires_grad=True); target=torch.tensor([0.]); var=torch.tensor([2.]); torch.testing.assert_close(gaussian_nll(p,target,var),torch.nn.GaussianNLLLoss(full=False,reduction='mean')(p,target,var)); masked_reference_mse(p,target,torch.zeros_like(p)).backward(); self.assertEqual(0.,float(p.grad))
    def test_tiny_dynamic_psf_sinr_film_uncertainty_backward(self):
        inr=CanonicalINR(levels=2,features_per_level=2,hash_size=32,min_resolution=4,max_resolution=8,hidden_dim=8,latent_dim=4); encoder=GeometryFiLMMotionEncoder(channels=8); lo,hi=torch.tensor([-20.,-20.,-20.]),torch.tensor([20.,20.,20.]); mbc=RespiratorySINRMBC(lo,hi,resolutions=(4,4,4));
        with torch.no_grad(): mbc.levels[0].sinr[-1].bias[0]=.02
        image=torch.randn(1,1,4,4); geo=dict(center_mm=torch.zeros(1,3),row_direction=torch.tensor([[1.,0,0]]),column_direction=torch.tensor([[0.,1,0]]),normal=torch.tensor([[0.,0,1.]]),pixel_spacing_mm=torch.ones(1,2),slice_thickness_mm=torch.ones(1,1)*6)
        scores=encoder(image,**geo); resp=ScoreWeightedMBCField(mbc,scores['resp_scores']); cardiac=ScoreWeightedMBCField(CardiacSINRMBC(lo,hi,resolution=4),scores['card_scores']); render_geo={name:value for name,value in geo.items() if name!='slice_thickness_mm'}; render_geo.update({'thickness_mm':geo['slice_thickness_mm'].squeeze(-1),'height':4,'width':4}); output=AnisotropicPSFRenderer(torch.eye(4))(inr,**render_geo,motion=SequentialPullbackMotion(resp,cardiac)); u=NamespacedObservationUncertainty(4,num_mean_slices=1,num_dynamic_frames=1,embedding_dim=2); variance=u(output['latent_samples'],torch.tensor([0]),output['psf_weights'],namespace='dynamic_frame',enabled=True)['total_variance']; loss=gaussian_nll(output['predicted_slice'].squeeze(1),torch.zeros(1,4,4),variance); loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertIsNotNone(inr.intensity_head.weight.grad); self.assertIsNotNone(encoder.resp.weight.grad); self.assertIsNotNone(mbc.levels[0].sinr[-1].bias.grad)
if __name__=='__main__': unittest.main()
