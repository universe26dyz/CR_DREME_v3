import numpy as np
import torch
from third_party.simple4d.simple4d.model import GL_NODES, GL_WEIGHTS
from cardioresp4d.adapters.simple4d_physical_thick_slice import PhysicalUpstreamThickSliceAdapter
from cardioresp4d.experiments.simple4d_native_sax_collapse import latent_stats, temporal_stats


def test_physical_quadrature_oblique_and_deformation():
    adapter=PhysicalUpstreamThickSliceAdapter(); centers=torch.tensor([[1.,2.,3.],[0.,0.,0.]])
    normals=torch.tensor([[2**-.5,2**-.5,0.],[0.,0.,1.]])
    resolution=torch.tensor([[1.,1.,4.],[1.,1.,8.]])
    seen=[]
    def motion(x):
        seen.append(x.clone()); return {'reference_points_mm':x+0.3}
    def canonical(x,return_features=False): return x.square().sum(-1),None
    result=adapter(canonical,centers,resolution,row_direction=normals,column_direction=normals,normal=normals,motion=motion)
    expected=centers[:,None]+normals[:,None]*resolution[:,2,None,None]*torch.tensor(GL_NODES)[None,:,None]/2
    assert expected.shape==(2,5,3)
    torch.testing.assert_close(seen[0],expected)
    weights=torch.tensor(GL_WEIGHTS)*torch.exp(-.5*(torch.tensor(GL_NODES)*2)**2); weights/=weights.sum()
    assert torch.isclose(weights.sum(),torch.tensor(1.))
    torch.testing.assert_close(result['predicted_intensity'],((expected+0.3).square().sum(-1)*weights).sum(-1))


def test_collapse_statistics():
    static=latent_stats(np.ones((4,3))); assert static['dynamic_rms']==0 and static['rank95']==0
    varying=latent_stats(np.array([[1,0],[2,0],[3,0]],float)); assert varying['dynamic_rms']>0
    observed=np.array([[[0.,1.]],[[2.,3.]]]); predicted=np.ones_like(observed)
    result=temporal_stats(observed,predicted)
    assert result['observed_mean_temporal_std']==1 and result['rendered_temporal_std_ratio']==0
    assert result['adjacent_difference_ratio']==0 and result['dynamic_only_mse']==1


def test_matches_upstream_regular_stack_predict():
    from third_party.simple4d.simple4d.model import SIMPLE4D
    upstream=SIMPLE4D.__new__(SIMPLE4D); upstream.thick_slice=True
    thickness=0.8; upstream._z_offsets=torch.tensor(GL_NODES)*thickness/2
    w=torch.tensor(GL_WEIGHTS)*torch.exp(-.5*(upstream._z_offsets/(thickness/4))**2); upstream._thick_weights=w/w.sum()
    upstream.dvf=lambda x,z: 0.1*x+z[:,None,:]
    upstream.feature_net=lambda x: x.square().sum(-1,keepdim=True)
    coords=torch.tensor([[[.1,.2,.3],[-.3,.4,.2]]]); code=torch.tensor([[.1,.2,.3]])
    expected=upstream.predict(coords,code)[0].squeeze(-1).squeeze(0)
    adapter=PhysicalUpstreamThickSliceAdapter(); centers=coords[0]; normal=torch.tensor([[1.,0,0]]).expand(2,-1); resolution=torch.tensor([[1.,1.,thickness]]).expand(2,-1)
    result=adapter(lambda x,return_features: ((x*x).sum(-1),None),centers,resolution,row_direction=normal,column_direction=normal,normal=normal,motion=lambda x:{'reference_points_mm':x+0.1*x+code[:,None,:]})
    torch.testing.assert_close(result['predicted_intensity'],expected)


def test_s05_shares_seeded_core_and_optimizer():
    from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiView, Simple4DCoreMultiViewThickSlice, make_s1_optimizer
    options={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    torch.manual_seed(6); a=Simple4DCoreMultiView([-10]*3,[10]*3,[1]*3,use_tcnn=False,ngp_options=options)
    torch.manual_seed(6); b=Simple4DCoreMultiViewThickSlice([-10]*3,[10]*3,[1]*3,use_tcnn=False,ngp_options=options)
    assert set(a.state_dict())==set(b.state_dict())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert type(b.psf) is PhysicalUpstreamThickSliceAdapter and not hasattr(b.respiratory_encoder,'view_embedding')
    oa,sa=make_s1_optimizer(a); ob,sb=make_s1_optimizer(b)
    assert [(g['name'],g['lr']) for g in oa.param_groups]==[(g['name'],g['lr']) for g in ob.param_groups]
    assert sa.state_dict()==sb.state_dict()


def test_s05_update_and_resume_reuses_s1_runner(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location("s1_tests",Path(__file__).with_name("test_simple4d_core_mv_model.py"))
    s1_tests=importlib.util.module_from_spec(spec); spec.loader.exec_module(s1_tests)
    from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
    monkeypatch.setattr(s1_tests,'Simple4DCoreMultiView',Simple4DCoreMultiViewThickSlice)
    s1_tests.test_joint_update_and_resume(tmp_path)


def test_s05_evaluator_adapter(tmp_path):
    from cardioresp4d.experiments.simple4d_unified_evaluation import S05CoreMVThickSliceAdapter
    path=tmp_path/'checkpoint.pt'; path.write_bytes(b'checkpoint')
    adapter=S05CoreMVThickSliceAdapter(None,path)
    assert adapter.method_id=='S05' and adapter.supports_target(None,type('Obs',(),{'view':'2CH'})(),None,'v1')
    adapter.verify_unchanged()
