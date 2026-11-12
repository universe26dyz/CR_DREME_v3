import numpy as np
import pytest
from cardioresp4d.experiments.simple4d_cardioresp_identity import pure_pc1_codes, difference_summary, vector_difference_summary, decompose_motion, residual_metrics, timestamp_interpolated_code
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import global_latent_pca


def test_pure_pc1():
    t=np.linspace(0,2*np.pi,40,endpoint=False); means=np.stack([3+np.sin(t),4+.1*np.cos(t)],axis=1)
    pca=global_latent_pca([means]); pure=pure_pc1_codes(means,pca['scores'][0][:,0],pca['basis'][:,0])
    assert np.allclose((pure-pure.mean(0))@pca['basis'][:,1],0,atol=1e-10)
    assert difference_summary(means,pure)['rmse']>0
    assert vector_difference_summary(np.column_stack([means,np.zeros(len(means))]),np.column_stack([pure,np.zeros(len(means))]))['rms_magnitude_mm']>0


def test_translation_and_local_deformation():
    x=np.linspace(-1,1,101); translation=np.tile([1.,2.,3.],(len(x),1)); det=np.ones(len(x))
    a=decompose_motion(translation,det); assert a['residual_rms_mm']<1e-12 and a['tz_mm']==3
    local=translation.copy(); local[:,0]+=.2*x
    b=decompose_motion(local,np.full(len(x),1.2)); assert b['residual_rms_mm']>0 and b['tx_mm']==pytest.approx(1)
    assert b['jacobian_mean_abs_deviation']==pytest.approx(.2)


def test_residual_and_interpolation():
    target=np.array([0.,1.,2.]); assert residual_metrics(target,target)['mse']==0
    assert residual_metrics(target,target+1)['mse']==1
    code=timestamp_interpolated_code([0.],[4.],0.,.25,1.); assert code['alpha']==.25 and code['time'][0]==1.
