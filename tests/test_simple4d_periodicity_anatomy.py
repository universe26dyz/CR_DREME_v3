import json
import numpy as np
import nibabel as nib
import pytest
import torch
from pathlib import Path
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import periodicity_analysis, representative_locations
from cardioresp4d.experiments.simple4d_resp_v1_anatomy import derive_anatomy_crop, crop_points, support_at_world, supported_stats


def test_irregular_periodicity():
    rng=np.random.default_rng(4); times=100+np.cumsum(rng.uniform(.17,.29,240)); signal=np.sin(2*np.pi*.22*(times-times[0]))
    result,arrays=periodicity_analysis(times,signal)
    assert abs(result['dominant_frequency_hz']-.22)<.01
    assert arrays['relative_time_s'][0]==0 and abs(result['autocorrelation_peak_lag_s']-1/.22)<.5
    assert periodicity_analysis(times,np.ones(len(times)))[0]['exclusion_reason']=='missing_or_low_variance'
    assert periodicity_analysis(times[:4],signal[:4])[0]['exclusion_reason']=='too_few_frames'
    assert representative_locations([('a',-.8),('b',0),('c',.8)]) == ['a','b','c']


def test_support_crop_and_geometry(tmp_path):
    arr=np.zeros((10,10,10),np.float32); arr[3:6,2:7,4:8]=10.
    # NIfTI RAS axis direction converts to external LPS.
    affine=np.diag([-1.,-1.,1.,1.]); path=tmp_path/'static.nii.gz'; nib.save(nib.Nifti1Image(arr,affine),path)
    args=(path,[0,0,0],[9,9,9],[1,1,1])
    crop,mask,inv=derive_anatomy_crop(*args,source_sha256='fixture',padding_mm=1)
    assert crop==derive_anatomy_crop(*args,source_sha256='fixture',padding_mm=1)[0]
    assert crop['shape'] != [16,16,16] and crop['source_sha256']=='fixture'
    points=crop_points(crop['lower_lps_mm'],crop['spacing_mm'],tuple(crop['shape']),0,int(np.prod(crop['shape'])),torch.device('cpu'))
    assert np.allclose(points[0].numpy(),crop['lower_lps_mm'])
    inside=support_at_world(points.numpy(),mask,inv); assert inside.any() and not inside.all()
    field=np.tile([1.,2.,3.],(len(points),1)); field[~inside]=100
    stats=supported_stats(field,np.ones(len(points)),inside)
    assert stats['anatomical_support']['mean_abs_dx_mm']==1 and stats['anatomical_support']['mean_abs_dy_mm']==2
    assert stats['anatomical_support']['mean_abs_dz_mm']==3 and stats['all_crop']['max_magnitude_mm'] > stats['anatomical_support']['max_magnitude_mm']
    assert np.allclose(np.linalg.norm(np.array([1.,2.,3.])-np.array([.5,1,2])),1.5)
    with pytest.raises(ValueError): derive_anatomy_crop(path,[100,100,100],[110,110,110],[1,1,1],source_sha256='fixture')
