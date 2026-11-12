import nibabel as nib
import numpy as np
import pytest
from cardioresp4d.experiments.simple4d_resp_v1_r0 import resolve_static_reference_spacing


def test_spacing_authority(tmp_path):
    path = tmp_path/'static_reference.nii.gz'
    manifest = {'nesvor':{'command':['nesvor','--output-resolution','1.0']}, 'outputs':{'volume':{'path':str(path)}}}
    nib.save(nib.Nifti1Image(np.zeros((2,2,2)),np.eye(4)),path)
    assert resolve_static_reference_spacing(manifest, [1,1,1])['resolved_reference_spacing_mm'] == [1,1,1]
    nib.save(nib.Nifti1Image(np.zeros((2,2,2)),np.diag([1.2,1,1,1])),path)
    with pytest.raises(ValueError): resolve_static_reference_spacing(manifest, [1,1,1])
    nib.save(nib.Nifti1Image(np.zeros((2,2,2)),np.diag([2,2,2,1])),path)
    manifest['nesvor']['command'][-1] = '2'
    with pytest.raises(ValueError): resolve_static_reference_spacing(manifest, [1,1,1])
