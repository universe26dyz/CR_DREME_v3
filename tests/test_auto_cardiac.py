from types import SimpleNamespace
import numpy as np
import pytest
from cardioresp4d.roi.auto_cardiac import central_plane, intersect_planes, soft_box
from cardioresp4d.roi.cardiac_box import CardiacBox


def test_intersection_and_singular():
    center, residual, condition = intersect_planes(np.diag([1., 2., 3.]), np.eye(3))
    np.testing.assert_allclose(center, [1, 2, 3])
    np.testing.assert_allclose(residual, 0)
    assert condition == 1
    with pytest.raises(ValueError, match='ill-conditioned'):
        intersect_planes(np.zeros((3, 3)), [[1,0,0],[0,1,0],[1,1,0]])


def test_physical_median_and_soft_box():
    obs = [SimpleNamespace(view='SAX', slice_id=str(i), qc_valid=True, center_mm=np.array([0.,0.,z]), normal=np.array([0.,0.,1.])) for i,z in enumerate([9, 1, 5])]
    plane, bounds = central_plane(obs)
    assert plane[0] == 5 and bounds == (1, 9)
    box = CardiacBox([0,0,0], [20,20,20])
    np.testing.assert_allclose(soft_box([[0,0,0], [9,0,0], [10,0,0], [11,0,0]], box, 2), [1, .5, 0, 0])


def test_corrected_planes_use_acquisition_stack_direction():
    obs = [SimpleNamespace(view='SAX', slice_id=str(i), qc_valid=True, center_mm=np.array([0., 0., z]), normal=np.array([0.05, 0., np.sqrt(1-0.05**2)]) if i else np.array([0.,0.,1.])) for i,z in enumerate([1, 3, 2])]
    with pytest.raises(ValueError, match='acquisition stack normal'):
        central_plane(obs)
    plane, bounds = central_plane(obs, stack_normal=[0,0,1])
    assert plane[1] == '2' and bounds == (1, 3)
