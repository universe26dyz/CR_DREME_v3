import pytest
import torch
from cardioresp4d.simple4d.coordinates import Simple4DWorldCoordinateAdapter


def test_coordinates():
    a = Simple4DWorldCoordinateAdapter([-100,-80,-60], [100,120,140], [1,2,3])
    world = torch.tensor([[-100.,-80,-60], [0,20,40], [100,120,140]])
    norm = torch.tensor([[-1.,-1,-1], [0,0,0], [1,1,1]])
    assert torch.allclose(a.world_to_normalized(world), norm)
    assert torch.allclose(a.normalized_to_world(norm), world)
    assert torch.allclose(a.raw_deformation_to_world_mm(torch.ones(3)), torch.tensor([1.,2,3]))
    b = Simple4DWorldCoordinateAdapter([-100,-80,-60], [100,120,140], [1,1,1])
    assert torch.allclose(b.raw_deformation_to_world_mm(torch.tensor([1.,2,3])), torch.tensor([1.,2,3]))
    with pytest.raises(ValueError): Simple4DWorldCoordinateAdapter([0]*3,[1]*3,[1]*3,canonical_spacing_mm=[2]*3)
