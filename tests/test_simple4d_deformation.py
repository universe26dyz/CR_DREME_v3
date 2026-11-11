import torch
from torch import nn
from third_party.simple4d.simple4d.networks import DeformCodeAdaptive
from cardioresp4d.simple4d.coordinates import Simple4DWorldCoordinateAdapter
from cardioresp4d.simple4d.deformation import Simple4DRespiratoryDeformation
from cardioresp4d.simple4d.losses import respiratory_loss


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__(); self.value = torch.tensor(value)
    def forward(self, x, code): return torch.ones_like(x) * self.value.to(x)


def test_displacement():
    a = Simple4DWorldCoordinateAdapter([-100]*3, [100]*3, [1,2,3])
    model = Simple4DRespiratoryDeformation(a)
    assert isinstance(model.network, DeformCodeAdaptive)
    x, z = torch.randn(1, 5, 3), torch.zeros(1, 32)
    model.network = Constant([0,0,0])
    assert torch.equal(model(x,z)['reference_points_mm'], x)
    model.network = Constant([1,1,1])
    assert torch.allclose(model(x,z)['reference_points_mm'], x + torch.tensor([1.,2,3]))


def test_loss_gradients():
    a = Simple4DWorldCoordinateAdapter([-100]*3, [100]*3, [1]*3)
    f, inv = Simple4DRespiratoryDeformation(a), Simple4DRespiratoryDeformation(a)
    points = torch.randn(1, 4, 3, requires_grad=True)
    mu = torch.randn(1,32,requires_grad=True); logvar = torch.zeros_like(mu,requires_grad=True)
    out = respiratory_loss(torch.zeros(4),torch.ones(4),mu,logvar,points,f,inv,mu,1)
    assert all(torch.isfinite(v) for v in out.values())
    out['total'].backward()
    assert f.network.layers[0].weight.grad is not None and inv.network.layers[0].weight.grad is not None and mu.grad is not None


def test_upstream_regularizers_zero_and_translation():
    from cardioresp4d.simple4d.losses import jacobian_loss, dvf_magnitude_loss
    x = torch.randn(1, 4, 3, requires_grad=True)
    for value in (0., 0.01):
        dvf = x * 0 + value
        assert torch.isfinite(jacobian_loss(x, dvf))
        assert torch.isfinite(dvf_magnitude_loss(dvf))
