import pytest
import torch
from cardioresp4d.simple4d.encoder import Simple4DViewConditionedMotionCodeVAE


def test_encoder():
    net = Simple4DViewConditionedMotionCodeVAE()
    image = torch.randn(4, 1, 64, 64)
    pos = torch.tensor([-1., 0., .5, 1.])
    view = torch.tensor([0, 1, 2, 0])
    net.eval()
    out = net(image, pos, view)
    assert all(out[k].shape == (4, 32) for k in ('code', 'mu', 'logvar'))
    assert torch.equal(out['code'], out['mu'])
    net.train()
    out = net(image, pos, view)
    assert all(out[k].requires_grad for k in ('code', 'mu', 'logvar'))
    net.kl_loss(out['mu'], out['logvar']).backward()
    assert net.fc_mu.weight.grad is not None
    with pytest.raises(ValueError): net(image, pos, torch.tensor([0, 1, 2, 3]))
    with pytest.raises(ValueError): net(image, pos * 2, view)
