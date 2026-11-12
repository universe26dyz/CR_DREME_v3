import copy
import random
import pytest
import torch
from torch import nn
from cardioresp4d.experiments import simple4d_resp_v1_r0 as r0
from cardioresp4d.simple4d.encoder import Simple4DViewConditionedMotionCodeVAE
from cardioresp4d.training.runtime_state import set_reproducibility


class Sampler:
    def __init__(self): self.rng = random.Random(12)
    def sample_step(self, observations_per_update): return [self.rng.random() for _ in range(observations_per_update)]
    def state_dict(self): return self.rng.getstate()
    def load_state_dict(self, state): self.rng.setstate(state)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.canonical = nn.Linear(1,1)
        self.respiratory_encoder = nn.Linear(1,1)
        self.forward_deformation = nn.Linear(1,1)
        self.inverse_deformation = nn.Linear(1,1)
    def freeze_r0(self): self.canonical.requires_grad_(False)


def fake_loss(model, obs, uv, step):
    x = torch.tensor([[obs]]) + torch.rand(1,1)
    y = model.inverse_deformation(model.forward_deformation(model.respiratory_encoder(x)))
    return {'total': y.square().mean()}


def test_pixels_and_explicit_posterior():
    uv = r0.sample_pixels(torch.zeros(1,3,4), 10000)
    assert uv.shape == (10000,2) and len(torch.unique(uv, dim=0)) <= 12
    net = Simple4DViewConditionedMotionCodeVAE().train()
    image = torch.randn(1,1,64,64); pos = torch.tensor([0.]); view = torch.tensor([0])
    a = net.encode_code(image, pos, view, sample=False); b = net.encode_code(image, pos, view, sample=False)
    assert torch.equal(a['code'], a['mu']) and torch.equal(a['mu'], b['mu']) and net.training
    sample = net.encode_code(image,pos,view,sample=True)
    sample['code'].sum().backward()
    assert net.fc_logvar.weight.grad is not None


def test_accumulation_and_resume(monkeypatch):
    monkeypatch.setattr(r0, 'observation_loss', fake_loss)
    monkeypatch.setattr(r0, 'sample_pixels', lambda image, count: None)
    # Sampler emits floats, so adapt the unused image for the update.
    class Obs(float):
        image = None
    original = Sampler.sample_step
    monkeypatch.setattr(Sampler, 'sample_step', lambda self, observations_per_update: [Obs(x) for x in original(self, observations_per_update)])
    set_reproducibility(6); full = r0.R0Runner(Tiny(), Sampler(), pixel_samples=2)
    initial_canonical = copy.deepcopy(full.model.canonical.state_dict())
    for _ in range(4): full.update()
    full_state = copy.deepcopy(full.model.state_dict()); full_runner = copy.deepcopy(full.state_dict())
    set_reproducibility(6); split = r0.R0Runner(Tiny(), Sampler(), pixel_samples=2)
    for _ in range(2): split.update()
    checkpoint = copy.deepcopy(split.checkpoint({'binding':'same'}))
    resumed = r0.R0Runner(Tiny(), Sampler(), pixel_samples=2)
    with pytest.raises(ValueError): resumed.restore(checkpoint, {'binding':'different'})
    resumed.restore(checkpoint, {'binding':'same'})
    for _ in range(2): resumed.update()
    assert all(torch.equal(full_state[k],v) for k,v in resumed.model.state_dict().items())
    assert resumed.global_step == 4 and resumed.scheduler.state_dict() == full_runner['scheduler']
    assert resumed.sampler.state_dict() == full_runner['sampler']
    assert all(torch.equal(initial_canonical[k],v) for k,v in resumed.model.canonical.state_dict().items())
    assert all(p.grad is None for p in resumed.model.canonical.parameters())
    assert [g['name'] for g in resumed.optimizer.param_groups] == list(r0.OWNERS)
    assert all(g['lr'] == 5e-5 for g in resumed.optimizer.param_groups)
    assert resumed.scheduler.step_size == 5000 and resumed.scheduler.gamma == .5
    for key, state in full_runner['optimizer']['state'].items():
        for field, value in state.items():
            other = resumed.optimizer.state_dict()['state'][key][field]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other


def test_same_code_and_pixel_centers(monkeypatch):
    from types import SimpleNamespace
    calls = []
    code = torch.randn(1,32,requires_grad=True)
    class Model:
        def __init__(self):
            self.forward_deformation = SimpleNamespace(coordinates=SimpleNamespace(world_to_normalized=lambda x: x))
            self.inverse_deformation = object()
        def encode_observation(self, observation, *, sample):
            calls.append(('encode', sample))
            return {'code':code,'mu':code,'logvar':code}
        def render_with_encoded(self, observation, uv, encoded):
            calls.append(('render', encoded['code']))
            return {'predicted_intensity': torch.zeros(len(uv))}
    def loss(pred, target, mu, logvar, coords, forward, inverse, z, step):
        calls.append(('loss', z))
        assert coords.shape == (1,2,3) and coords.requires_grad
        return {'total':z.sum()}
    monkeypatch.setattr(r0, 'respiratory_loss', loss)
    obs = SimpleNamespace(image=torch.zeros(1,4,4), center_mm=torch.zeros(3), pixel_spacing_mm=torch.ones(2), row_direction=torch.tensor([1.,0,0]), column_direction=torch.tensor([0.,1,0]))
    r0.observation_loss(Model(), obs, torch.tensor([[0,0],[1,1]]), 2)
    assert [c[0] for c in calls] == ['encode','render','loss']
    assert calls[1][1] is code and calls[2][1] is code


def test_accumulation_divides_by_batch(monkeypatch):
    class Obs:
        image = None
    class FixedSampler(Sampler):
        def sample_step(self, observations_per_update): return [Obs() for _ in range(observations_per_update)]
    def simple_loss(model, obs, uv, step):
        return {'total':sum(p.sum() for name in r0.OWNERS for p in getattr(model,name).parameters())}
    monkeypatch.setattr(r0, 'observation_loss', simple_loss)
    monkeypatch.setattr(r0, 'sample_pixels', lambda image,count: None)
    runner = r0.R0Runner(Tiny(), FixedSampler(), observations_per_update=8, pixel_samples=2)
    runner.update()
    for name in r0.OWNERS:
        for param in getattr(runner.model,name).parameters():
            assert torch.equal(param.grad, torch.ones_like(param))
    assert all(int(state['step']) == 1 for state in runner.optimizer.state.values())


def test_seed_after_temporary_initialization():
    set_reproducibility(88)
    _ = Tiny()
    set_reproducibility(6)
    first = Tiny()
    set_reproducibility(42)
    _ = torch.rand(1000)
    set_reproducibility(6)
    second = Tiny()
    assert all(torch.equal(v, second.state_dict()[k]) for k,v in first.state_dict().items())


def test_formal_config_validation():
    from pathlib import Path
    import yaml
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/simple4d_resp_v1.yaml').read_text())
    r0.validate_config(config)
    config['training']['pixel_samples_per_observation'] = 2
    with pytest.raises(ValueError): r0.validate_config(config)
