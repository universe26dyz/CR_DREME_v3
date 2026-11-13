import copy
import random
import torch
from torch import nn
from cardioresp4d.experiments import simple4d_resp_v1_r1 as r1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        for name in r1.R1_OWNERS: setattr(self,name,nn.Linear(1,1))


class Sampler:
    def __init__(self): self.r=random.Random(12)
    def sample_step(self,observations_per_update): return [type('Obs',(),{'image':None,'value':self.r.random()})() for _ in range(observations_per_update)]
    def state_dict(self): return self.r.getstate()
    def load_state_dict(self,s): self.r.setstate(s)


def test_r1_ownership_and_resume(monkeypatch):
    def loss(model,obs,uv,step):
        x=torch.tensor([[obs.value]])+torch.rand(1,1)
        for name in r1.R1_OWNERS: x=getattr(model,name)(x)
        return {'total':x.square().mean()}
    monkeypatch.setattr(r1,'observation_loss',loss); monkeypatch.setattr(r1,'sample_pixels',lambda *a:None)
    runner=r1.R1Runner(Model(),Sampler(),10000,observations_per_update=2,pixel_samples=2)
    before=copy.deepcopy(runner.model.canonical.state_dict()); runner.update()
    assert runner.global_step==1 and runner.source_r0_step==10000
    assert any(not torch.equal(before[k],v) for k,v in runner.model.canonical.state_dict().items())
    assert [x['lr'] for x in runner.optimizer.param_groups]==[2e-5,5e-5,5e-5,5e-5]
    saved=copy.deepcopy(runner.checkpoint({'source':'hash'})); runner.update(); expected=copy.deepcopy(runner.model.state_dict())
    resumed=r1.R1Runner(Model(),Sampler(),10000,observations_per_update=2,pixel_samples=2); resumed.restore(saved,{'source':'hash'}); resumed.update()
    assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in expected.items())
    assert resumed.canonical_drift()['canonical_parameter_delta_norm']>0
