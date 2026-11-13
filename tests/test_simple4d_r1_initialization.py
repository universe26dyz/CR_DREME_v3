import copy
import json
import hashlib
import random
from pathlib import Path
import torch
from torch import nn
import pytest
from cardioresp4d.experiments.simple4d_resp_v1_r1 import initialize_from_r0, R1Runner, R1_OWNERS


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        for name in R1_OWNERS: setattr(self,name,nn.Linear(1,1))


class Sampler:
    def __init__(self): self.r=random.Random(12)
    def sample_step(self,observations_per_update): return [self.r.random() for _ in range(observations_per_update)]
    def state_dict(self): return self.r.getstate()
    def load_state_dict(self,state): self.r.setstate(state)


def test_source_init_fresh_randomness_and_rejection(tmp_path):
    config={'experiment':{'stage':'R0'},'training':{'boundaries':{}}}; provenance={'source':'bound','effective_config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()}; model=Model()
    source=Model(); state=copy.deepcopy(source.state_dict())
    payload={'checkpoint_schema':1,'global_step':2000,'runner_state':{'global_step':2000},'model':state,'effective_config':config,'provenance':provenance}
    path=tmp_path/'source.pt'; torch.save(payload,path); before=path.read_bytes()
    audit=initialize_from_r0(model,payload,provenance,config)
    assert audit['all_tensors_equal'] and all(torch.equal(model.state_dict()[key],value) for key,value in state.items())
    assert all(p.requires_grad for p in model.parameters())
    runner=R1Runner(model,Sampler(),2000); assert not runner.optimizer.state and runner.global_step==0
    r1_random=torch.rand(4); sampled=runner.sampler.sample_step(8)
    other=Model(); payload2=copy.deepcopy(payload); payload2['global_step']=10000; payload2['runner_state']['global_step']=10000
    initialize_from_r0(other,payload2,provenance,config); runner2=R1Runner(other,Sampler(),10000)
    assert torch.equal(r1_random,torch.rand(4)) and sampled==runner2.sampler.sample_step(8)
    invalid=copy.deepcopy(payload); invalid['global_step']=5000; invalid['runner_state']['global_step']=5000
    with pytest.raises(ValueError): initialize_from_r0(Model(),invalid,provenance,config)
    with pytest.raises(ValueError): initialize_from_r0(Model(),payload,{'source':'different'},config)
    assert path.read_bytes()==before


def test_r1_resume_provenance_rejections():
    runner=R1Runner(Model(),Sampler(),2000); data=copy.deepcopy(runner.checkpoint({'source_sha':'a','config':'x','inputs':'m'}))
    for provenance in ({'source_sha':'b','config':'x','inputs':'m'},{'source_sha':'a','config':'y','inputs':'m'},{'source_sha':'a','config':'x','inputs':'other'}):
        with pytest.raises(ValueError): runner.restore(data,provenance)
    with pytest.raises(ValueError): R1Runner(Model(),Sampler(),10000).restore(data,{'source_sha':'a','config':'x','inputs':'m'})
    with pytest.raises(ValueError): runner.restore({'checkpoint_schema':1},{'source_sha':'a','config':'x','inputs':'m'})
