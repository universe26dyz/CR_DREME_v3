import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from cardioresp4d.experiments import simple4d_resp_v1_r1 as r1

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r1_cli',ROOT/'scripts/train_simple4d_resp_v1_r1.py')
cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        for name in r1.R1_OWNERS: setattr(self,name,nn.Linear(1,1))


class Sampler:
    def __init__(self,obs,seed): self.r=random.Random(seed)
    def sample_step(self,observations_per_update): return [SimpleNamespace(image=None,value=self.r.random()) for _ in range(observations_per_update)]
    def state_dict(self): return self.r.getstate()
    def load_state_dict(self,state): self.r.setstate(state)


def test_config_validation():
    r0=cli._load_config(ROOT/'configs/simple4d_resp_v1.yaml'); config=cli._load_config(ROOT/'configs/simple4d_resp_v1_r1.yaml')
    cli.validate_r1_config(config,r0)
    config['training']['optimizer']['canonical_lr']=1e-3
    with pytest.raises(ValueError): cli.validate_r1_config(config,r0)


def test_synthetic_launcher_resume(tmp_path,monkeypatch):
    r0=cli._load_config(ROOT/'configs/simple4d_resp_v1.yaml'); hash_config=hashlib.sha256(json.dumps(r0,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    provenance={'effective_config_sha256':hash_config,'hashes':{'inputs':'fixed'}}
    source_model=Model(); source_state=copy.deepcopy(source_model.state_dict())
    source_dir=tmp_path/'r0'; source_dir.mkdir(); source=source_dir/'checkpoint_step_2000.pt'
    torch.save({'checkpoint_schema':1,'global_step':2000,'runner_state':{'global_step':2000},'model':source_state,'effective_config':r0,'provenance':provenance,'diagnostics':{'step_2000':{'static_baseline_sampled_mse_full':.2,'static_baseline_sampled_mae_full':.3}}},source)
    before=source.read_bytes()
    def load(args):
        model=Model(); model.load_state_dict(source_state)
        return r0,provenance,model,(('SAX','x',(object(),)),)
    monkeypatch.setattr(cli,'load_inputs',load); monkeypatch.setattr(cli,'ViewLocationBalancedSampler',Sampler)
    monkeypatch.setattr(cli,'validate_r1_config',lambda *args:None)
    monkeypatch.setattr(cli,'R1_BOUNDARIES',{'r1_short':1,'r1_diagnostic':2})
    monkeypatch.setattr(cli,'diagnostic_metrics',lambda model,sampler:{'static_baseline_sampled_mse_full':.2,'diagnostic_sampled_reconstruction_mse_full':.1})
    monkeypatch.setattr(r1,'sample_pixels',lambda *args:None)
    def loss(model,obs,uv,step):
        value=torch.tensor([[obs.value]])+torch.rand(1,1)
        for name in r1.R1_OWNERS: value=getattr(model,name)(value)
        return {'total':value.square().mean()}
    monkeypatch.setattr(r1,'observation_loss',loss)
    output=tmp_path/'r1'; args=['train','--manifest','m','--qc-table','q','--canonical-domain','d','--static-manifest','s','--output-dir',str(output),'--source-r0-checkpoint',str(source),'--device','cpu','--stop-after']
    monkeypatch.setattr(sys,'argv',args+['r1_short']); cli.main()
    checkpoint=output/'checkpoint_r1_step_1.pt'; assert checkpoint.exists()
    report=json.loads((output/'training_report.json').read_text()); assert report['r1_joint_step']==1 and report['source_r0_step']==2000
    assert 'current_r1_canonical_no_motion_sampled_mse_full' in report['diagnostics']['r1_step_1']
    monkeypatch.setattr(sys,'argv',args+['r1_diagnostic','--resume',str(checkpoint)]); cli.main()
    assert (output/'checkpoint_r1_step_2.pt').exists() and source.read_bytes()==before
    assert json.loads((output/'training_report.json').read_text())['r1_joint_step']==2
