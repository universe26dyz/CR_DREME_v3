import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from cardioresp4d.experiments.simple4d_core_mv import validate_s1_resume


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('s1_extension_launcher',ROOT/'scripts/train_simple4d_core_mv_v1.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_target_cli_and_unchanged_config():
    parser=launcher.build_parser()
    required=['--manifest','m','--qc-table','q','--canonical-domain','d','--static-manifest','s','--output-dir','o']
    cfg_path=ROOT/'configs/simple4d_core_mv_v1.yaml'
    cfg=yaml.safe_load(cfg_path.read_text())
    for name,step in [('smoke',200),('diagnostic',2000),('mid',5000),('formal',10000)]:
        assert cfg['boundaries'][parser.parse_args(required+['--stop-after',name]).stop_after]==step
    assert parser.parse_args(required+['--target-step','20000']).target_step==20000
    for invalid in (['--target-step','20001'],['--target-step','0'],['--stop-after','formal','--target-step','20000']):
        with pytest.raises(SystemExit): parser.parse_args(required+invalid)
    assert hashlib.sha256(cfg_path.read_bytes()).hexdigest()=='bc31cd0faf0a20ec40a084503e510ef2cff97b1a1a8973ab27231493762bd936'


class TinyRunner:
    def __init__(self):
        self.param=torch.nn.Parameter(torch.tensor(1.))
        self.optimizer=torch.optim.Adam([{'params':[self.param],'name':'hash','lr':.01}])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=5000,gamma=.5)
        self.global_step=0; self.metrics=[]

    def update(self):
        row={'global_step':self.global_step+1,'lr_hash':self.optimizer.param_groups[0]['lr']}
        self.param.grad=torch.ones_like(self.param); self.optimizer.step(); self.scheduler.step()
        self.global_step+=1; self.metrics.append(row)

    def state_dict(self):
        return {'global_step':self.global_step,'model':self.param.detach().clone(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'metrics':self.metrics}

    def load_state_dict(self,state):
        self.global_step=state['global_step']; self.param.data.copy_(state['model']); self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler']); self.metrics=state['metrics']


def test_continuation_checkpoints_summary_and_scheduler(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'diagnostic_metrics',lambda model,sampler:{'diagnostic_sampled_reconstruction_mse_full':.2,'static_baseline_sampled_mse_full':.3,'jacobian_det_p01':.9})
    cfg=yaml.safe_load((ROOT/'configs/simple4d_core_mv_v1.yaml').read_text())
    config_hash=hashlib.sha256((ROOT/'configs/simple4d_core_mv_v1.yaml').read_bytes()).hexdigest()
    provenance={'config':config_hash,'backend':'tcnn'}
    runner=TinyRunner()
    for _ in range(2000): runner.update()
    original={'schema':'simple4d_s1_v1','provenance':provenance,'runner':runner.state_dict(),'diagnostics':{'step_2000':{'old':1}}}
    path=tmp_path/'checkpoint_step_2000.pt'; torch.save(original,path); original_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    saved=torch.load(path,weights_only=False); validate_s1_resume(saved,provenance)
    resumed=TinyRunner(); resumed.load_state_dict(saved['runner'])
    steps=set(cfg['boundaries'].values())|{15000,20000}
    launcher.train_to_target(resumed,None,None,10000,steps,saved['diagnostics'],tmp_path,provenance,cfg,{},[])
    assert resumed.global_step==10000 and resumed.scheduler.last_epoch==10000
    assert resumed.optimizer.param_groups[0]['lr']==pytest.approx(.01*.25)
    checkpoint=torch.load(tmp_path/'checkpoint_step_10000.pt',weights_only=False)
    validate_s1_resume(checkpoint,provenance)
    again=TinyRunner(); again.load_state_dict(checkpoint['runner'])
    assert again.optimizer.state_dict()['param_groups']==resumed.optimizer.state_dict()['param_groups']
    assert again.scheduler.state_dict()==resumed.scheduler.state_dict()
    launcher.train_to_target(again,None,None,20000,steps,checkpoint['diagnostics'],tmp_path,provenance,cfg,{},[])
    assert again.global_step==20000 and again.scheduler.last_epoch==20000
    assert again.optimizer.param_groups[0]['lr']==pytest.approx(.01*.0625)
    assert all((tmp_path/f'checkpoint_step_{step}.pt').exists() for step in (2000,5000,10000,15000,20000))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original_hash
    with (tmp_path/'s1_extension_summary.csv').open() as f: rows=list(csv.DictReader(f))
    assert [int(row['step']) for row in rows]==[5000,10000,15000,20000]
    assert [float(row['lr_hash_after_scheduler']) for row in rows]==pytest.approx([.005,.0025,.00125,.000625])
    assert rows[0]['current_canonical_no_motion_sampled_mse_full']=='0.3'
    assert 'step_2000' in checkpoint['diagnostics']
    with pytest.raises(ValueError,match='exceeds target'):
        launcher.train_to_target(again,None,None,10000,steps,{},tmp_path,provenance,cfg,{},[])
