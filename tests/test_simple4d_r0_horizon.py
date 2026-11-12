import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from cardioresp4d.experiments import simple4d_resp_v1_r0 as r0

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('simple4d_r0_launcher', ROOT/'scripts/train_simple4d_resp_v1_r0.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_boundaries_and_legacy_provenance():
    config = yaml.safe_load((ROOT/'configs/simple4d_resp_v1.yaml').read_text())
    assert r0.BOUNDARIES == {'r0_smoke':200, 'r0_diagnostic':2000, 'r0_mid':5000, 'r0_formal':10000}
    r0.validate_config(config)
    assert config['training']['boundaries'] == r0.BOUNDARIES
    legacy = copy.deepcopy(config); legacy['training']['boundaries'] = r0.LEGACY_BOUNDARIES.copy()
    provenance = {'effective_config_sha256':config_hash(config), 'hashes':{'manifest':'fixed'}}
    old = {'effective_config':legacy, 'provenance':{**provenance, 'effective_config_sha256':config_hash(legacy)}}
    assert r0.compatible_resume_provenance(old, provenance, config)
    altered = copy.deepcopy(old); altered['effective_config']['loss']['kl_weight'] = 2e-3
    assert not r0.compatible_resume_provenance(altered, provenance, config)
    altered = copy.deepcopy(old); altered['provenance']['hashes']['manifest'] = 'other'
    assert not r0.compatible_resume_provenance(altered, provenance, config)
    altered = copy.deepcopy(old); altered['provenance']['effective_config_sha256'] = 'wrong'
    assert not r0.compatible_resume_provenance(altered, provenance, config)


def test_step_lr_boundaries():
    parameter = nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.Adam([parameter],lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=.5)
    assert optimizer.param_groups[0]['lr'] == 5e-5
    for step in range(1, 10001):
        recorded = optimizer.param_groups[0]['lr']
        optimizer.step(); scheduler.step()
        if step == 4999: assert optimizer.param_groups[0]['lr'] == 5e-5
        if step == 5000:
            assert recorded == 5e-5 and optimizer.param_groups[0]['lr'] == 2.5e-5
            saved_opt, saved_scheduler = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
            resumed_opt = torch.optim.Adam([nn.Parameter(torch.tensor(1.))], lr=5e-5)
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_opt, step_size=5000, gamma=.5)
            resumed_opt.load_state_dict(saved_opt); resumed_scheduler.load_state_dict(saved_scheduler)
            assert resumed_scheduler.state_dict() == saved_scheduler and resumed_opt.param_groups[0]['lr'] == 2.5e-5
        if step == 5001: assert recorded == 2.5e-5 and optimizer.param_groups[0]['lr'] == 2.5e-5
        if step == 10000: assert recorded == 2.5e-5 and optimizer.param_groups[0]['lr'] == 1.25e-5
    assert scheduler.last_epoch == 10000


def test_boundary_resume_and_files(tmp_path, monkeypatch):
    class Runner:
        def __init__(self, step): self.global_step = step
        def update(self): self.global_step += 1
    monkeypatch.setattr(launcher, 'BOUNDARIES', {'r0_mid':2001, 'r0_formal':2002})
    old_diagnostics = {'step_0':{'x':0}, 'step_200':{'x':200}, 'step_2000':{'x':2000}}
    runner = Runner(2000)
    def save(): launcher.save_checkpoints(tmp_path, runner.global_step, {'global_step':runner.global_step, 'diagnostics':copy.deepcopy(old_diagnostics)})
    launcher.run_to_boundary(runner, 2001, old_diagnostics, lambda:{'x':runner.global_step}, save)
    checkpoint = torch.load(tmp_path/'checkpoint_step_2001.pt', weights_only=False)
    assert set(checkpoint['diagnostics']) == {'step_0','step_200','step_2000','step_2001'}
    runner = Runner(checkpoint['global_step']); old_diagnostics = checkpoint['diagnostics']
    launcher.run_to_boundary(runner, 2002, old_diagnostics, lambda:{'x':runner.global_step}, save)
    last = torch.load(tmp_path/'checkpoint_last.pt', weights_only=False)
    assert last['global_step'] == 2002 and set(last['diagnostics']) == {'step_0','step_200','step_2000','step_2001','step_2002'}
    with pytest.raises(ValueError): launcher.run_to_boundary(runner, 2001, old_diagnostics, lambda:{}, save)
    for step in (200,2000,5000,10000):
        launcher.save_checkpoints(tmp_path, step, {'global_step':step})
        assert torch.load(tmp_path/f'checkpoint_step_{step}.pt', weights_only=False)['global_step'] == step
    assert torch.load(tmp_path/'checkpoint_last.pt', weights_only=False)['global_step'] == 10000
    runner = Runner(2000); direct = copy.deepcopy(checkpoint['diagnostics'])
    launcher.run_to_boundary(runner, 2002, direct, lambda:{'x':runner.global_step}, lambda:None)
    assert runner.global_step == 2002 and 'step_2001' in direct and 'step_2002' in direct


def test_progression():
    def values(latent, dvf, ratio, pc1, static, motion, cycle, jac):
        return {'latent_mu_dynamic_rms':latent, 'respiratory_dvf_dynamic_rms_mm':dvf, 'respiratory_dvf_static_dynamic_ratio':ratio, 'latent_dynamic_pc1_explained_variance_ratio':pc1, 'static_baseline_sampled_mse_full':static, 'diagnostic_sampled_reconstruction_mse_full':motion, 'cycle_error_mean_mm':cycle, 'jacobian_det_p01':jac}
    out = r0.boundary_progression({'step_2000':values(2,3,1,.5,10,8,.4,.9), 'step_5000':values(4,6,1.5,.6,10,7,.3,.95)})['step_2000_to_step_5000']
    assert out['latent_dynamic_fold'] == 2 and out['dvf_dynamic_fold'] == 2
    assert out['static_dynamic_ratio_change'] == .5
    assert out['PC1_change'] == pytest.approx(.1)
    assert out['motion_vs_static_mse_improvement_percent'] == 30
    assert out['cycle_error_change'] == pytest.approx(-.1) and out['jacobian_p01_change'] == pytest.approx(.05)
    out = r0.boundary_progression({'step_0':values(0,0,None,0,0,0,0,0), 'step_1':values(1,1,1,1,1,0,1,1)})['step_0_to_step_1']
    assert out['latent_dynamic_fold'] is None and out['dvf_dynamic_fold'] is None and out['static_dynamic_ratio_change'] is None


def test_actual_runner_legacy_resume_and_recorded_lr(monkeypatch):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.canonical = nn.Linear(1,1)
            self.respiratory_encoder = nn.Linear(1,1)
            self.forward_deformation = nn.Linear(1,1)
            self.inverse_deformation = nn.Linear(1,1)
        def freeze_r0(self): self.canonical.requires_grad_(False)
    class Obs:
        image = None
    class Sampler:
        def sample_step(self, observations_per_update): return [Obs()] * observations_per_update
        def state_dict(self): return {'sample_state':1}
        def load_state_dict(self, state): assert state == {'sample_state':1}
    monkeypatch.setattr(r0, 'sample_pixels', lambda image, count: None)
    monkeypatch.setattr(r0, 'observation_loss', lambda model, obs, pixels, step: {'total':sum(p.square().sum() for owner in r0.OWNERS for p in getattr(model, owner).parameters())})
    config = yaml.safe_load((ROOT/'configs/simple4d_resp_v1.yaml').read_text())
    legacy = copy.deepcopy(config); legacy['training']['boundaries'] = r0.LEGACY_BOUNDARIES.copy()
    provenance = {'effective_config_sha256':config_hash(config), 'input':'fixed'}
    old_provenance = {**provenance, 'effective_config_sha256':config_hash(legacy)}
    runner = r0.R0Runner(Tiny(), Sampler(), observations_per_update=1, pixel_samples=1)
    runner.global_step = 4999
    runner.scheduler.last_epoch = 4999
    checkpoint = copy.deepcopy(runner.checkpoint(old_provenance, effective_config=legacy, diagnostics={'step_0':{},'step_200':{},'step_2000':{}}))
    resumed = r0.R0Runner(Tiny(), Sampler(), observations_per_update=1, pixel_samples=1)
    resumed.restore(checkpoint, provenance, effective_config=config)
    row = resumed.update()
    assert row['global_step'] == 5000 and row['lr_encoder'] == 5e-5
    assert all(group['lr'] == 2.5e-5 for group in resumed.optimizer.param_groups)
    row = resumed.update()
    assert row['lr_encoder'] == 2.5e-5
    corrupted = copy.deepcopy(checkpoint); corrupted['provenance']['input'] = 'different'
    with pytest.raises(ValueError): resumed.restore(corrupted, provenance, effective_config=config)
