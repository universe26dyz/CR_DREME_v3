"""SIMPLE-4D R0 protocol, updates, checkpointing and deterministic diagnostics."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shlex

import nibabel as nib
import numpy as np
import torch
from torch import nn
from third_party.simple4d.simple4d.losses import jacobian_matrix
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import deterministic_diagnostic_cohort, diagnostic_rng_context
from cardioresp4d.simple4d.losses import respiratory_loss
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.runtime_state import capture_rng_state, restore_rng_state

OWNERS = ('respiratory_encoder', 'forward_deformation', 'inverse_deformation')
BOUNDARIES = {'r0_smoke': 200, 'r0_diagnostic': 2000, 'r0_mid': 5000, 'r0_formal': 10000}
LEGACY_BOUNDARIES = {'r0_smoke': 200, 'r0_diagnostic': 2000}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_config(config):
    required = {
        'experiment': {'name':'simple4d_resp_v1', 'respiratory_only':True, 'stage':'R0'},
        'data': {'normalization':'per_series'},
        'reproducibility': {'model_seed':6, 'sampler_seed':12},
        'training': {'observations_per_update':8, 'pixel_samples_per_observation':10000, 'boundaries':BOUNDARIES, 'optimizer':{'type':'adam','encoder_lr':5e-5,'forward_deformation_lr':5e-5,'inverse_deformation_lr':5e-5}, 'scheduler':{'type':'step','step_size':5000,'gamma':0.5}},
        'encoder': {'source':'vendored_simple4d','type':'MotionCodeVAE','code_dim':32,'base_channels':32,'position_embedding_dim':32,'view_embedding_dim':8,'dropout':0.1},
        'deformation': {'source':'vendored_simple4d','type':'DeformCodeAdaptive','hidden_dims':[256]*4,'omega':7,'residual':False,'bidirectional':True,'expected_reference_spacing_mm':[1.,1.,1.]},
        'loss': {'photometric':'l1','kl_weight':1e-3,'jacobian_weight':1e-4,'magnitude_weight':1e-2,'cycle_weight':0.1,'regularizer_ramp_slope':0.2,'regularizer_ramp_max':10.},
        'canonical': {'source':'static_nesvor','r0_frozen':True,'bootstrap_config':'respiratory_s2vdreme_reference_v1.yaml'},
        'renderer': {'type':'existing_nesvor_psf','psf_samples':8},
    }
    if any(config.get(section) != expected for section, expected in required.items()) or set(config.get('disabled', [])) != {'reference_gauge','score_gain','respiratory_mbc','cardiac','uncertainty','groupwise'}:
        raise ValueError('configuration does not match fixed SIMPLE-4D R0 protocol')


def resolve_static_reference_spacing(static_manifest, expected_spacing_mm):
    command = static_manifest['nesvor']['command']
    command = shlex.split(command) if isinstance(command, str) else list(command)
    indices = [i for i, token in enumerate(command) if token == '--output-resolution']
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError('static command must contain one --output-resolution')
    resolution = float(command[indices[0] + 1])
    volume_path = Path(static_manifest['outputs']['volume']['path'])
    spacing = np.asarray(nib.load(str(volume_path)).header.get_zooms()[:3], dtype=float)
    expected = np.asarray(expected_spacing_mm, dtype=float)
    if not math.isfinite(resolution) or resolution <= 0 or spacing.shape != (3,) or expected.shape != (3,) or not np.all(np.isfinite(spacing)) or not np.all(np.isfinite(expected)) or not np.allclose(spacing, resolution, atol=1e-5, rtol=0) or not np.allclose(spacing, expected, atol=1e-5, rtol=0):
        raise ValueError('static manifest, NIfTI and expected spacing disagree')
    return {'reference_spacing_policy':'static_manifest_output_resolution_crosschecked_with_static_nifti_header', 'manifest_output_resolution_mm':resolution, 'static_nifti_spacing_mm':spacing.tolist(), 'resolved_reference_spacing_mm':spacing.tolist()}


def sample_pixels(image, count):
    h, w = image.shape[-2:]
    linear = torch.randint(h*w, (count,), device=image.device)
    return torch.stack((linear % w, torch.div(linear, w, rounding_mode='floor')), -1)


def observation_loss(model, observation, pixel_uv, step):
    encoded = model.encode_observation(observation, sample=True)
    render = model.render_with_encoded(observation, pixel_uv, encoded)
    uv = pixel_uv.long()
    target = observation.image[0, uv[:, 1], uv[:, 0]]
    world = SourceFirstDynamicModel._pixel_world(observation, pixel_uv.to(observation.image)).unsqueeze(0)
    coords = model.forward_deformation.coordinates.world_to_normalized(world).detach().clone().requires_grad_(True)
    return respiratory_loss(render['predicted_intensity'], target, encoded['mu'], encoded['logvar'], coords, model.forward_deformation, model.inverse_deformation, encoded['code'], step)


def make_optimizer(model):
    model.freeze_r0()
    groups = [{'name': name, 'params': list(getattr(model, name).parameters()), 'lr':5e-5} for name in OWNERS]
    return torch.optim.Adam(groups), torch.optim.lr_scheduler.StepLR  # scheduler is constructed by runner


def grad_stats(module):
    gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('nonfinite gradient')
    sq = sum(float(g.double().square().sum()) for g in gradients)
    count = sum(g.numel() for g in gradients)
    return math.sqrt(sq), math.sqrt(sq/count) if count else 0.


def parameter_norm(module):
    return math.sqrt(sum(float(p.detach().double().square().sum()) for p in module.parameters()))


class R0Runner:
    def __init__(self, model, sampler, *, observations_per_update=8, pixel_samples=10000):
        self.model, self.sampler = model, sampler
        self.observations_per_update, self.pixel_samples = observations_per_update, pixel_samples
        self.optimizer, scheduler = make_optimizer(model)
        self.scheduler = scheduler(self.optimizer, step_size=5000, gamma=0.5)
        self.global_step = 0
        self.metrics = []

    def update(self):
        self.optimizer.zero_grad(set_to_none=True)
        selected = self.sampler.sample_step(observations_per_update=self.observations_per_update)
        if len(selected) != self.observations_per_update:
            raise RuntimeError('sampler returned incorrect observation count')
        sums = {}
        for obs in selected:
            loss = observation_loss(self.model, obs, sample_pixels(obs.image, self.pixel_samples), self.global_step)
            if not torch.isfinite(loss['total']):
                raise RuntimeError('nonfinite training loss')
            (loss['total'] / len(selected)).backward()
            for name, value in loss.items():
                sums[name] = sums.get(name, 0.) + float(value.detach()) / len(selected)
        if any(p.grad is not None for p in self.model.canonical.parameters()):
            raise RuntimeError('frozen canonical received gradient')
        stats = [grad_stats(getattr(self.model, name)) for name in OWNERS]
        if not any(norm > 0 for norm, _ in stats):
            raise RuntimeError('no nonzero trainable gradient')
        row = {'global_step': self.global_step + 1, **sums}
        for label, (norm, rms) in zip(('encoder','forward_deform','inverse_deform'), stats):
            row[label + '_grad_norm'], row[label + '_grad_rms'] = norm, rms
        for label, group in zip(('encoder','forward','inverse'), self.optimizer.param_groups):
            row['lr_' + label] = group['lr']
        self.optimizer.step(); self.scheduler.step(); self.global_step += 1
        self.metrics.append(row)
        return row

    def state_dict(self):
        return {'global_step':self.global_step, 'optimizer':self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict(), 'sampler':self.sampler.state_dict(), 'metrics':copy.deepcopy(self.metrics)}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler']); self.sampler.load_state_dict(state['sampler']); self.global_step = state['global_step']; self.metrics = copy.deepcopy(state['metrics'])

    def checkpoint(self, provenance, **metadata):
        return {'checkpoint_schema':1, 'global_step':self.global_step, 'model':self.model.state_dict(), 'runner_state':self.state_dict(), 'rng_state':capture_rng_state(), 'provenance':provenance, **metadata}

    def restore(self, checkpoint, provenance, *, effective_config=None):
        if checkpoint.get('checkpoint_schema') != 1 or not compatible_resume_provenance(checkpoint, provenance, effective_config) or checkpoint.get('global_step') != checkpoint.get('runner_state', {}).get('global_step'):
            raise ValueError('checkpoint provenance mismatch')
        self.model.load_state_dict(checkpoint['model'], strict=True); self.load_state_dict(checkpoint['runner_state']); restore_rng_state(checkpoint['rng_state'])


def compatible_resume_provenance(checkpoint, provenance, effective_config):
    """Accept the exact earlier protocol only when the two new boundaries are the sole change."""
    saved = checkpoint.get('provenance')
    if saved == provenance:
        return True
    if effective_config is None or not isinstance(saved, dict):
        return False
    legacy = copy.deepcopy(effective_config)
    legacy['training']['boundaries'] = LEGACY_BOUNDARIES.copy()
    legacy_hash = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    expected = {**provenance, 'effective_config_sha256':legacy_hash}
    return saved == expected and checkpoint.get('effective_config') == legacy


def boundary_progression(diagnostics):
    """Summarize changes between adjacent available boundary diagnostics."""
    keys = sorted(diagnostics, key=lambda key: int(key.removeprefix('step_')))
    result = {}
    for previous, current in zip(keys, keys[1:]):
        a, b = diagnostics[previous], diagnostics[current]
        def fold(field):
            return b[field] / a[field] if a[field] is not None and a[field] != 0 and b[field] is not None else None
        def change(field):
            return b[field] - a[field] if a[field] is not None and b[field] is not None else None
        baseline = b['static_baseline_sampled_mse_full']
        result[f'{previous}_to_{current}'] = {
            'latent_dynamic_fold':fold('latent_mu_dynamic_rms'),
            'dvf_dynamic_fold':fold('respiratory_dvf_dynamic_rms_mm'),
            'static_dynamic_ratio_change':change('respiratory_dvf_static_dynamic_ratio'),
            'PC1_change':change('latent_dynamic_pc1_explained_variance_ratio'),
            'motion_vs_static_mse_improvement_percent':100 * (baseline - b['diagnostic_sampled_reconstruction_mse_full']) / baseline if baseline else None,
            'cycle_error_change':change('cycle_error_mean_mm'),
            'jacobian_p01_change':change('jacobian_det_p01'),
        }
    return result


def decompose(groups, prefix):
    means = [x.mean(dim=0, keepdim=True) for x in groups]
    all_values = torch.cat(groups, 0).double()
    static = torch.cat([m.expand_as(x) for m, x in zip(means, groups)], 0).double()
    dynamic = all_values - static
    rms = lambda x: float(x.square().mean().sqrt())
    total, stat, dyn = rms(all_values), rms(static), rms(dynamic)
    return {prefix+'_total_rms':total, prefix+'_static_rms':stat, prefix+'_dynamic_rms':dyn, prefix+'_static_dynamic_ratio': stat/dyn if dyn > 1e-12 else None}, dynamic


def dynamic_pca(centered):
    matrix = centered.reshape(centered.shape[0], -1).double()
    eigen = torch.linalg.eigvalsh(matrix.T @ matrix).clamp_min(0).flip(0)
    if float(eigen.sum()) <= 1e-20:
        return {'latent_dynamic_pc1_explained_variance_ratio':0., 'latent_dynamic_pc2_explained_variance_ratio':0., 'latent_dynamic_rank_95pct':0}
    ratios = eigen / eigen.sum()
    rank = int(torch.searchsorted(ratios.cumsum(0).contiguous(), ratios.new_tensor(.95)).item()) + 1
    return {'latent_dynamic_pc1_explained_variance_ratio':float(ratios[0]), 'latent_dynamic_pc2_explained_variance_ratio':float(ratios[1]) if len(ratios)>1 else 0., 'latent_dynamic_rank_95pct':rank}


def diagnostic_metrics(model, sampler, *, pixel_count=256):
    cohort = deterministic_diagnostic_cohort(sampler)
    if not cohort:
        raise ValueError('diagnostic cohort is empty')
    coords = model.forward_deformation.coordinates
    axes = [torch.linspace(float(lo), float(lo + extent), 4, device=coords.lower_world_mm.device) for lo, extent in zip(coords.lower_world_mm, coords.extent_mm)]
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(1, -1, 3)
    mus, dvfs, logvars, determinants, cycles = [], [], [], [], []
    errors = {}
    with diagnostic_rng_context():
        for view, _, frames in cohort:
            location_mus, location_dvfs = [], []
            for index, obs in enumerate(frames):
                with torch.no_grad():
                    encoded = model.encode_observation(obs, sample=False)
                    code = encoded['mu']
                    location_mus.append(code.squeeze(0).cpu())
                    logvars.append(encoded['logvar'].squeeze(0).cpu())
                    field = model.forward_deformation(grid, code)['displacement_mm']
                    location_dvfs.append(field.squeeze(0).cpu())
                    h, w = obs.image.shape[-2:]
                    linear = torch.linspace(0, h*w-1, pixel_count, device=obs.image.device).round().long()
                    uv = torch.stack((linear % w, torch.div(linear, w, rounding_mode='floor')), -1)
                    centers = SourceFirstDynamicModel._pixel_world(obs, uv.to(obs.image))
                    n = len(uv)
                    resolution = obs.image.new_tensor([obs.pixel_spacing_mm[1], obs.pixel_spacing_mm[0], obs.slice_thickness_mm]).expand(n, 3)
                    samples = model.psf.sample(centers, resolution, row_direction=obs.row_direction.to(obs.image).expand(n,-1), column_direction=obs.column_direction.to(obs.image).expand(n,-1), normal=obs.normal.to(obs.image).expand(n,-1))
                    reference = model.forward_deformation(samples, code.expand(n,-1))['reference_points_mm']
                    dynamic = model.canonical(reference, return_features=True)[0].mean(-1)
                    static = model.canonical(samples, return_features=True)[0].mean(-1)
                    target = obs.image[0].flatten()[linear]
                    for kind, prediction in (('diagnostic_sampled_reconstruction', dynamic), ('static_baseline_sampled', static)):
                        err = prediction - target
                        key = (kind, view)
                        accum = errors.setdefault(key, [0., 0., 0])
                        accum[0] += float(err.square().sum()); accum[1] += float(err.abs().sum()); accum[2] += err.numel()
                if index == len(frames)//2:
                    with torch.enable_grad():
                        x = coords.world_to_normalized(grid).detach().clone().requires_grad_(True)
                        _, displacement = model.forward_deformation.forward_normalized(x, code.detach())
                        det = torch.linalg.det(jacobian_matrix(x, displacement)).detach().cpu().flatten()
                        determinants.append(det)
                        xc = (x + displacement).detach().clone().requires_grad_(True)
                        _, inverse = model.inverse_deformation.forward_normalized(xc, code.detach())
                        delta = coords.normalized_deformation_to_world_mm(xc + inverse - x)
                        cycles.append(torch.linalg.vector_norm(delta, dim=-1).detach().cpu().flatten())
            mus.append(torch.stack(location_mus)); dvfs.append(torch.stack(location_dvfs))
    result, centered = decompose(mus, 'latent_mu')
    result.update(dynamic_pca(centered))
    d, _ = decompose(dvfs, 'respiratory_dvf')
    for key, value in d.items():
        result[key.replace('_rms', '_rms_mm')] = value
    result['respiratory_dvf_max_abs_mm'] = float(torch.cat(dvfs).abs().max())
    logvar = torch.stack(logvars).double(); std = torch.exp(0.5 * logvar)
    result.update({'logvar_mean':float(logvar.mean()), 'logvar_std':float(logvar.std(unbiased=False)), 'logvar_min':float(logvar.min()), 'logvar_max':float(logvar.max()), 'posterior_std_mean':float(std.mean()), 'posterior_std_min':float(std.min()), 'posterior_std_max':float(std.max()), 'kl_mean':float(-.5*(1+logvar-torch.cat(mus).double().square()-logvar.exp()).mean())})
    det = torch.cat(determinants).double(); cycle = torch.cat(cycles).double()
    result.update({'jacobian_det_mean':float(det.mean()), 'jacobian_det_median':float(torch.quantile(det, .5)), 'jacobian_det_p01':float(torch.quantile(det, .01)), 'jacobian_det_min':float(det.min()), 'jacobian_det_nonpositive_fraction':float((det <= 0).double().mean()), 'cycle_error_mean_mm':float(cycle.mean()), 'cycle_error_p95_mm':float(torch.quantile(cycle, .95)), 'cycle_error_max_mm':float(cycle.max()), 'diagnostic_pixels_per_frame':pixel_count, 'diagnostic_frame_count':sum(len(frames) for _,_,frames in cohort), 'diagnostic_location_count':len(cohort)})
    for kind in ('diagnostic_sampled_reconstruction','static_baseline_sampled'):
        entries = [(view, values) for (name, view), values in errors.items() if name == kind]
        for view, (squared, absolute, count) in entries + [('full', [sum(v[i] for _, v in entries) for i in range(3)])]:
            result[f'{kind}_mse_{view}'] = squared/count
            result[f'{kind}_mae_{view}'] = absolute/count
    result.update({name+'_parameter_norm':parameter_norm(getattr(model, owner)) for name, owner in [('encoder','respiratory_encoder'),('forward_deformation','forward_deformation'),('inverse_deformation','inverse_deformation'),('canonical','canonical')]})
    return result


def summarize_metrics(rows):
    if not rows:
        return {}, {}
    names = [name for name in rows[0] if name not in {'global_step'} and not name.startswith('lr_')]
    summaries = {name:{'first':rows[0][name], 'last':rows[-1][name], 'mean':sum(row[name] for row in rows)/len(rows)} for name in names}
    return ({k:v for k,v in summaries.items() if '_grad_' in k}, {k:v for k,v in summaries.items() if '_grad_' not in k})
