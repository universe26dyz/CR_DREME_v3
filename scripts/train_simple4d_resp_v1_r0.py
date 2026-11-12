#!/usr/bin/env python3
"""Train SIMPLE-4D respiratory-only R0 from an admitted static NeSVoR model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from train_respiratory_s2vdreme_reference_v1 import _load_config
from train_respiratory_reference_gauge_exp2a import _build_exact_model
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import apply_corrected_static_poses, derive_effective_external_domain, load_corrected_pose_map
from cardioresp4d.experiments.simple4d_resp_v1_r0 import BOUNDARIES, OWNERS, R0Runner, diagnostic_metrics, resolve_static_reference_spacing, sha256, state_digest, summarize_metrics, validate_config
from cardioresp4d.simple4d.model import Simple4DRespiratoryModel
from cardioresp4d.training.runtime_state import set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','qc-table','canonical-domain','static-manifest','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--experiment-config', type=Path, default=ROOT/'configs/simple4d_resp_v1.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stop-after', required=True, choices=BOUNDARIES)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    config = _load_config(args.experiment_config); validate_config(config)
    static = json.loads(args.static_manifest.read_text())
    paths = {'manifest':args.manifest, 'qc_table':args.qc_table, 'canonical_domain':args.canonical_domain, 'static_manifest':args.static_manifest, 'static_model':Path(static['outputs']['model']['path']), 'static_volume':Path(static['outputs']['volume']['path']), 'corrected_pose_manifest':Path(static['corrected_pose_manifest']['path']), 'simple4d_vendor_manifest':ROOT/'third_party/simple4d_VENDOR.json', 'bootstrap_config':ROOT/'configs'/config['canonical']['bootstrap_config']}
    hashes = {key:sha256(path) for key,path in paths.items()}
    if static.get('source_manifest_sha256') != hashes['manifest'] or static.get('qc_table_sha256') != hashes['qc_table']:
        raise ValueError('static manifest does not bind manifest/QC')
    for section, key, actual in [(static['outputs']['model'], 'model', hashes['static_model']), (static['outputs']['volume'], 'volume', hashes['static_volume']), (static['corrected_pose_manifest'], 'poses', hashes['corrected_pose_manifest'])]:
        if not section.get('sha256') or section['sha256'] != actual:
            raise ValueError(f'static {key} hash missing or mismatched')
    vendor = json.loads(paths['simple4d_vendor_manifest'].read_text())
    if vendor.get('commit') != 'c466b276a438b3e052ad3b44d38f398872d8aeae':
        raise ValueError('unexpected SIMPLE-4D commit')
    for name, digest in vendor['key_files'].items():
        if sha256(ROOT/'third_party/simple4d'/name) != digest:
            raise ValueError('SIMPLE-4D vendor hash mismatch: '+name)
    spacing = resolve_static_reference_spacing(static, config['deformation']['expected_reference_spacing_mm'])
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    provenance = {'hashes':hashes, 'effective_config_sha256':config_sha, 'upstream_commit':vendor['commit'], 'spacing':spacing, 'model_seed':6, 'sampler_seed':12}
    domain = json.loads(args.canonical_domain.read_text())
    poses = load_corrected_pose_map(paths['corrected_pose_manifest'])
    effective = derive_effective_external_domain(poses, domain['world_min_mm'], domain['world_max_mm'])
    box = effective['model_float32_effective_domain']
    model_domain = {**domain, 'world_min_mm':box['lower_lps_mm'], 'world_max_mm':box['upper_lps_mm']}
    observations, normalization, groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode='per_series', assign_slice_positions=True)
    observations = apply_corrected_static_poses(observations, poses)
    static_payload = torch.load(paths['static_model'], map_location='cpu', weights_only=False)
    temporary, backend, internal_bounds = _build_exact_model(_load_config(paths['bootstrap_config']), model_domain, static_payload, observations, device)
    canonical = temporary.canonical
    del temporary, static_payload
    set_reproducibility(6)
    model = Simple4DRespiratoryModel(canonical, box['lower_lps_mm'], box['upper_lps_mm'], spacing['resolved_reference_spacing_mm'], canonical_spacing_mm=spacing['static_nifti_spacing_mm'], psf_samples=8).to(device)
    model.freeze_r0()
    initial = {name+'_initial_state_sha256':state_digest(getattr(model, name)) for name in (*OWNERS, 'canonical')}
    initial['encoder_initial_state_sha256'] = initial.pop('respiratory_encoder_initial_state_sha256')
    initial.update({'static_model_sha256':hashes['static_model'], 'vendored_simple4d_manifest_sha256':hashes['simple4d_vendor_manifest'], 'initialization_policy':'seed 6 after exact static canonical construction'})
    sampler = ViewLocationBalancedSampler(observations, seed=12)
    runner = R0Runner(model, sampler)
    static_initialization = {'mode':'direct_exact_state_with_coordinate_bridge', 'static_model_sha256':hashes['static_model'], 'internal_inr_bounds_mm':internal_bounds.tolist(), 'encoding_backend':backend, 'canonical_frozen':True, 'effective_external_domain':effective}
    diagnostics = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        runner.restore(checkpoint, provenance)
        initial = checkpoint['initialization_audit']; diagnostics = checkpoint['diagnostics']
        if runner.global_step > BOUNDARIES[args.stop_after]:
            raise ValueError('checkpoint is beyond requested boundary')
    else:
        diagnostics['step_0'] = diagnostic_metrics(model, sampler)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bindings = {key:{'path':str(path.resolve()), 'sha256':hashes[key]} for key,path in paths.items()}
    write_json(args.output_dir/'experiment_inputs.json', bindings)
    write_json(args.output_dir/'effective_config.json', config)
    target = BOUNDARIES[args.stop_after]
    while runner.global_step < target:
        runner.update()
        if runner.global_step in BOUNDARIES.values():
            diagnostics['step_'+str(runner.global_step)] = diagnostic_metrics(model, sampler)
            gradient_summary, loss_summary = summarize_metrics(runner.metrics)
            metadata = {'protocol':config, 'effective_config':config, 'initialization_audit':initial, 'diagnostics':diagnostics, 'static_initialization':static_initialization, 'spacing_provenance':spacing, 'input_bindings':bindings, 'normalization_parameters':normalization, 'normalization_groups':groups, 'simple4d_vendor_provenance':model.provenance()}
            payload = runner.checkpoint(provenance, **metadata)
            torch.save(payload, args.output_dir/f'checkpoint_step_{runner.global_step}.pt')
            torch.save(payload, args.output_dir/'checkpoint_last.pt')
            report = {**metadata, 'provenance':provenance, 'training_summary':{'global_step':runner.global_step, 'owners':list(OWNERS), 'canonical_frozen':True}, 'gradient_summary':gradient_summary, 'loss_component_summary':loss_summary}
            write_json(args.output_dir/'training_report.json', report)
            with (args.output_dir/'training_metrics.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(runner.metrics[0])); writer.writeheader(); writer.writerows(runner.metrics)
    print(json.dumps({'global_step':runner.global_step, 'output':str(args.output_dir)}))


if __name__ == '__main__':
    main()
