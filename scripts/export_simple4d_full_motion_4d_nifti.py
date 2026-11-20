#!/usr/bin/env python3
"""Export matched observation-conditioned 4D NIfTI volumes for S05PROP and R2."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from train_source_first import observations_from_manifest
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
from cardioresp4d.experiments.simple4d_disentangle_freq import DualLatentThickSlice
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map, apply_corrected_static_poses
from cardioresp4d.experiments.simple4d_resp_v1_identity_render import render_volume
from cardioresp4d.experiments.simple4d_resp_v1_anatomy import derive_anatomy_crop
from cardioresp4d.training.runtime_state import is_hard_invalid_reason


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s05prop-run', type=Path, required=True)
    parser.add_argument('--r2-run', type=Path, required=True)
    parser.add_argument('--view', default='SAX')
    parser.add_argument('--slice-id', default='SAX_s018')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--chunk-size', type=int, default=65536)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'output directory must be empty: {args.output_dir}')

    runs = [('S05PROP', args.s05prop_run, 10000, Simple4DCoreMultiViewThickSlice),
            ('R2', args.r2_run, 10000, DualLatentThickSlice)]
    loaded = []
    for label, run, step, model_type in runs:
        checkpoint_path = run / f'checkpoint_step_{step}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint.get('runner', {}).get('global_step') != step:
            raise ValueError(f'{label} checkpoint is not step {step}')
        provenance = checkpoint['provenance']
        bounds = provenance['effective_domain']
        spacing = provenance['spacing']['resolved_reference_spacing_mm']
        model = model_type(bounds['lower_lps_mm'], bounds['upper_lps_mm'], spacing,
                           use_tcnn=provenance['backend'] == 'tcnn',
                           require_tcnn=provenance['backend'] == 'tcnn').to(device)
        model.load_state_dict(checkpoint['runner']['model'], strict=True)
        model.eval()
        bindings = json.loads((run / 'experiment_inputs.json').read_text())['paths']
        loaded.append((label, checkpoint_path, checkpoint, model, bindings))

    first = loaded[0][3].forward_deformation.coordinates
    lower = first.lower_world_mm.detach().cpu().numpy()
    upper = lower + first.extent_mm.detach().cpu().numpy()
    spacing = first.reference_spacing_mm.detach().cpu().numpy()
    second = loaded[1][3].forward_deformation.coordinates
    if not (np.array_equal(lower, second.lower_world_mm.detach().cpu().numpy()) and
            np.array_equal(upper, (second.lower_world_mm + second.extent_mm).detach().cpu().numpy()) and
            np.array_equal(spacing, second.reference_spacing_mm.detach().cpu().numpy())):
        raise ValueError('S05PROP and R2 physical grids differ')

    bindings = loaded[0][4]
    static_manifest = json.loads(Path(bindings['static_manifest']).read_text())
    static_volume = Path(static_manifest['outputs']['volume']['path'])
    crop, _, _ = derive_anatomy_crop(static_volume, lower, upper, spacing,
        source_sha256=loaded[0][2]['provenance']['static_volume'])
    shape = tuple(crop['shape'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'crop_manifest.json').write_text(json.dumps(crop, indent=2) + '\n')
    metadata = {'view': args.view, 'slice_id': args.slice_id, 'checkpoint_step': 10000,
        'frame_policy': 'each input observation is encoded directly with posterior mean; V1/full-motion; not withheld-frame prediction',
        'volume_policy': 'full canonical intensity volume evaluated under the observation-conditioned deformation',
        'axis_convention': 'array xyz in external patient LPS; NIfTI affine maps voxel indices to RAS',
        'crop': crop, 'models': {}}

    for label, checkpoint_path, checkpoint, model, paths in loaded:
        observations, _, _ = observations_from_manifest(Path(paths['manifest']), Path(paths['qc_table']), device,
            normalization_mode='per_series', assign_slice_positions=True)
        pose_path = Path(static_manifest['corrected_pose_manifest']['path'])
        observations = apply_corrected_static_poses(observations, load_corrected_pose_map(pose_path))
        sequence = sorted((o for o in observations if o.qc_valid and not is_hard_invalid_reason(o.qc_reason)
            and o.view.upper() == args.view.upper() and str(o.slice_id) == args.slice_id),
            key=lambda o: (o.timestamp_s, o.dynamic_frame_id))
        if len(sequence) != 50 or any(b.timestamp_s <= a.timestamp_s for a, b in zip(sequence, sequence[1:])):
            raise ValueError(f'{label} expected 50 strictly ordered valid frames at {args.view}/{args.slice_id}; found {len(sequence)}')
        nifti_path = args.output_dir / f'{label}_step_10000_{args.view}_{args.slice_id}_full_motion_4d.nii.gz'
        temp_path = args.output_dir / f'.{label}_step_10000_4d.dat'
        volume_4d = np.memmap(temp_path, mode='w+', dtype=np.float32, shape=(*shape, len(sequence)))
        with torch.inference_mode():
            for index, observation in enumerate(sequence):
                code = model.encode_observation(observation, sample=False)['mu']
                volume_4d[..., index] = render_volume(model, code, crop['lower_lps_mm'],
                    crop['spacing_mm'], shape, args.chunk_size)
                del code
        volume_4d.flush()
        lps_affine = np.eye(4, dtype=float)
        lps_affine[:3, :3] = np.diag(crop['spacing_mm'])
        lps_affine[:3, 3] = crop['lower_lps_mm']
        lps_to_ras = np.diag([-1., -1., 1., 1.])
        image = nib.Nifti1Image(volume_4d, lps_to_ras @ lps_affine)
        intervals = np.diff([o.timestamp_s for o in sequence])
        image.header.set_zooms((*crop['spacing_mm'], float(np.median(intervals))))
        image.header.set_xyzt_units('mm', 'sec')
        image.set_data_dtype(np.float32)
        nib.save(image, str(nifti_path))
        del image, volume_4d
        temp_path.unlink()
        saved = nib.load(str(nifti_path))
        if saved.shape != (*shape, 50) or not np.isfinite(np.asarray(saved.dataobj[..., 0])).all():
            raise RuntimeError(f'{label} NIfTI validation failed')
        metadata['models'][label] = {'checkpoint': str(checkpoint_path),
            'checkpoint_sha256': sha256(checkpoint_path), 'nifti': str(nifti_path),
            'nifti_sha256': sha256(nifti_path), 'shape_xyz_t': list(saved.shape),
            'voxel_spacing_mm': list(crop['spacing_mm']), 'frame_count': len(sequence),
            'frame_ids': [int(o.dynamic_frame_id) for o in sequence],
            'timestamps_s': [float(o.timestamp_s) for o in sequence],
            'median_frame_interval_s': float(np.median(intervals)),
            'nifti_size_bytes': nifti_path.stat().st_size}
        del saved, model
        if device.type == 'cuda': torch.cuda.empty_cache()

    metadata_path = args.output_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_dir': str(args.output_dir), 'crop_shape': shape,
        'frame_count': 50, 'niftis': [v['nifti'] for v in metadata['models'].values()]}, indent=2))


if __name__ == '__main__':
    main()
