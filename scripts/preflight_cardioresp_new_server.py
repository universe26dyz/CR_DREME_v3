#!/usr/bin/env python3
"""Read-only destination verification; output a machine-readable READY/BLOCKED report."""
import argparse,csv,importlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from cardioresp4d.migration.inventory import digest


def check_inventory(inventory,data_root):
    failures=[]; skipped=[]; root=Path(data_root).absolute(); files={}
    for row in inventory['records']:
        if row['category']!='MUST' or row.get('is_directory'): continue
        if row.get('unstable_during_scan'): failures.append('source file changed during inventory: '+row['source_path'])
        rel=row.get('portable_relative_path')
        path=root/rel if rel is not None else Path(row['source_path'])
        if not path.is_file(): failures.append(f'missing required file: {path}'); continue
        if row.get('size_bytes') is not None and path.stat().st_size!=row['size_bytes']: failures.append(f'size mismatch: {path}')
        if row.get('sha256') and digest(path)!=row['sha256']: failures.append(f'SHA256 mismatch: {path}')
        if not row.get('sha256'): skipped.append(str(path))
        files.setdefault(row['reason'],[]).append(path)
    for ref in inventory.get('references',[]):
        if any(k in ref for k in ('missing_field','missing_token','error')): failures.append('unresolved source inventory reference: '+str(ref))
    for role in ('observation manifest','qc_table','canonical_domain','static reference manifest','corrected LPS pose map','static volume and spacing','static model required by unified evaluator'):
        if role not in files: failures.append('missing required inventory role: '+role)
    return failures,skipped,files



def check_embedded_paths(inventory, data_root):
    """Check actual loader references even when GPU checks cannot run."""
    failures=[]; primary=inventory.get('primary_inputs',{})
    def find(key):
        source=primary.get(key)
        if not source: raise ValueError('missing primary S05 input: '+key)
        row=next((r for r in inventory['records'] if r['source_path']==source),None)
        if not row: raise ValueError('primary input missing from inventory: '+source)
        return Path(data_root)/row['portable_relative_path'] if row.get('portable_relative_path') is not None else Path(source)
    try:
        manifest=find('manifest'); static=json.loads(find('static_manifest').read_text())
        for path in [static['corrected_pose_manifest']['path'],static['outputs']['volume']['path'],static['outputs']['model']['path']]:
            if not Path(path).is_file(): failures.append('embedded static path unavailable: '+path)
        with manifest.open(newline='') as f: rows=list(csv.DictReader(f))
        if not rows: raise ValueError('empty observation manifest')
        if 'source_file_token' in rows[0]:
            sidecar=manifest.with_name(manifest.stem+'.sensitive_runtime_paths.json')
            mapping=json.loads(sidecar.read_text())['token_to_absolute_path']
            targets=[mapping.get(row['source_file_token']) for row in rows]
        else: targets=[row.get('dicom_path') for row in rows]
        missing=[str(path) for path in targets if not path or not Path(path).is_file()]
        if missing: failures.append(f'{len(missing)} DICOM references unavailable (first 20): '+repr(missing[:20]))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        failures.append(f'embedded runtime paths: {type(exc).__name__}: {exc}')
    return failures


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo-root',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--migration-inventory',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--expected-gpu'); p.add_argument('--legacy-s0',action='store_true'); a=p.parse_args()
    if a.output.exists(): p.error('report output already exists')
    output=a.output.resolve(); data_root=a.data_root.resolve()
    if output==data_root or data_root in output.parents: p.error('report must be outside the data root')
    failures=[]; checks={};
    if a.repo_root.resolve()!=ROOT.resolve(): failures.append('repo-root must match the checkout running this script')
    inv=json.loads(a.migration_inventory.read_text()); bad,skipped,files=check_inventory(inv,a.data_root); failures.extend(bad); failures.extend(check_embedded_paths(inv,a.data_root))
    try:
        git=lambda *x:subprocess.check_output(['git','-C',str(a.repo_root),*x],text=True).strip()
        checks['git_head']=git('rev-parse','HEAD'); checks['git_branch']=git('branch','--show-current'); checks['git_dirty']=git('status','--short')
        vendor=json.loads((a.repo_root/'third_party/simple4d_VENDOR.json').read_text()); checks['vendor_commit']=vendor['commit']
        from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
        checks['source_lock']=verify_vendored_source_lock(a.repo_root)
        for name,expected in vendor['key_files'].items():
            path=a.repo_root/'third_party/simple4d'/name
            if not path.is_file() or digest(path)!=expected: failures.append('vendor source hash mismatch: '+str(path))
    except Exception as e: failures.append(f'code verification: {type(e).__name__}: {e}')
    import os
    overrides={k:v for k,v in os.environ.items() if k.startswith('CARDIORESP4D_') and k.endswith('_ROOT')}
    checks['upstream_root_overrides']=overrides
    if overrides: failures.append('external upstream root override(s) must be audited or unset')
    if sys.version_info < (3,9): failures.append('Python >=3.9 required')
    try:
        import torch
        checks.update({'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
        if not torch.cuda.is_available(): failures.append('CUDA unavailable')
        if a.expected_gpu and not any(a.expected_gpu.lower() in name.lower() for name in checks['gpus']): failures.append('expected GPU not visible: '+a.expected_gpu)
        import tinycudann; checks['tinycudann_path']=getattr(tinycudann,'__file__',None)
    except Exception as e: failures.append(f'runtime/TCNN: {type(e).__name__}: {e}')
    if a.legacy_s0:
        try: importlib.import_module('torchmetrics')
        except Exception as e: failures.append(f'torchmetrics for legacy S0: {type(e).__name__}: {e}')
    if not failures:
        try:
            import torch
            from train_source_first import observations_from_manifest
            from cardioresp4d.experiments.simple4d_resp_v1_r0 import sha256,resolve_static_reference_spacing
            from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import load_corrected_pose_map,apply_corrected_static_poses,derive_effective_external_domain
            from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
            primary=inv.get('primary_inputs',{})
            def primary_file(key):
                source=primary.get(key)
                if not source: raise ValueError('missing primary S05 input: '+key)
                row=next((r for r in inv['records'] if r['source_path']==source),None)
                if not row: raise ValueError('primary input missing from inventory: '+source)
                return Path(a.data_root)/row['portable_relative_path'] if row.get('portable_relative_path') is not None else Path(source)
            manifest=primary_file('manifest'); qc=primary_file('qc_table'); domain=primary_file('canonical_domain'); static=json.loads(primary_file('static_manifest').read_text())
            pose=Path(static['corrected_pose_manifest']['path']); volume=Path(static['outputs']['volume']['path']); model_file=Path(static['outputs']['model']['path'])
            for artifact in (pose,volume,model_file):
                if not artifact.is_file(): raise FileNotFoundError(f'embedded path unavailable: {artifact}; restore original prefix or remap copies and revalidate provenance')
            if static.get('source_manifest_sha256')!=sha256(manifest) or static.get('qc_table_sha256')!=sha256(qc) or static['corrected_pose_manifest'].get('sha256')!=sha256(pose) or static['outputs']['volume'].get('sha256')!=sha256(volume) or static['outputs']['model'].get('sha256')!=sha256(model_file): raise ValueError('static manifest embedded provenance hash mismatch')
            spacing=resolve_static_reference_spacing(static,[1.,1.,1.]); poses=load_corrected_pose_map(pose); box_data=json.loads(domain.read_text()); box=derive_effective_external_domain(poses,box_data['world_min_mm'],box_data['world_max_mm'])['model_float32_effective_domain']
            observations,_,_=observations_from_manifest(manifest,qc,torch.device('cuda:0'),normalization_mode='per_series',assign_slice_positions=True); observations=apply_corrected_static_poses(observations,poses)
            views={o.view.upper() for o in observations if o.qc_valid}; checks['valid_views']=sorted(views)
            if views!={'SAX','2CH','4CH'}: raise ValueError('valid views must be SAX, 2CH and 4CH')
            model=Simple4DCoreMultiViewThickSlice(box['lower_lps_mm'],box['upper_lps_mm'],spacing['resolved_reference_spacing_mm'],use_tcnn=True,require_tcnn=True).cuda().eval(); observation=next(o for o in observations if o.qc_valid)
            with torch.no_grad():
                encoded=model.encode_observation(observation,sample=False); prediction=model.render_with_encoded(observation,torch.tensor([[0,0],[1,1]],device='cuda'),encoded)['predicted_intensity']
            if not torch.isfinite(prediction).all(): raise ValueError('nonfinite render')
            checks['tiny_forward_finite']=True; checks['backend']=model.canonical.backend
        except Exception as e: failures.append(f'S05 runtime readiness: {type(e).__name__}: {e}')
    status='READY' if not failures else 'BLOCKED_'+('DATA_OR_RUNTIME')
    report={'status':status,'failures':failures,'checks':checks,'large_files_without_hash':skipped,'note':'Remapped copies can change provenance hashes; old checkpoint resume requires original identity or separately validated migration.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); return 0 if not failures else 1

if __name__=='__main__': sys.exit(main())
