#!/usr/bin/env python3
"""Capture machine/environment provenance without modifying source data."""
import argparse,importlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path


def run(command):
    try:
        p=subprocess.run(command,text=True,capture_output=True,check=False)
        return {'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except OSError as e: return {'command':command,'error':str(e)}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); out=a.output_dir
    if out.exists() and any(out.iterdir()): p.error('output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    commands={'conda_from_history.yml':['conda','env','export','--from-history'],'conda_full.yml':['conda','env','export'],'pip_freeze.txt':[sys.executable,'-m','pip','freeze']}
    reports={}
    for name,cmd in commands.items():
        result=run(cmd); reports[name]=result; (out/name).write_text(result.get('stdout',''))
    versions={'python':sys.version,'platform':platform.platform(),'executable':sys.executable,'packages':{},'commands':reports}
    for package in ('torch','torchmetrics','numpy','scipy','nibabel','matplotlib','pydicom','tinycudann'):
        try:
            mod=importlib.import_module(package); versions['packages'][package]={'version':getattr(mod,'__version__',None),'module_path':getattr(mod,'__file__',None)}
        except Exception as e: versions['packages'][package]={'error':f'{type(e).__name__}: {e}'}
    try:
        import torch
        versions['torch_cuda']=torch.version.cuda; versions['cuda_available']=torch.cuda.is_available(); versions['gpus']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e: versions['torch_error']=str(e)
    versions['upstream_root_overrides']={k:v for k,v in os.environ.items() if k.startswith('CARDIORESP4D_') and k.endswith('_ROOT')}
    versions['conda_version']=run(['conda','--version']); versions['git_version']=run(['git','--version'])
    (out/'runtime_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    gpu=[run(['nvidia-smi']),run(['nvcc','--version'])]; (out/'gpu_cuda_report.txt').write_text(json.dumps(gpu,indent=2)+'\n')
    git={k:run(['git','-C',str(a.repo_root),*cmd]) for k,cmd in {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short'],'remote':['remote','-v']}.items()}
    (out/'git_identity.json').write_text(json.dumps(git,indent=2)+'\n')
    vendor=a.repo_root/'third_party/simple4d_VENDOR.json'; (out/'vendor_identity.json').write_text(vendor.read_text() if vendor.is_file() else json.dumps({'error':'vendor manifest missing'}))
    print(f'Wrote {out}; inspect errors in runtime_versions.json.')

if __name__=='__main__': main()
