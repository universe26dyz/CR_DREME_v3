#!/usr/bin/env python3
"""Discover migration dependencies from S05 experiment_inputs without modifying inputs."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cardioresp4d.migration.inventory import Inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path('/data/dengyz/code/CR_DREME_v3'))
    p.add_argument('--data-root',type=Path,default=Path('/data/dengyz/dataset/CR_DREME_v3'))
    p.add_argument('--s05-run',type=Path); p.add_argument('--r0-run',type=Path); p.add_argument('--s05sax-run',type=Path); p.add_argument('--s05bm-run',type=Path); p.add_argument('--s1-run',type=Path)
    p.add_argument('--optional-dir',type=Path,action='append',default=[]); p.add_argument('--environment-dir',type=Path); p.add_argument('--output-dir',required=True,type=Path); p.add_argument('--hash-large-files',action='store_true')
    a=p.parse_args(); root=a.data_root.absolute(); out=a.output_dir.absolute()
    if out==root or root in out.parents: p.error('output directory must be outside the data root')
    if out.exists() and any(out.iterdir()): p.error('output directory must be empty')
    inv=Inventory(a.repo_root,root,a.hash_large)
    s05=a.s05_run or root/'simple4d_core_mv_thickslice_v1_smoke'
    inv.run(s05,'S05')
    for name,path in [('S05SAX',a.s05sax_run or root/'simple4d_core_sax_thickslice_v1_smoke'),('S05BM',a.s05bm_run)]:
        if path and (path.exists() or name=='S05BM'): inv.run(path,name)
    for label,run in [('R0',a.r0_run),('S1',a.s1_run)]:
        if run:
            for name in ('checkpoint_step_2000.pt','training_report.json','training_metrics.csv','experiment_inputs.json','effective_config.json','s1_extension_summary.csv'):
                path=run/name
                if path.exists(): inv.add(path,'SHOULD',f'{label} comparison/historical artifact',logical_root='checkpoints')
            if label=='S1':
                for step in (10000,15000,20000):
                    path=run/f'checkpoint_step_{step}.pt'
                    if path.exists(): inv.add(path,'SHOULD','S1 long-training historical checkpoint',logical_root='checkpoints')
    for d in a.optional_dir: inv.add(d,'OPTIONAL','user-selected audit or visualization archive',logical_root='evaluation_audit')
    if a.environment_dir: inv.add(a.environment_dir,'SHOULD','environment capture for reconstruction',logical_root='environment_metadata')
    vendor=a.repo_root/'third_party/simple4d_VENDOR.json'; inv.json(vendor,'SHOULD','vendor code identity',logical_root='other')
    inputs=Path(s05)/'experiment_inputs.json'
    if not inputs.is_file(): inv.references.append({'source':str(inputs),'missing_field':'S05 experiment_inputs.json: cannot discover authoritative runtime closure'})
    for record in list(inv.records.values()):
        if record['logical_root']=='checkpoints' and not record['is_directory'] and record['source_path'].endswith(('.pt','.pth','.ckpt')): inv.inspect_checkpoint(record['source_path'])
    payload=inv.write(out)
    try:
        head=subprocess.check_output(['git','-C',str(a.repo_root),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(a.repo_root),'status','--short'],text=True).strip()
    except (OSError,subprocess.CalledProcessError): head=None; dirty=None
    (out/'code_identity.json').write_text(json.dumps({'head':head,'dirty_status':dirty,'repository':'universe26dyz/CR_DREME_v3','bundle':'CARDIORESP_DISENTANGLE_RUNTIME_V1'},indent=2)+'\n')
    print((out/'migration_summary.md').read_text())
    print(f'Inventory: {out}; source files were not modified.')

if __name__=='__main__': main()
