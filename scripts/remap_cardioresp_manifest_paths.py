#!/usr/bin/env python3
"""Write path-remapped copies; do not modify authoritative originals or checkpoints."""
import argparse,csv,hashlib,json
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def remap(value,old,new):
    if isinstance(value,dict): return {k:remap(v,old,new) for k,v in value.items()}
    if isinstance(value,list): return [remap(v,old,new) for v in value]
    if isinstance(value,str) and (value==old or value.startswith(old+'/')): return new+value[len(old):]
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--old-root',required=True); p.add_argument('--new-root',required=True); p.add_argument('--input',required=True,type=Path); p.add_argument('--output',required=True,type=Path); p.add_argument('--report',type=Path,required=True); a=p.parse_args()
    old=a.old_root.rstrip('/'); new=a.new_root.rstrip('/')
    if not old.startswith('/') or not new.startswith('/') or not old or not new: p.error('roots must be non-root absolute paths')
    if a.input.resolve()==a.output.resolve() or a.output.exists() or a.report.exists() or a.output.resolve()==a.report.resolve(): p.error('output and report must be distinct new files')
    if a.input.suffix.lower()=='.json':
        value=json.loads(a.input.read_text()); result=json.dumps(remap(value,old,new),indent=2)+'\n'
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(result)
    elif a.input.suffix.lower()=='.csv':
        with a.input.open(newline='') as f: reader=csv.DictReader(f); fields=reader.fieldnames; rows=[remap(r,old,new) for r in reader]
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    else: p.error('input must be JSON or CSV')
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps({'input':str(a.input.resolve()),'input_sha256':sha(a.input),'output':str(a.output.resolve()),'output_sha256':sha(a.output),'old_root':old,'new_root':new,'warning':'Changed manifests may invalidate embedded hashes and checkpoint provenance; no checkpoint or dependent hash is rewritten.'},indent=2)+'\n')

if __name__=='__main__': main()
