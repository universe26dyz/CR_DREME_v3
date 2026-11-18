"""File-level inventory: explicit dependency graph, never traverses entire data root."""
from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path

PRIORITY={'MUST':0,'SHOULD':1,'OPTIONAL':2}
CHECKPOINT_EXTENSIONS={'.pt','.pth','.ckpt'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def relative_to_root(path,root):
    try: return str(Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root))))
    except ValueError: return None

def path_values(value, trail='$'):
    """All absolute path strings, including strings embedded in command lines."""
    import re
    if isinstance(value,dict):
        for k,v in value.items(): yield from path_values(v,f'{trail}.{k}')
    elif isinstance(value,list):
        for i,v in enumerate(value): yield from path_values(v,f'{trail}[{i}]')
    elif isinstance(value,str):
        if trail.lower().endswith('command'):
            import shlex
            try: tokens=shlex.split(value)
            except ValueError: tokens=value.split()
            for token in tokens:
                candidate=token.split('=',1)[-1] if '=' in token else token
                if candidate.startswith('/'): yield trail,candidate
        elif value.startswith('/') and not any(c in value for c in '\n\r'): yield trail,value
        else:
            for match in re.findall(r'(?<!\S)(/[^\s\'\";,]+)',value): yield trail,match

def duplicates(records):
    groups={}
    for r in records:
        if r.get('sha256'): groups.setdefault(r['sha256'],[]).append(r['source_path'])
    return {k:v for k,v in groups.items() if len(v)>1}

class Inventory:
    def __init__(self,repo_root,data_root,hash_large=False,limit=64*1024*1024):
        self.repo=Path(repo_root).absolute(); self.root=Path(data_root).absolute(); self.hash_large=hash_large; self.limit=limit
        self.records={}; self.references=[]; self.visited=set(); self.primary_inputs={}; self.checkpoint_metadata=[]
    def add(self,path,category,reason,referenced_by=None,chain=(),logical_root='other'):
        path=Path(os.path.abspath(Path(path).expanduser())); key=str(path); exists=path.exists(); symlink=path.is_symlink()
        if key in self.records:
            rec=self.records[key]
            if PRIORITY[category]<PRIORITY[rec['category']]: rec['category']=category; rec['reason']=reason; rec['logical_root']=logical_root
            if referenced_by and str(referenced_by) not in rec['referenced_by']: rec['referenced_by'].append(str(referenced_by))
            return rec
        stat=path.stat() if exists else None
        if exists and path.is_dir():
            # Only explicit directories (e.g. user-specified optional archives) are expanded.
            children=[p for p in path.rglob('*') if p.is_file()]
            for child in children: self.add(child,category,reason,path,(*chain,key),logical_root)
            size=sum(p.stat().st_size for p in children); count=len(children); checksum=None
        else:
            size=stat.st_size if stat else None; count=1 if stat else 0
            checksum=digest(path) if stat and path.is_file() and (size<=self.limit or self.hash_large or path.suffix.lower() in CHECKPOINT_EXTENSIONS) else None
        after=path.stat() if exists else None
        unstable=bool(stat and after and (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino))
        rel=relative_to_root(path,self.root); real=path.resolve(strict=False)
        rec={'category':category,'reason':reason,'source_path':key,'size_bytes':size,'file_count':count,'sha256':checksum,'hash_skipped_large':bool(exists and path.is_file() and checksum is None),'referenced_by':[str(referenced_by)] if referenced_by else [],'reference_chain':list(chain)+[key],'exists':exists,'portable_relative_path':rel,'outside_data_root':rel is None,'resolved_outside_data_root':relative_to_root(real,self.root.resolve()) is None,'is_symlink':symlink,'symlink_target':os.readlink(path) if symlink else None,'hardlink_count':stat.st_nlink if stat else None,'device':stat.st_dev if stat else None,'inode':stat.st_ino if stat else None,'logical_root':logical_root,'is_directory':bool(stat and path.is_dir()),'unstable_during_scan':unstable}
        self.records[key]=rec
        return rec
    def json(self,path,category,reason,chain=(),logical_root='static_reference',scan_paths=True):
        p=Path(path).absolute(); self.add(p,category,reason,chain[-1] if chain else None,chain,logical_root)
        if not p.is_file(): return None
        try: value=json.loads(p.read_text())
        except (ValueError,UnicodeError) as exc:
            self.references.append({'source':str(p),'error':str(exc)}); return None
        if scan_paths:
            self.references.extend({'source':str(p),'field':k,'path':v} for k,v in path_values(value))
        return value
    def source_manifest(self,path,chain):
        p=Path(path).absolute(); self.add(p,'MUST','observation manifest',chain[-1],chain,'dynamic_inputs')
        if not p.is_file(): return
        with p.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        if not rows: return
        if 'source_file_token' in rows[0]:
            side=p.with_name(p.stem+'.sensitive_runtime_paths.json'); data=self.json(side,'MUST','sensitive token-to-DICOM runtime sidecar',(*chain,str(p)),'dynamic_inputs')
            mapping=(data or {}).get('token_to_absolute_path',{})
            for row in rows:
                token=row.get('source_file_token',''); target=mapping.get(token)
                if not target:
                    self.references.append({'source':str(side),'field':'token_to_absolute_path','missing_token':token}); continue
                self.add(target,'MUST','DICOM loaded for observation and geometry/normalization',side,(*chain,str(p),str(side)),'dynamic_inputs')
        else:
            for row in rows:
                target=row.get('dicom_path')
                if target:
                    self.references.append({'source':str(p),'field':'dicom_path','path':target})
                    self.add(target,'MUST','DICOM loaded for observation and normalization',p,(*chain,str(p)),'dynamic_inputs')
    def static_manifest(self,path,chain):
        p=Path(path).absolute(); data=self.json(p,'MUST','static reference manifest',chain,'static_reference')
        if not data: return
        for key,label in [('volume','static volume and spacing'),('model','static model required by unified evaluator')]:
            val=data.get('outputs',{}).get(key,{}).get('path')
            if val:
                rec=self.add(val,'MUST',label,p,(*chain,str(p)),'static_reference')
                expected=data['outputs'][key].get('sha256'); rec['expected_sha256']=expected
                if expected and rec.get('sha256') and expected!=rec['sha256']: self.references.append({'source':str(p),'error':'static '+key+' SHA256 mismatch'})
            else: self.references.append({'source':str(p),'missing_field':'outputs.'+key+'.path'})
        val=data.get('corrected_pose_manifest',{}).get('path')
        if val:
            pose=self.json(val,'MUST','corrected LPS pose map',(*chain,str(p)),'static_reference')
            expected=data['corrected_pose_manifest'].get('sha256'); rec=self.records[str(Path(val).absolute())]; rec['expected_sha256']=expected
            if expected and rec.get('sha256') and expected!=rec['sha256']: self.references.append({'source':str(p),'error':'corrected pose SHA256 mismatch'})
            # Pose data are normally inline; path-valued fields, if present, are dependencies.
            if pose:
                for field,target in path_values(pose):
                    if 'path' in field.lower() or 'file' in field.lower(): self.add(target,'MUST','pose-manifest file reference',val,(*chain,str(p),str(val)),'static_reference')
        else: self.references.append({'source':str(p),'missing_field':'corrected_pose_manifest.path'})
    def run(self,run_dir,label):
        run=Path(run_dir).absolute(); inputs=run/'experiment_inputs.json'; data=self.json(inputs,'SHOULD',f'{label} experiment provenance',(), 'checkpoints')
        for name in ('checkpoint_step_2000.pt','training_report.json','training_metrics.csv','effective_config.json'):
            self.add(run/name,'SHOULD',f'{label} baseline checkpoint or compact provenance',None,(), 'checkpoints')
        if not data: return
        paths=data.get('paths',{}); provenance=data.get('provenance',{})
        if label=='S05': self.primary_inputs={key:paths[key] for key in ('manifest','qc_table','canonical_domain','static_manifest') if key in paths}
        for name in ('manifest','qc_table','canonical_domain','static_manifest'):
            path=paths.get(name)
            if not path:
                self.references.append({'source':str(inputs),'missing_field':'paths.'+name}); continue
            if name=='manifest': self.source_manifest(path,(str(inputs),))
            elif name=='static_manifest': self.static_manifest(path,(str(inputs),))
            elif name=='canonical_domain': self.json(path,'MUST',name,(str(inputs),),'static_reference')
            else: self.add(path,'MUST',name,inputs,(str(inputs),),'dynamic_inputs')
            expected=provenance.get(name)
            record=self.records.get(str(Path(path).absolute()))
            if expected and record and record.get('sha256') and record['sha256']!=expected:
                self.references.append({'source':str(inputs),'error':'provenance hash mismatch: '+name})
    def inspect_checkpoint(self,path):
        """Load only trusted user checkpoints on CPU; record metadata, never write weights."""
        p=Path(path).absolute()
        if not p.is_file(): return
        try:
            import torch
            state=torch.load(p,map_location='cpu',weights_only=False)
            provenance=state.get('provenance',{})
            paths=[{'field':field,'path':value} for field,value in path_values(provenance)]
            self.references.extend({'source':str(p),'field':'provenance'+x['field'][1:],'path':x['path']} for x in paths)
            self.checkpoint_metadata.append({'source':str(p),'schema':state.get('schema',state.get('checkpoint_schema')),'global_step':state.get('runner',{}).get('global_step',state.get('global_step')),'provenance_path_references':paths,'provenance_scalars':{k:v for k,v in provenance.items() if isinstance(v,(str,int,float,bool)) or v is None}})
        except Exception as exc:
            self.checkpoint_metadata.append({'source':str(p),'inspection_error':f'{type(exc).__name__}: {exc}'})

    def write(self,output):
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        records=sorted(self.records.values(),key=lambda r:(PRIORITY[r['category']],r['source_path']))
        payload={'schema':'cardioresp_migration_inventory_v1','data_root':str(self.root),'repo_root':str(self.repo),'records':records,'references':self.references,'primary_inputs':self.primary_inputs}
        (output/'migration_inventory.json').write_text(json.dumps(payload,indent=2)+'\n')
        with (output/'migration_inventory.csv').open('w',newline='') as f:
            fields=list(dict.fromkeys(k for row in records for k in row)) if records else ['category','source_path']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for row in records: w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        for category,name in [('MUST','required'),('SHOULD','recommended'),('OPTIONAL','optional')]:
            group=[r for r in records if r['category']==category and not r['is_directory']]
            (output/f'migration_{name}_paths.txt').write_text(''.join(r['source_path']+'\n' for r in group))
            (output/f'migration_{name}.rsync-files-from').write_text(''.join(r['portable_relative_path']+'\n' for r in group if r['exists'] and not r['unstable_during_scan'] and r['portable_relative_path'] is not None))
        missing=[r for r in records if not r['exists']]+[r for r in self.references if 'missing_field' in r or 'missing_token' in r or 'error' in r]
        (output/'missing_dependencies.json').write_text(json.dumps(missing,indent=2)+'\n')
        (output/'checkpoint_metadata.json').write_text(json.dumps(self.checkpoint_metadata,indent=2)+'\n')
        (output/'absolute_path_references.json').write_text(json.dumps(self.references,indent=2)+'\n')
        (output/'duplicate_files_by_sha256.json').write_text(json.dumps(duplicates(records),indent=2)+'\n')
        lines=['# Migration inventory','',f'Data root: `{self.root}`','', 'Paths and sidecar references may contain sensitive information; protect this report.','']
        files=[r for r in records if r['exists'] and not r['is_directory']]
        for category in ('MUST','SHOULD','OPTIONAL'):
            group=[r for r in files if r['category']==category]; lines.append(f'- {category}: {len(group)} files, {sum(r["size_bytes"] for r in group)} bytes ({sum(r["size_bytes"] for r in group)/2**30:.3f} GiB)')
        lines+=['','## Logical roots']
        for name in ('dynamic_inputs','static_reference','checkpoints','evaluation_audit','environment_metadata','other'):
            group=[r for r in files if r['logical_root']==name]; lines.append(f'- {name}: {len(group)} files, {sum(r["size_bytes"] for r in group)/2**30:.3f} GiB')
        lines+=['','## Largest 20 files']+[f'- {r["size_bytes"]/2**30:.3f} GiB: `{r["source_path"]}`' for r in sorted(files,key=lambda r:r['size_bytes'],reverse=True)[:20]]
        outside=[r['source_path'] for r in files if r['outside_data_root']]
        unstable=[r['source_path'] for r in files if r['unstable_during_scan']]
        lines+=['',f'Files changed during scan (excluded from rsync lists): {len(unstable)}; rerun after they settle.',f'Missing dependencies/references: {len(missing)}.',f'Files outside data root: {len(outside)}; transfer separately after inspection.','', 'Stages: (1) code and environment metadata, manifests and geometry; a subset is only a preliminary check, (2) all runtime DICOMs and static references, (3) selected baseline checkpoints and summaries, (4) optional visualizations.','', 'Use rsync -aH --info=progress2 --files-from=migration_required.rsync-files-from DATA_ROOT/ NEW_SERVER:TARGET/ without --delete; repeat for recommended/optional lists. Verify hashes after transfer.']
        (output/'migration_summary.md').write_text('\n'.join(lines)+'\n')
        return payload
