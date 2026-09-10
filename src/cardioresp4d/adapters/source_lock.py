"""Runtime integrity verification for vendored source-first primitives."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
def verify_vendored_source_lock(project_root: str|Path) -> dict:
 root=Path(project_root); lock=json.loads((root/'third_party/SOURCE_LOCK.json').read_text())
 checked=[]
 for name,entry in lock.items():
  directory={'FiLM':'film'}.get(name,name)
  for relative,expected in entry['key_files_sha256'].items():
   path=root/'third_party'/directory/relative; actual=hashlib.sha256(path.read_bytes()).hexdigest()
   if actual!=expected: raise ValueError(f'vendored source hash mismatch: {name}/{relative}')
   checked.append(f'{name}/{relative}')
 return {'verified':True,'checked_files':checked,'lock_path':str(root/'third_party/SOURCE_LOCK.json')}
