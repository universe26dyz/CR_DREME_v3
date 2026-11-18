import csv,importlib.util,json
from pathlib import Path
import pytest
from cardioresp4d.migration.inventory import Inventory,duplicates,path_values,relative_to_root,digest
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
preflight=module('preflight_migration','scripts/preflight_cardioresp_new_server.py')
remapper=module('remap_migration','scripts/remap_cardioresp_manifest_paths.py')


def test_discovery_and_inventory(tmp_path):
    data=tmp_path/'data'; data.mkdir(); run=data/'s05'; run.mkdir(); source=data/'input'; source.mkdir(); static=data/'static'; static.mkdir()
    dcm=source/'frame.dcm'; dcm.write_bytes(b'dicom'); manifest=source/'frames.csv'; manifest.write_text('source_file_token,view\ntoken,SAX\n'); (source/'frames.sensitive_runtime_paths.json').write_text(json.dumps({'token_to_absolute_path':{'token':str(dcm)}}))
    qc=source/'qc.csv'; qc.write_text('qc\n'); domain=source/'domain.json'; domain.write_text('{}'); pose=static/'poses.json'; pose.write_text(json.dumps({'poses':{}})); volume=static/'vol.nii'; volume.write_bytes(b'v'); model=static/'model.pt'; model.write_bytes(b'v')
    static_manifest=static/'manifest.json'; static_manifest.write_text(json.dumps({'corrected_pose_manifest':{'path':str(pose)},'outputs':{'volume':{'path':str(volume)},'model':{'path':str(model)}}}))
    (run/'experiment_inputs.json').write_text(json.dumps({'paths':{'manifest':str(manifest),'qc_table':str(qc),'canonical_domain':str(domain),'static_manifest':str(static_manifest)}}))
    (run/'checkpoint_step_2000.pt').write_bytes(b'v')
    inv=Inventory(tmp_path/'repo',data); inv.run(run,'S05'); out=tmp_path/'output'; result=inv.write(out)
    assert str(dcm) in inv.records and inv.records[str(dcm)]['category']=='MUST'
    assert 'frames.sensitive_runtime_paths.json' in ' '.join(inv.records[str(dcm)]['reference_chain'])
    assert relative_to_root(dcm,data)=='input/frame.dcm' and relative_to_root(tmp_path/'other',data) is None
    assert list(path_values({'a':[str(dcm)]}))[0][1]==str(dcm)
    assert [p for _,p in path_values({'command':'/bin/tool --in /data/file'})]==['/bin/tool','/data/file']
    assert preflight.check_embedded_paths(result,data)==[]
    assert digest(model) in duplicates(result['records'])
    assert 'input/frame.dcm' in (out/'migration_required.rsync-files-from').read_text()
    assert json.loads((out/'missing_dependencies.json').read_text())
    failures,skipped,files=preflight.check_inventory(result,data); assert not failures and 'observation manifest' in files
    dcm.unlink(); failures,_,_=preflight.check_inventory(result,data); assert any('missing required file' in f for f in failures)


def test_remap_no_original_overwrite(tmp_path,monkeypatch):
    source=tmp_path/'in.json'; source.write_text(json.dumps({'path':'/old/a','not_path':'/older/a','nested':['/old/b']})); out=tmp_path/'new.json'; report=tmp_path/'path_remap_manifest.json'
    monkeypatch.setattr('sys.argv',['remap','--old-root','/old','--new-root','/new','--input',str(source),'--output',str(out),'--report',str(report)])
    remapper.main(); result=json.loads(out.read_text()); assert result=={'path':'/new/a','not_path':'/older/a','nested':['/new/b']}
    assert json.loads(source.read_text())['path']=='/old/a' and json.loads(report.read_text())['input_sha256']==digest(source)
    with pytest.raises(SystemExit): remapper.main()
