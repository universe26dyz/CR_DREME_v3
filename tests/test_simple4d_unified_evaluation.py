from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from PIL import Image
from cardioresp4d.experiments.simple4d_unified_evaluation import ProjectAdapter,S0NativeAdapter,evaluate_matched,residual_metrics


class Model:
    def __init__(self): self.seen=[]
    def encode_observation(self,obs,*,sample): self.seen.append(obs.dynamic_frame_id); return {'mu':torch.tensor([[float(obs.dynamic_frame_id)]])}
    def render_with_encoded(self,obs,pixels,encoded): return {'predicted_intensity':torch.ones(len(pixels))*encoded['code'][0,0]}


def ob(index,view='SAX'):
    return SimpleNamespace(view=view,slice_id='x',dynamic_frame_id=index,timestamp_s=float(index),image=torch.ones(1,3,3)*index)


def test_v2_excludes_target_and_shared_gif(tmp_path):
    paths=[tmp_path/'a.pt',tmp_path/'b.pt']
    for p in paths: p.write_bytes(b'unchanged')
    models=[Model(),Model()]; adapters=[ProjectAdapter(models[0],paths[0],'A'),ProjectAdapter(models[1],paths[1],'B')]
    pixels=torch.tensor([[0,0],[1,1]])
    assert np.allclose(adapters[0].predict_v2(ob(0),ob(1),ob(2),pixels),1)
    assert models[0].seen==[0,2]
    assert np.allclose(adapters[0].predict_v1(ob(1),pixels),1) and models[0].seen[-1]==1
    out=tmp_path/'out'; evaluate_matched(adapters,[(ob(1),ob(2),ob(3)),(ob(0),ob(1),ob(2))],out,mode='v2',pairs=[('A','B')])
    manifest=json.loads((out/'evaluation_manifest.json').read_text()); pair=next(x for x in manifest['comparisons'] if x['methods']==['A','B'])
    assert pair['timestamps_s']==[1.,2.] and pair['residual_vmin']==0 and len(pair['method_provenance'])==2
    with Image.open(out/pair['gif']) as gif: assert gif.n_frames>=2
    assert all(p.read_bytes()==b'unchanged' for p in paths)
    metric=residual_metrics(np.array([0.,1.]),np.array([1.,0.]))
    assert metric==pytest.approx({'mse':1.,'mae':1.,'ncc':-1.})
    data=next((out/'data').glob('A_B*.npz'))
    with np.load(data) as arrays:
        assert np.max(np.abs(arrays['observed']-arrays['A']))<=pair['residual_vmax']
    assert {entry['checkpoint_sha256'] for entry in pair['method_provenance']}=={a.checkpoint_or_model_sha256 for a in adapters}
    first=ob(0); target=ob(1); last=ob(4); target.timestamp_s=1.; last.timestamp_s=4.
    assert np.allclose(adapters[0].predict_v2(first,target,last,pixels),1.)
    s0=S0NativeAdapter(None,None,[],paths[0])
    with pytest.raises(ValueError,match='SAX-only'): s0._position(ob(0,'2CH'))


def test_evaluator_cli_r0_only(tmp_path,monkeypatch):
    import importlib.util,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('method_comparison_cli',root/'scripts/evaluate_simple4d_method_comparison.py'); cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    run=tmp_path/'run'; run.mkdir(); cp=run/'checkpoint.pt'; torch.save({'global_step':2000,'provenance':{'test':'fixed'}},cp); original=cp.read_bytes()
    data=tmp_path/'input'; data.mkdir()
    for name in ('manifest','qc','domain','static'): (data/name).write_text('fixture')
    frames=[]
    for index in range(5):
        frame=ob(index); frame.image=torch.rand(1,3,3); frame.normalized_slice_position=0.; frames.append(frame)
    model=Model(); cohort=(('SAX','x',tuple(frames)),)
    monkeypatch.setattr(cli,'load_inputs',lambda args: ({},{'hashes':{'manifest':'m'}},model,cohort))
    monkeypatch.setattr(cli,'load_checkpoint',lambda *args:None)
    out=tmp_path/'evaluation'
    monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(data/'manifest'),'--qc-table',str(data/'qc'),'--canonical-domain',str(data/'domain'),'--static-manifest',str(data/'static'),'--r0-checkpoint',str(cp),'--output-dir',str(out),'--device','cpu','--mode','v2'])
    cli.main()
    manifest=json.loads((out/'evaluation_manifest.json').read_text())
    assert manifest['methods'][0]['method_id']=='R0' and manifest['input_sha256']=={'manifest':'m'}
    assert (out/'cohort.csv').exists() and (out/'frame_metrics.csv').exists() and (out/'comparison.csv').exists()
    assert list((out/'gif').glob('*.gif')) and cp.read_bytes()==original
