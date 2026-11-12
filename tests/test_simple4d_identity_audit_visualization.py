import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import global_latent_pca

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('simple4d_identity_cli',ROOT/'scripts/audit_simple4d_resp_v1_checkpoints.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


def test_summary_and_latent_stage_smoke(tmp_path, monkeypatch):
    t=np.sin(np.arange(8)*.6)
    groups=[np.stack([t,.2*np.cos(np.arange(8)*.6),np.zeros(8)]+[np.zeros(8)]*29,axis=1).astype('float32')+i for i in (1,2,3)]
    cohort=[]
    for index, view in enumerate(('SAX','2CH','4CH')):
        cohort.append((view,'A',tuple(SimpleNamespace(vector=torch.tensor(group),dynamic_frame_id=n,timestamp_s=float(n),normalized_slice_position=0.) for n,group in enumerate(groups[index]))))
    class Model:
        def encode_observation(self, frame, *, sample):
            assert not sample
            return {'mu':frame.vector[None,:]}
    pca=global_latent_pca(groups)
    diag={k:0.2 for k in cli.SUMMARY_FIELDS}
    diag.update({'latent_dynamic_pc1_explained_variance_ratio':float(pca['explained_variance_ratios'][0]),'latent_dynamic_pc2_explained_variance_ratio':float(pca['explained_variance_ratios'][1]),'latent_dynamic_rank_95pct':pca['rank95'],'latent_mu_dynamic_rms':pca['latent_dynamic_rms'],'static_baseline_sampled_mse_full':1.,'diagnostic_sampled_reconstruction_mse_full':.8})
    monkeypatch.setattr(cli,'load_inputs',lambda args: ({},{'hashes':{'test':'a'}},Model(),cohort))
    monkeypatch.setattr(cli,'load_checkpoint',lambda path,step,config,prov,model: {'diagnostics':{f'step_{step}':diag}})
    run=tmp_path/'run'; run.mkdir(); (run/'checkpoint_step_2000.pt').write_bytes(b'unchanged')
    out=tmp_path/'out'
    args=['audit','--manifest','m','--qc-table','q','--canonical-domain','d','--static-manifest','s','--run-dir',str(run),'--output-dir',str(out),'--checkpoint-steps','2000','--stage']
    for stage in ('summary','latent'):
        monkeypatch.setattr(sys,'argv',args+[stage]); cli.main()
        dest=out/('00_summary' if stage=='summary' else '01_latent')
        assert (dest/'stage_manifest.json').exists()
        assert all(p.stat().st_size>0 for p in dest.rglob('*.png'))
        assert len(list(dest.rglob('*.png'))) == 5
        assert json.loads((dest/'stage_manifest.json').read_text())['checkpoints']['2000']['sha256'] == cli.sha256(run/'checkpoint_step_2000.pt')
    assert (run/'checkpoint_step_2000.pt').read_bytes() == b'unchanged'


def test_checkpoint_validation_and_no_input_mutation(tmp_path):
    import pytest
    from torch import nn
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.canonical=nn.Linear(1,1); self.respiratory_encoder=nn.Linear(1,1)
    model=Tiny()
    before={k:v.clone() for k,v in model.state_dict().items()}
    config=cli._load_config(ROOT/'configs/simple4d_resp_v1.yaml')
    provenance={'hash':'input'}
    checkpoint={'checkpoint_schema':1,'global_step':2000,'runner_state':{'global_step':2000},'model':model.state_dict(),'provenance':provenance}
    path=tmp_path/'checkpoint_step_2000.pt'; torch.save(checkpoint,path)
    digest=cli.sha256(path)
    cli.load_checkpoint(path,2000,config,provenance,model)
    assert cli.sha256(path)==digest and all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    with pytest.raises(ValueError): cli.load_checkpoint(path,2000,config, {'hash':'other'},model)
    with pytest.raises(ValueError): cli.load_checkpoint(path,5000,config,provenance,model)
    checkpoint['model']['respiratory_encoder.weight']=torch.zeros(2,2); torch.save(checkpoint,path)
    with pytest.raises(ValueError): cli.load_checkpoint(path,2000,config,provenance,model)
