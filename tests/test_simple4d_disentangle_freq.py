import math
from types import SimpleNamespace
import pytest
import torch
from cardioresp4d.experiments.simple4d_disentangle_freq import FrequencyDisentangledMotionCodeVAE, SameLocationSampler, band_residual
from cardioresp4d.training.sampler import proportional_view_counts

@pytest.mark.parametrize('irregular',[False,True])
def test_frequency_projection(irregular):
    t=torch.linspace(0,20,201,dtype=torch.float64)
    if irregular: t[1:-1] += .025*torch.sin(torch.arange(1,200,dtype=t.dtype))
    resp=(torch.sin(2*math.pi*.25*t)+7).float().unsqueeze(1).requires_grad_()
    card=torch.sin(2*math.pi*1.2*t).float().unsqueeze(1)
    rr=band_residual(t,resp,(.1,.6)); rc=band_residual(t,resp,(.8,2.))
    cc=band_residual(t,card,(.8,2.)); cr=band_residual(t,card,(.1,.6))
    assert rr < .01 and cc < .01 and rc > .8 and cr > .8
    mixed=resp+card
    assert band_residual(t,mixed,(.1,.6)) > .3
    (rr+rc).backward(); assert torch.isfinite(resp.grad).all()

def test_gate_rejects_insufficient_span():
    with pytest.raises(ValueError,match='insufficient span'):
        band_residual(torch.linspace(0,8.36,50),torch.randn(50,16),(.1,.6))

def test_heads_and_downstream_gradients():
    torch.manual_seed(3)
    model=FrequencyDisentangledMotionCodeVAE(); image=torch.rand(2,1,64,64); pos=torch.zeros(2)
    model.train(); z,mu,logvar=model(image,pos)
    assert z.shape==mu.shape==logvar.shape==(2,32) and not torch.equal(z,mu)
    decoder=torch.nn.Linear(32,3); decoder(z).square().mean().backward()
    for layer in [model.image_encoder,model.resp_mu,model.card_mu,model.resp_logvar,model.card_logvar,decoder]:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in layer.parameters())
    model.eval(); a,mu,_=model(image,pos); b,_,_=model(image,pos)
    assert torch.equal(a,mu) and torch.equal(a,b)
    with pytest.raises(ValueError): FrequencyDisentangledMotionCodeVAE(15,16,32)

def test_temporal_sampler_and_resume():
    grouped={'SAX':{'a':[SimpleNamespace(timestamp_s=float(i),slice_id='a') for i in range(12)],'b':[SimpleNamespace(timestamp_s=float(i),slice_id='b') for i in range(12)]}}
    sampler=SameLocationSampler(SimpleNamespace(_grouped=grouped),max_points=8)
    state=sampler.state_dict(); one=sampler.sample(); sampler.load_state_dict(state); two=sampler.sample()
    assert [x.slice_id for x in one]==[x.slice_id for x in two]
    assert len({x.slice_id for x in one})==1
    assert all(b.timestamp_s>a.timestamp_s for a,b in zip(one,one[1:]))
    with pytest.raises(ValueError): SameLocationSampler(SimpleNamespace(_grouped={'SAX':{'a':grouped['SAX']['a'][:2]}}))

def test_fixed_proportional_quota():
    assert proportional_view_counts({'SAX':49,'2CH':52,'4CH':42},24)=={'SAX':8,'2CH':9,'4CH':7}

def test_real_deformation_receives_dual_code_gradients():
    from cardioresp4d.simple4d.coordinates import Simple4DWorldCoordinateAdapter
    from cardioresp4d.simple4d.deformation import Simple4DRespiratoryDeformation
    from cardioresp4d.experiments.simple4d_disentangle_freq import intervention_diagnostics
    model=FrequencyDisentangledMotionCodeVAE(); decoder=Simple4DRespiratoryDeformation(Simple4DWorldCoordinateAdapter([-10]*3,[10]*3,[1]*3))
    code,_,_=model(torch.rand(1,1,64,64),torch.zeros(1))
    output=decoder(torch.randn(1,5,3),code)['displacement_mm']; output.square().mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.parameters()) for layer in [model.image_encoder,model.resp_mu,model.card_mu,decoder])
    with torch.no_grad():
        result=intervention_diagnostics(decoder,torch.randn(1,5,3),torch.randn(4,32))
    assert all(math.isfinite(v) and v>=0 for v in result.values())

def test_runner_resume_restores_both_samplers_optimizer_scheduler_and_rng(monkeypatch):
    import io, random
    import cardioresp4d.experiments.simple4d_core_mv as core
    from cardioresp4d.experiments.simple4d_disentangle_freq import FrequencyRunner
    from cardioresp4d.training.runtime_state import set_reproducibility
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.layer=torch.nn.Linear(2,32)
        def encode_observation(self,obs,*,sample): return {'mu':self.layer(obs.x[None,:])}
    class MainSampler:
        def __init__(self):
            self.rng=random.Random(12)
            self._grouped={'SAX':{'a':[SimpleNamespace(timestamp_s=i*.2,x=torch.tensor([math.sin(i*.3),math.cos(i*.3)])) for i in range(111)],'b':[SimpleNamespace(timestamp_s=i*.2,x=torch.tensor([math.sin(i*.4),math.cos(i*.4)])) for i in range(111)]}}
        def state_dict(self): return self.rng.getstate()
        def load_state_dict(self,s): self.rng.setstate(s)
    def optimizer(model):
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        return opt,torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.5)
    def update(self):
        self.optimizer.zero_grad(); n=self.sampler.rng.random()
        loss=(self.model.layer(torch.rand(2))+n).square().mean(); loss.backward()
        self.optimizer.step(); self.scheduler.step(); self.global_step+=1
        row={'global_step':self.global_step,'loss':float(loss.detach())}; self.metrics.append(row); return row
    monkeypatch.setattr(core,'make_s1_optimizer',optimizer); monkeypatch.setattr(core.S1Runner,'update',update)
    cfg={'sampler_seed':13,'min_temporal_points':8,'temporal_batch_size':111,'temporal_loss_every':1,'enabled':True,'resp_band_hz':[.1,.6],'card_band_hz':[.8,2.],'lambda_resp_freq':.001,'lambda_card_freq':.001}
    set_reproducibility(6); runner=FrequencyRunner(Model(),MainSampler(),view_counts={'SAX':8,'2CH':9,'4CH':7},frequency=cfg)
    runner.update(); buf=io.BytesIO(); torch.save(runner.state_dict(),buf); runner.update()
    target=[p.detach().clone() for p in runner.model.parameters()]; target_row=runner.metrics[-1].copy()
    restored=FrequencyRunner(Model(),MainSampler(),view_counts={'SAX':8,'2CH':9,'4CH':7},frequency=cfg)
    buf.seek(0); restored.load_state_dict(torch.load(buf,weights_only=False)); restored.update()
    assert all(torch.equal(a,b) for a,b in zip(target,restored.model.parameters()))
    assert restored.metrics[-1]==target_row and restored.scheduler.state_dict()==runner.scheduler.state_dict()
