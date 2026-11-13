"""R1-local joint optimization primitives; no R0 optimizer state is inherited."""
import copy
import hashlib
import json
import math
import torch
from .simple4d_resp_v1_r0 import observation_loss, sample_pixels, grad_stats, state_digest, compatible_resume_provenance
from cardioresp4d.training.runtime_state import set_reproducibility, capture_rng_state, restore_rng_state

R1_OWNERS=('canonical','respiratory_encoder','forward_deformation','inverse_deformation')
R1_BOUNDARIES={'r1_short':1000,'r1_diagnostic':2000,'r1_mid':5000,'r1_formal':10000}


def initialize_from_r0(model,checkpoint,provenance,r0_config):
    step=checkpoint.get('global_step')
    if checkpoint.get('checkpoint_schema')!=1 or step not in (2000,10000) or checkpoint.get('runner_state',{}).get('global_step')!=step:
        raise ValueError('source must be R0 step 2000 or 10000')
    source_config=checkpoint.get('effective_config',{})
    config_hash=hashlib.sha256(json.dumps(source_config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if source_config.get('experiment',{}).get('stage')!='R0' or checkpoint.get('provenance',{}).get('effective_config_sha256')!=config_hash or not compatible_resume_provenance(checkpoint,provenance,r0_config):
        raise ValueError('source R0 protocol/provenance mismatch')
    model.load_state_dict(checkpoint['model'],strict=True)
    actual=model.state_dict()
    if any(not torch.equal(value.detach().cpu(),checkpoint['model'][name].detach().cpu()) for name,value in actual.items()): raise ValueError('source state mismatch')
    for owner in R1_OWNERS: getattr(model,owner).requires_grad_(True)
    set_reproducibility(6)
    return {'source_r0_step':step,'per_module_sha256':{name:state_digest(getattr(model,name)) for name in R1_OWNERS},'all_tensors_equal':True,'randomness_policy':'reset model RNG seed 6 after source weights; sampler seed 12'}


class R1Runner:
    def __init__(self,model,sampler,source_r0_step,*,observations_per_update=8,pixel_samples=10000):
        if source_r0_step not in (2000,10000): raise ValueError('invalid source R0 step')
        self.model=model; self.sampler=sampler; self.source_r0_step=source_r0_step; self.global_step=0; self.metrics=[]
        self.observations_per_update=observations_per_update; self.pixel_samples=pixel_samples
        for name in R1_OWNERS: getattr(model,name).requires_grad_(True)
        self.initial_canonical={k:v.detach().cpu().clone() for k,v in model.canonical.named_parameters()}
        self.initial_canonical_hash=state_digest(model.canonical)
        self.optimizer=torch.optim.Adam([{'name':name,'params':list(getattr(model,name).parameters()),'lr':2e-5 if name=='canonical' else 5e-5} for name in R1_OWNERS])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=5000,gamma=.5)

    def update(self):
        self.optimizer.zero_grad(set_to_none=True); observations=self.sampler.sample_step(observations_per_update=self.observations_per_update)
        if len(observations)!=self.observations_per_update: raise RuntimeError('invalid sampler count')
        sums={}
        for obs in observations:
            loss=observation_loss(self.model,obs,sample_pixels(obs.image,self.pixel_samples),self.global_step)
            if not torch.isfinite(loss['total']): raise RuntimeError('nonfinite loss')
            (loss['total']/len(observations)).backward()
            for name,value in loss.items(): sums[name]=sums.get(name,0.)+float(value.detach())/len(observations)
        stats=[grad_stats(getattr(self.model,name)) for name in R1_OWNERS]
        if stats[0][0]<=0: raise RuntimeError('canonical gradient must be finite and nonzero')
        row={'source_r0_step':self.source_r0_step,'r1_joint_step':self.global_step+1,**sums}
        for owner,(norm,rms),group in zip(R1_OWNERS,stats,self.optimizer.param_groups): row.update({owner+'_grad_norm':norm,owner+'_grad_rms':rms,'lr_'+owner:group['lr']})
        self.optimizer.step(); self.scheduler.step(); self.global_step+=1; self.metrics.append(row)
        return row

    def canonical_drift(self):
        delta=0.; initial=0.; current=0.
        for name,p in self.model.canonical.named_parameters():
            x=p.detach().cpu().double(); y=self.initial_canonical[name].double()
            delta+=float((x-y).square().sum()); initial+=float(y.square().sum()); current+=float(x.square().sum())
        delta=math.sqrt(delta)
        return {'canonical_initial_state_sha256':self.initial_canonical_hash,'canonical_current_state_sha256':state_digest(self.model.canonical),'canonical_parameter_norm':math.sqrt(current),'canonical_parameter_delta_norm':delta,'canonical_parameter_relative_delta':delta/math.sqrt(initial) if initial else None}

    def checkpoint(self,provenance):
        return {'schema':'simple4d_r1_v1','source_r0_step':self.source_r0_step,'r1_joint_step':self.global_step,'provenance':copy.deepcopy(provenance),'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'sampler':self.sampler.state_dict(),'rng':capture_rng_state(),'metrics':copy.deepcopy(self.metrics),'initial_canonical':self.initial_canonical,'initial_canonical_hash':self.initial_canonical_hash}

    def restore(self,payload,provenance):
        if payload.get('schema')!='simple4d_r1_v1' or payload.get('source_r0_step')!=self.source_r0_step or payload.get('provenance')!=provenance: raise ValueError('R1 checkpoint provenance mismatch')
        self.model.load_state_dict(payload['model'],strict=True); self.optimizer.load_state_dict(payload['optimizer']); self.scheduler.load_state_dict(payload['scheduler']); self.sampler.load_state_dict(payload['sampler']); self.global_step=payload['r1_joint_step']; self.metrics=copy.deepcopy(payload['metrics']); self.initial_canonical=payload['initial_canonical']; self.initial_canonical_hash=payload['initial_canonical_hash']; restore_rng_state(payload['rng'])
