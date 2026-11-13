"""S1 learning core with the project physical multi-view observation model.

SIMPLE-4D upstream c466b276a438b3e052ad3b44d38f398872d8aeae is used
without modification. LPS xyz is mapped to network xyz, not array axes.
"""
import torch
from torch import nn
from third_party.simple4d.simple4d.networks import MotionCodeVAE, InstantNGPINR
from cardioresp4d.simple4d.coordinates import Simple4DWorldCoordinateAdapter
from cardioresp4d.simple4d.deformation import Simple4DRespiratoryDeformation
from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.training.model import SourceFirstDynamicModel


class PhysicalSimple4DCanonical(nn.Module):
    def __init__(self, coordinates, *, use_tcnn=True, require_tcnn=False, ngp_options=None):
        super().__init__()
        self.coordinates=coordinates
        self.network=InstantNGPINR(use_tcnn=use_tcnn,**(ngp_options or {}))
        if require_tcnn and not self.network.use_tcnn:
            raise RuntimeError('BLOCKED_TCNN_UNAVAILABLE')
        self.backend='tcnn' if self.network.use_tcnn else 'pytorch_fallback'

    def forward(self,points,return_features=False):
        density=self.network(self.coordinates.world_to_normalized(points)).squeeze(-1)
        return (density,None) if return_features else density


class Simple4DCoreMultiView(nn.Module):
    """From-scratch upstream NGP/VAE/DVF; no view ID enters the encoder."""
    def __init__(self,lower_world_mm,upper_world_mm,reference_spacing_mm,*,use_tcnn=True,require_tcnn=False,ngp_options=None,psf_samples=8):
        super().__init__()
        def coords(): return Simple4DWorldCoordinateAdapter(lower_world_mm,upper_world_mm,reference_spacing_mm)
        self.canonical=PhysicalSimple4DCanonical(coords(),use_tcnn=use_tcnn,require_tcnn=require_tcnn,ngp_options=ngp_options)
        self.respiratory_encoder=MotionCodeVAE(code_dim=32,base_channels=32,pos_embed_dim=32,dropout=.1)
        self.forward_deformation=Simple4DRespiratoryDeformation(coords())
        self.inverse_deformation=Simple4DRespiratoryDeformation(coords())
        self.psf=NeSVoRPSFAdapter(psf_samples)

    def encode_observation(self,observation,*,sample):
        if observation.normalized_slice_position is None: raise ValueError('normalized slice position required')
        encoder=self.respiratory_encoder; was_training=encoder.training
        encoder.train(sample)
        try:
            image=observation.image.unsqueeze(0); pos=image.new_tensor([observation.normalized_slice_position])
            code,mu,logvar=encoder(image,pos)
        finally: encoder.train(was_training)
        return {'code':code,'mu':mu,'logvar':logvar}

    def render_with_encoded(self,observation,pixel_uv,encoded):
        image=observation.image.unsqueeze(0); points=SourceFirstDynamicModel._pixel_world(observation,pixel_uv.to(image)); n=len(points)
        resolution=image.new_tensor([observation.pixel_spacing_mm[1],observation.pixel_spacing_mm[0],observation.slice_thickness_mm]).expand(n,3)
        def motion(samples): return self.forward_deformation(samples,encoded['code'].expand(samples.shape[0],-1))
        result=self.psf(self.canonical,points,resolution,row_direction=observation.row_direction.to(image).expand(n,-1),column_direction=observation.column_direction.to(image).expand(n,-1),normal=observation.normal.to(image).expand(n,-1),motion=motion)
        result.update(encoded); return result


def make_s1_optimizer(model):
    """Upstream paper's joint-from-scratch learning rates and StepLR."""
    groups=[{'name':'hash','params':list(model.canonical.network.encoding_parameters()),'lr':1e-2},
            {'name':'template','params':list(model.canonical.network.mlp_parameters()),'lr':2e-5},
            {'name':'encoder','params':list(model.respiratory_encoder.parameters()),'lr':5e-5},
            {'name':'forward','params':list(model.forward_deformation.parameters()),'lr':5e-5},
            {'name':'inverse','params':list(model.inverse_deformation.parameters()),'lr':5e-5}]
    optimizer=torch.optim.Adam(groups)
    return optimizer,torch.optim.lr_scheduler.StepLR(optimizer,step_size=5000,gamma=.5)


class S1Runner:
    """Joint update with upstream step-dependent encoder evaluation behavior."""
    def __init__(self,model,sampler,*,observations_per_update=8,pixel_samples=10000):
        self.model=model; self.sampler=sampler; self.observations_per_update=observations_per_update; self.pixel_samples=pixel_samples
        self.optimizer,self.scheduler=make_s1_optimizer(model); self.global_step=0; self.metrics=[]

    def update(self):
        from cardioresp4d.experiments.simple4d_resp_v1_r0 import sample_pixels,grad_stats
        from cardioresp4d.simple4d.losses import respiratory_loss
        self.optimizer.zero_grad(set_to_none=True)
        observations=self.sampler.sample_step(observations_per_update=self.observations_per_update)
        if len(observations)!=self.observations_per_update: raise RuntimeError('unexpected sampler batch size')
        step=self.global_step+1; sums={}
        for obs in observations:
            pixels=sample_pixels(obs.image,self.pixel_samples); encoded=self.model.encode_observation(obs,sample=step%10!=0)
            rendered=self.model.render_with_encoded(obs,pixels,encoded); uv=pixels.long(); target=obs.image[0,uv[:,1],uv[:,0]]
            world=SourceFirstDynamicModel._pixel_world(obs,pixels.to(obs.image)).unsqueeze(0)
            coords=self.model.forward_deformation.coordinates.world_to_normalized(world).detach().clone().requires_grad_(True)
            loss=respiratory_loss(rendered['predicted_intensity'],target,encoded['mu'],encoded['logvar'],coords,self.model.forward_deformation,self.model.inverse_deformation,encoded['code'],step)
            if not torch.isfinite(loss['total']): raise RuntimeError('nonfinite S1 loss')
            (loss['total']/len(observations)).backward()
            for key,value in loss.items(): sums[key]=sums.get(key,0.)+float(value.detach())/len(observations)
        row={'global_step':step,**sums}
        for name,module in [('canonical',self.model.canonical),('encoder',self.model.respiratory_encoder),('forward',self.model.forward_deformation),('inverse',self.model.inverse_deformation)]:
            norm,rms=grad_stats(module); row[name+'_grad_norm']=norm; row[name+'_grad_rms']=rms
        if not all(row[name+'_grad_norm']>0 for name in ('canonical','encoder','forward','inverse')): raise RuntimeError('S1 owner has zero gradient')
        for group in self.optimizer.param_groups: row['lr_'+group['name']]=group['lr']
        self.optimizer.step(); self.scheduler.step(); self.global_step=step; self.metrics.append(row); return row

    def state_dict(self):
        from cardioresp4d.training.runtime_state import capture_rng_state
        return {'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'sampler':self.sampler.state_dict(),'rng':capture_rng_state(),'global_step':self.global_step,'metrics':self.metrics}

    def load_state_dict(self,state):
        from cardioresp4d.training.runtime_state import restore_rng_state
        self.model.load_state_dict(state['model'],strict=True); self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler']); self.sampler.load_state_dict(state['sampler']); self.global_step=state['global_step']; self.metrics=state['metrics']; restore_rng_state(state['rng'])


def validate_s1_resume(payload,provenance):
    if payload.get('schema')!='simple4d_s1_v1' or payload.get('provenance')!=provenance or not isinstance(payload.get('runner'),dict): raise ValueError('S1 checkpoint provenance mismatch')
