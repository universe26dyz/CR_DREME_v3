"""Heart-localized additive residual primitives for staged S05PROP refinement.

Forward is the existing observation-to-canonical pullback. Inverse is a
separately learned canonical-to-observation mapping, evaluated at canonical
coordinates; its residual is not assumed to equal the negative forward field.
"""
import torch
from torch import nn
from third_party.simple4d.simple4d.networks import MotionCodeVAE, DeformCodeAdaptive
from cardioresp4d.roi.auto_cardiac import soft_box_torch, observation_soft_roi


class CardiacROIEncoder(nn.Module):
    def __init__(self, code_dim=16):
        super().__init__()
        self.network = MotionCodeVAE(code_dim=code_dim, base_channels=32, pos_embed_dim=32, dropout=.1)

    def forward(self, observation, box, taper_mm):
        if observation.normalized_slice_position is None:
            raise ValueError('normalized slice position required')
        mask = observation_soft_roi(observation, box, taper_mm)
        image = (observation.image * mask).unsqueeze(0)
        position = image.new_tensor([observation.normalized_slice_position])
        return self.network(image, position)


class CardiacResidualDeformation(nn.Module):
    def __init__(self, coordinates, code_dim=16, hidden_dim=128):
        super().__init__()
        self.coordinates = coordinates
        self.code_dim = code_dim
        self.network = DeformCodeAdaptive([3, hidden_dim, hidden_dim, hidden_dim, 3], code_dim=code_dim, omega=7, residual=False)

    def forward(self, points_world_mm, code, box, taper_mm):
        if points_world_mm.ndim != 3 or points_world_mm.shape[-1] != 3 or code.shape != (points_world_mm.shape[0], self.code_dim):
            raise ValueError('expected points [B,N,3] and cardiac code [B,code_dim]')
        normalized = self.coordinates.world_to_normalized(points_world_mm)
        raw = self.network(normalized, code)
        displacement = self.coordinates.normalized_deformation_to_world_mm(self.coordinates.raw_deformation_to_normalized(raw))
        return displacement * soft_box_torch(points_world_mm, box, taper_mm).unsqueeze(-1)


class StagedCardiacResidual(nn.Module):
    """Keep a pretrained base frozen and learn distinct forward/inverse fields."""
    def __init__(self, base, box, taper_mm=15., code_dim=16, hidden_dim=128):
        super().__init__()
        self.base = base
        self.box = box
        self.taper_mm = taper_mm
        self.base.requires_grad_(False)
        self.base.eval()
        self.cardiac_encoder = CardiacROIEncoder(code_dim)
        self.forward_residual = CardiacResidualDeformation(base.forward_deformation.coordinates, code_dim, hidden_dim)
        self.inverse_residual = CardiacResidualDeformation(base.inverse_deformation.coordinates, code_dim, hidden_dim)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward_displacement(self, points, base_code, cardiac_code):
        base = self.base.forward_deformation(points, base_code)['displacement_mm']
        return base + self.forward_residual(points, cardiac_code, self.box, self.taper_mm)

    def inverse_displacement(self, canonical_points, base_code, cardiac_code):
        base = self.base.inverse_deformation(canonical_points, base_code)['displacement_mm']
        return base + self.inverse_residual(canonical_points, cardiac_code, self.box, self.taper_mm)


def render_residual(model, observation, pixels, base_code, cardiac_code):
    from cardioresp4d.training.model import SourceFirstDynamicModel
    image = observation.image.unsqueeze(0)
    points = SourceFirstDynamicModel._pixel_world(observation, pixels.to(image))
    n = len(points)
    resolution = image.new_tensor([observation.pixel_spacing_mm[1], observation.pixel_spacing_mm[0], observation.slice_thickness_mm]).expand(n, 3)
    def motion(samples):
        displacement = model.forward_displacement(samples, base_code.expand(n, -1), cardiac_code.expand(n, -1))
        return {'reference_points_mm': samples + displacement}
    return model.base.psf(model.base.canonical, points, resolution, row_direction=observation.row_direction.to(image).expand(n, -1), column_direction=observation.column_direction.to(image).expand(n, -1), normal=observation.normal.to(image).expand(n, -1), motion=motion)


class CardiacRunner:
    """Stage-2 frozen-base optimization, with resumable sampler and RNG."""
    def __init__(self, model, sampler, *, pixel_samples=10000, global_fraction=.5, lambda_roi=1., lr=5e-5, reg_samples=256):
        if pixel_samples < 2 or not 0 < global_fraction < 1 or lambda_roi < 0 or reg_samples < 1:
            raise ValueError('invalid stage 2 sampling or loss settings')
        self.model=model; self.sampler=sampler; self.pixel_samples=pixel_samples; self.global_fraction=global_fraction; self.lambda_roi=lambda_roi; self.reg_samples=reg_samples
        params=list(model.cardiac_encoder.parameters())+list(model.forward_residual.parameters())+list(model.inverse_residual.parameters())
        self.optimizer=torch.optim.Adam(params,lr=lr)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=5000,gamma=.5)
        self.step=0; self.metrics=[]

    def update(self):
        import torch.nn.functional as F
        from third_party.simple4d.simple4d.losses import jacobian_matrix, dvf_magnitude_loss
        from cardioresp4d.training.model import SourceFirstDynamicModel
        from cardioresp4d.experiments.simple4d_resp_v1_r0 import sample_pixels
        from cardioresp4d.training.sampler import proportional_view_counts
        counts=proportional_view_counts({v:len(self.sampler._grouped[v]) for v in self.sampler._views},24)
        observations=self.sampler.sample_step(view_counts=counts)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True); sums={}
        for obs in observations:
            mask=observation_soft_roi(obs,self.model.box,self.model.taper_mm).detach()
            n_global=round(self.pixel_samples*self.global_fraction); n_roi=self.pixel_samples-n_global
            global_pixels=sample_pixels(obs.image,n_global)
            # A location outside the box has no cardiac support: retain its normal global observations.
            if mask.sum() > 0:
                ids=torch.multinomial(mask.flatten(),n_roi,replacement=True)
                extra=torch.stack((ids % mask.shape[1],torch.div(ids,mask.shape[1],rounding_mode='floor')),-1)
            else:
                extra=sample_pixels(obs.image,n_roi)
            pixels=torch.cat((global_pixels,extra))
            with torch.no_grad(): base_code=self.model.base.encode_observation(obs,sample=False)['code']
            card,mu,logvar=self.model.cardiac_encoder(obs,self.model.box,self.model.taper_mm)
            predicted=render_residual(self.model,obs,pixels,base_code,card)['predicted_intensity']
            uv=pixels.long(); target=obs.image[0,uv[:,1],uv[:,0]]; error=(predicted-target).abs(); weights=mask[uv[:,1],uv[:,0]]
            # Estimate global and weighted ROI losses using the uniformly sampled half.
            photo=error[:n_global].mean(); wg=weights[:n_global]; denom=wg.sum()
            heart=(error[:n_global]*wg).sum()/denom.clamp_min(1e-8)
            # ROI-prioritized half adds a separate importance-normalized heart estimate.
            roi_extra=error[n_global:].mean() if mask.sum()>0 else error.new_zeros(())
            heart=(heart+roi_extra)/2 if mask.sum()>0 and denom>0 else heart+roi_extra
            world=SourceFirstDynamicModel._pixel_world(obs,global_pixels[:min(self.reg_samples,n_global)].to(obs.image)).unsqueeze(0)
            coords=self.model.base.forward_deformation.coordinates
            x=coords.world_to_normalized(world).detach().requires_grad_(True); world=coords.normalized_to_world(x)
            displacement=self.model.forward_displacement(world,base_code,card); dvf=displacement*2/coords.extent_mm
            residual=self.model.forward_residual(world,card,self.model.box,self.model.taper_mm)*2/coords.extent_mm
            canonical=world+displacement
            inverse=self.model.inverse_displacement(canonical,base_code,card)
            cycle=((canonical+inverse-world)*2/coords.extent_mm).abs().mean()
            det=torch.linalg.det(jacobian_matrix(x,dvf)); jac=(det.clamp(.1,10)+1/det.clamp(.1,10)-2).mean(); folding=(det<=0).float().mean()
            mag=dvf_magnitude_loss(residual); kl=MotionCodeVAE.kl_loss(mu,logvar)
            loss=photo+self.lambda_roi*heart+1e-3*kl+1e-3*jac+1e-2*mag+.1*cycle
            if not torch.isfinite(loss): raise RuntimeError('nonfinite stage 2 loss')
            (loss/len(observations)).backward()
            for name,value in [('total',loss),('global_mae',photo),('roi_mae',heart),('cycle',cycle),('jacobian',jac),('folding_fraction_sampled',folding),('residual_magnitude',mag)]: sums[name]=sums.get(name,0.)+float(value.detach())/len(observations)
        if sums['folding_fraction_sampled'] > .01: raise RuntimeError('sampled folding fraction exceeds 1%')
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all(): raise RuntimeError('nonfinite stage 2 gradient')
        self.optimizer.step(); self.scheduler.step(); self.step+=1; row={'step':self.step,**sums}; self.metrics.append(row); return row

    def state_dict(self):
        from cardioresp4d.training.runtime_state import capture_rng_state
        return {'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'sampler':self.sampler.state_dict(),'rng':capture_rng_state(),'step':self.step,'metrics':self.metrics}

    def load_state_dict(self,state):
        from cardioresp4d.training.runtime_state import restore_rng_state
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler']); self.sampler.load_state_dict(state['sampler']); self.step=state['step']; self.metrics=state['metrics']; restore_rng_state(state['rng'])
