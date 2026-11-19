"""Scheme 3 v2: independent full-capacity motion inside a physical heart ROI."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from third_party.simple4d.simple4d.networks import MotionCodeVAE, DeformCodeAdaptive
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.roi.roi_qc import box_plane_intersection
from cardioresp4d.roi.auto_cardiac import soft_box_torch


def crop_rectangle(observation, box, margin=.1):
    """Pixel rectangle from the corrected box-plane intersection, with margin.

    If the plane misses the box, use its orthogonal projection; the localized
    deformation is still exactly zero wherever the support has no intersection.
    """
    if margin < 0: raise ValueError('margin must be nonnegative')
    h,w=observation.image.shape[-2:]
    row=observation.row_direction.detach().cpu().numpy(); col=observation.column_direction.detach().cpu().numpy(); spacing=observation.pixel_spacing_mm.detach().cpu().numpy(); center=observation.center_mm.detach().cpu().numpy()
    origin=center-(w-1)*spacing[1]*row/2-(h-1)*spacing[0]*col/2
    plane=DicomPlane(origin,row,col,spacing,h,w)
    points=box_plane_intersection(box,plane)['pixel']
    if len(points)<3: points=plane.world_to_pixel(box.corners_mm)
    low=points.min(axis=0); high=points.max(axis=0); delta=(high-low)*margin
    return np.array([*(low-delta),*(high+delta)])


def heart_crop(observation, rectangle, size=128):
    if size<32: raise ValueError('crop size must be at least 32')
    h,w=observation.image.shape[-2:]; x0,y0,x1,y1=rectangle
    x=torch.linspace(float(x0),float(x1),size,device=observation.image.device,dtype=observation.image.dtype)
    y=torch.linspace(float(y0),float(y1),size,device=observation.image.device,dtype=observation.image.dtype)
    yy,xx=torch.meshgrid(y,x,indexing='ij'); grid=torch.stack((2*xx/(w-1)-1,2*yy/(h-1)-1),-1).unsqueeze(0)
    return F.grid_sample(observation.image.unsqueeze(0),grid,mode='bilinear',padding_mode='zeros',align_corners=True)


class FullCardiacEncoder(nn.Module):
    def __init__(self, box, crop_size=128, margin=.1):
        super().__init__(); self.network=MotionCodeVAE(code_dim=32,base_channels=32,pos_embed_dim=32,dropout=.1); self.box=box; self.crop_size=crop_size; self.margin=margin; self._rectangles={}
    def forward(self,obs,box=None,taper_mm=None):
        if obs.normalized_slice_position is None: raise ValueError('normalized slice position required')
        key=(obs.view,obs.slice_id)
        if key not in self._rectangles: self._rectangles[key]=crop_rectangle(obs,self.box,self.margin)
        image=heart_crop(obs,self._rectangles[key],self.crop_size)
        return self.network(image,image.new_tensor([obs.normalized_slice_position]))


class FullCardiacDeformation(nn.Module):
    def __init__(self,coordinates):
        super().__init__(); self.coordinates=coordinates; self.network=DeformCodeAdaptive([3,256,256,256,256,3],code_dim=32,omega=7,residual=False)
    def forward(self,points,code,box,taper_mm):
        if points.ndim!=3 or code.shape!=(points.shape[0],32): raise ValueError('expected points [B,N,3] and code [B,32]')
        norm=self.coordinates.world_to_normalized(points); raw=self.network(norm,code)
        displacement=self.coordinates.raw_deformation_to_world_mm(raw)
        return displacement*soft_box_torch(points,box,taper_mm).unsqueeze(-1)


class FullCardiacModel(nn.Module):
    def __init__(self,base,box,taper_mm=15.,crop_size=128,margin=.1):
        super().__init__(); self.base=base; self.box=box; self.taper_mm=taper_mm; self.stage=2; base.requires_grad_(False); base.eval()
        self.cardiac_encoder=FullCardiacEncoder(box,crop_size,margin)
        self.forward_residual=FullCardiacDeformation(base.forward_deformation.coordinates)
        self.inverse_residual=FullCardiacDeformation(base.inverse_deformation.coordinates)
    def set_stage(self,stage):
        if stage not in (2,3): raise ValueError('stage must be 2 or 3')
        self.stage=stage; self.base.requires_grad_(False)
        if stage==3:
            for m in (self.base.respiratory_encoder,self.base.forward_deformation,self.base.inverse_deformation): m.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode); self.base.eval(); return self
    def forward_displacement(self,points,base_code,cardiac_code):
        return self.base.forward_deformation(points,base_code)['displacement_mm']+self.forward_residual(points,cardiac_code,self.box,self.taper_mm)
    def inverse_displacement(self,points,base_code,cardiac_code):
        return self.base.inverse_deformation(points,base_code)['displacement_mm']+self.inverse_residual(points,cardiac_code,self.box,self.taper_mm)


def temporal_energy(means,floor):
    if floor<=0 or means.ndim!=2 or means.shape[0]<2: raise ValueError('need a positive floor and temporal batch')
    variance=(means-means.mean(dim=0,keepdim=True)).square().mean()
    return variance,torch.relu((floor-variance)/floor).square()


def regularization_weights(step):
    ramp=min(max((step-500)/1500,0),1)
    return {'kl':1e-4,'magnitude':1e-4+(1e-2-1e-4)*ramp,'cycle':1e-2+(.1-1e-2)*ramp,'jacobian':1e-4+(1e-3-1e-4)*ramp}


class FullCardiacRunner:
    """Stage-2 frozen-base optimization, with resumable sampler and RNG."""
    def __init__(self, model, sampler, *, pixel_samples=10000, global_fraction=.25, lambda_roi=2., lr=5e-5, reg_samples=256, floor=None, energy_weight=.01, stage=2):
        if pixel_samples < 2 or not 0 < global_fraction < 1 or lambda_roi < 0 or reg_samples < 1:
            raise ValueError('invalid stage 2 sampling or loss settings')
        if floor is None or floor <= 0: raise ValueError('positive calibrated floor required')
        self.floor=floor; self.energy_weight=energy_weight; self.stage=stage; model.set_stage(stage)
        self.model=model; self.sampler=sampler; self.pixel_samples=pixel_samples; self.global_fraction=global_fraction; self.lambda_roi=lambda_roi; self.reg_samples=reg_samples
        params=list(model.cardiac_encoder.parameters())+list(model.forward_residual.parameters())+list(model.inverse_residual.parameters())
        groups=[{'name':'cardiac','params':params,'lr':lr}]
        if stage==3:
            groups.append({'name':'base','params':list(model.base.respiratory_encoder.parameters())+list(model.base.forward_deformation.parameters())+list(model.base.inverse_deformation.parameters()),'lr':lr*.1})
        self.optimizer=torch.optim.Adam(groups)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=5000,gamma=.5)
        self.step=0; self.metrics=[]

    def update(self):
        from cardioresp4d.roi.auto_cardiac import observation_soft_roi
        from cardioresp4d.experiments.simple4d_cardiac_residual import render_residual
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
            if self.stage==2:
                with torch.no_grad(): base_code=self.model.base.encode_observation(obs,sample=False)['code']
            else: base_code=self.model.base.encode_observation(obs,sample=False)['code']
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
            weights=regularization_weights(self.step+1 if self.stage==2 else 5000)
            loss=photo+self.lambda_roi*heart+weights['kl']*kl+weights['jacobian']*jac+weights['magnitude']*mag+weights['cycle']*cycle
            if self.stage==3:
                _,base_dvf=self.model.base.forward_deformation.forward_normalized(x,base_code)
                base_inverse=self.model.base.inverse_deformation(world+base_dvf*coords.extent_mm/2,base_code)['displacement_mm']*2/coords.extent_mm
                base_det=torch.linalg.det(jacobian_matrix(x,base_dvf)).clamp(.1,10)
                base_reg=1e-3*(base_det+1/base_det-2).mean()+.1*dvf_magnitude_loss(base_dvf)+.1*(base_dvf+base_inverse).abs().mean()
                loss=loss+base_reg
            if not torch.isfinite(loss): raise RuntimeError('nonfinite stage loss')
            (loss/len(observations)).backward()
            for name,value in [('total',loss),('global_mae',photo),('roi_mae',heart),('cycle',cycle),('jacobian',jac),('folding_fraction_sampled',folding),('residual_magnitude',mag)]: sums[name]=sums.get(name,0.)+float(value.detach())/len(observations)
        sequence=self.sampler.temporal_batch(max_items=8)
        if len(sequence)<3: raise RuntimeError('same-location temporal batch unavailable')
        encoder=self.model.cardiac_encoder.network; encoder.eval()
        means=torch.cat([self.model.cardiac_encoder(o,self.model.box,self.model.taper_mm)[1] for o in sequence])
        encoder.train()
        variance,energy=temporal_energy(means,self.floor)
        (self.energy_weight*energy).backward()
        sums['card_variance']=float(variance.detach()); sums['energy_loss']=float(energy.detach()); sums['variance_floor_ratio']=sums['card_variance']/self.floor
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
