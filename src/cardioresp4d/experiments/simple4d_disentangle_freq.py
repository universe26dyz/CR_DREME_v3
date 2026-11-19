"""Dual latent and band projection adapted from Zou et al. (2023), doi:10.3390/bioengineering10030345, section 3.2."""
import math
import random
import torch
from torch import nn
from third_party.simple4d.simple4d.networks import MotionCodeVAE
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice, S1Runner

class FrequencyDisentangledMotionCodeVAE(MotionCodeVAE):
    def __init__(self, resp_dim=16, card_dim=16, total_dim=32):
        if resp_dim + card_dim != total_dim:
            raise ValueError('latent branch dimensions must sum to total')
        super().__init__(code_dim=total_dim, base_channels=32, pos_embed_dim=32, dropout=.1)
        hidden = self.fc_mu.in_features
        del self.fc_mu, self.fc_logvar
        self.resp_dim, self.card_dim = resp_dim, card_dim
        self.resp_mu, self.resp_logvar = nn.Linear(hidden, resp_dim), nn.Linear(hidden, resp_dim)
        self.card_mu, self.card_logvar = nn.Linear(hidden, card_dim), nn.Linear(hidden, card_dim)

    def encode(self, image, slice_position):
        if image.dim() == 3: image = image.unsqueeze(1)
        h = self.fc_hidden(torch.cat([self.image_encoder(image), self.pos_embedding(slice_position)], -1))
        return torch.cat((self.resp_mu(h), self.card_mu(h)), -1), torch.cat((self.resp_logvar(h), self.card_logvar(h)), -1)

class DualLatentThickSlice(Simple4DCoreMultiViewThickSlice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.respiratory_encoder = FrequencyDisentangledMotionCodeVAE()

def band_residual(times, latent, band, eps=1e-8, *, observability_mode='full_band'):
    """Centered SVD basis, frequencies spaced 1/span Hz, no saturated basis."""
    if times.ndim != 1 or latent.ndim != 2 or latent.shape[0] != times.numel() or len(times)<8:
        raise ValueError('expected at least eight aligned time points')
    times = times.to(device=latent.device, dtype=torch.float64)
    if not bool(torch.isfinite(times).all()) or not bool((times[1:] > times[:-1]).all()):
        raise ValueError('timestamps must be finite and strictly increasing')
    t = times-times[0]; span=float(t[-1]); low,high=map(float,band)
    if observability_mode not in ('full_band','observable_basis_subset'): raise ValueError('unknown observability mode')
    if not 0 < low <= high or (observability_mode=='full_band' and span < 1/low):
        raise ValueError('insufficient span for lower frequency')
    if 0.5/float(torch.median(torch.diff(times))) < high:
        raise ValueError('band exceeds median-interval Nyquist frequency')
    freqs=torch.arange(math.ceil(low*span), math.floor(high*span)+1, device=latent.device, dtype=torch.float64)/span
    if not len(freqs): raise ValueError('no resolvable frequencies in band')
    phase=2*math.pi*t[:,None]*freqs[None,:]
    basis=torch.cat((torch.sin(phase),torch.cos(phase)),dim=1)
    basis=basis-basis.mean(dim=0,keepdim=True)
    u,s,_=torch.linalg.svd(basis,full_matrices=False)
    rank=int((s > s[0]*1e-8).sum())
    if rank >= len(times)-2: raise ValueError('basis saturates temporal samples')
    centered=latent-latent.mean(dim=0,keepdim=True)
    q=u[:,:rank].to(dtype=latent.dtype)
    residual=centered-q@(q.T@centered)
    return residual.square().sum()/(centered.square().sum()+eps)

class SameLocationSampler:
    def __init__(self, main_sampler, seed=13, min_points=8, max_points=50):
        self.rng=random.Random(seed); self.max_points=max_points
        self.candidates=[]
        for view, locations in main_sampler._grouped.items():
            for location, frames in locations.items():
                ordered=sorted(frames,key=lambda x:x.timestamp_s)
                if len(ordered)>=min_points and all(b.timestamp_s>a.timestamp_s for a,b in zip(ordered,ordered[1:])):
                    self.candidates.append((view,location,ordered))
        if not self.candidates: raise ValueError('no same-location sequences with sufficient distinct timestamps')
    def sample(self):
        view,location,frames=self.rng.choice(self.candidates)
        if len(frames)>self.max_points:
            indices=torch.linspace(0,len(frames)-1,self.max_points).round().long().tolist()
            frames=[frames[i] for i in indices]
        return frames
    def state_dict(self): return {'random_state':self.rng.getstate()}
    def load_state_dict(self,state): self.rng.setstate(state['random_state'])

class FrequencyRunner(S1Runner):
    """Add auxiliary gradient before existing S1 optimizer step."""
    def __init__(self, model, sampler, *, view_counts, frequency):
        super().__init__(model,sampler,view_counts=view_counts,sampling_mode='location_proportional_fixed_budget')
        self.frequency=frequency
        self.temporal=SameLocationSampler(sampler,seed=frequency['sampler_seed'],min_points=frequency['min_temporal_points'],max_points=frequency['temporal_batch_size'])

    def update(self):
        metrics={'resp_band_loss':None,'card_band_loss':None,'resp_mean_rms':None,'card_mean_rms':None,'weighted_resp_freq':0.,'weighted_card_freq':0.}
        def before_step(optimizer, args, kwargs):
            step=self.global_step+1; cfg=self.frequency
            if step % cfg['temporal_loss_every'] or not cfg['enabled'] or (cfg['lambda_resp_freq']==0 and cfg['lambda_card_freq']==0):
                return
            seq=self.temporal.sample()
            means=torch.cat([self.model.encode_observation(o,sample=False)['mu'] for o in seq],dim=0)
            times=torch.tensor([o.timestamp_s for o in seq],device=means.device,dtype=torch.float64)
            resp,card=means[:,:16],means[:,16:]
            lr=band_residual(times,resp,cfg['resp_band_hz'],observability_mode=cfg.get('resp_observability_mode','full_band')) if cfg['lambda_resp_freq'] else None
            lc=band_residual(times,card,cfg['card_band_hz']) if cfg['lambda_card_freq'] else None
            loss=(cfg['lambda_resp_freq']*lr if lr is not None else 0)+(cfg['lambda_card_freq']*lc if lc is not None else 0)
            if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite frequency loss')
            loss.backward()
            metrics.update(resp_band_loss=float(lr.detach()) if lr is not None else None,card_band_loss=float(lc.detach()) if lc is not None else None,resp_mean_rms=float(resp.detach().square().mean().sqrt()),card_mean_rms=float(card.detach().square().mean().sqrt()),weighted_resp_freq=float((cfg['lambda_resp_freq']*lr).detach()) if lr is not None else 0.,weighted_card_freq=float((cfg['lambda_card_freq']*lc).detach()) if lc is not None else 0.)
        handle=self.optimizer.register_step_pre_hook(before_step)
        try: row=super().update()
        finally: handle.remove()
        row.update(metrics)
        row['total_with_frequency']=row['total']+metrics['weighted_resp_freq']+metrics['weighted_card_freq'] if 'total' in row else None
        return row
    def state_dict(self):
        state=super().state_dict(); state['temporal_sampler']=self.temporal.state_dict(); return state
    def load_state_dict(self,state):
        super().load_state_dict(state); self.temporal.load_state_dict(state['temporal_sampler'])

def latent_diagnostics(times, means):
    result={}
    for name,z,own,other in [('resp',means[:,:16],(.1,.6),(.8,2.)),('card',means[:,16:],(.8,2.),(.1,.6))]:
        centered=z-z.mean(0); s=torch.linalg.svdvals(centered); energy=s.square(); total=energy.sum()
        p=energy/(total+1e-12)
        rank95=int(torch.searchsorted(torch.cumsum(p,0),p.new_tensor(.95)).item()+1) if float(total)>0 else 0
        entry={'mean':float(z.mean()),'std':float(z.std()),'rms':float(z.square().mean().sqrt()),'per_dimension_std':z.std(0).detach().cpu().tolist(),'pca_explained_variance':p.detach().cpu().tolist(),'rank95':rank95,'effective_rank':float(torch.exp(-(p*torch.log(p+1e-12)).sum())) if float(total)>0 else 0.,'temporal_variance':float(centered.square().mean())}
        for label,band in [('in_band_energy_ratio',own),('cross_band_energy_ratio',other)]:
            try: entry[label]=float(1-band_residual(times,z,band,observability_mode='observable_basis_subset' if band==(.1,.6) else 'full_band'))
            except ValueError: entry[label]=None
        result[name]=entry
    return result

def intervention_diagnostics(deformation, points, means):
    """RMS DVF temporal variation for each branch with the other held at mean."""
    ref=means.mean(0,keepdim=True); results={}
    for name,indices in [('resp',slice(0,16)),('card',slice(16,32))]:
        codes=ref.expand_as(means).clone(); codes[:,indices]=means[:,indices]
        fields=torch.stack([deformation(points,code.unsqueeze(0))['displacement_mm'] for code in codes])
        results[name+'_dvf_temporal_rms_mm']=float((fields-fields.mean(0)).square().mean().sqrt())
    return results


def validate_enabled_bands(audit, frequency):
    for branch in ('resp','card'):
        if not frequency['enabled'] or not frequency['lambda_'+branch+'_freq']:
            continue
        expected={'resp':[0.1,0.6],'card':[0.8,2.0]}[branch]
        if list(frequency[branch+'_band_hz']) != expected:
            raise ValueError('new frequency band requires a matching temporal audit')
        if branch=='resp' and frequency.get('resp_observability_mode')=='observable_basis_subset':
            if any(v['missing_timestamps'] or v['duplicate_intervals'] or v['non_monotonic_intervals'] or v['nyquist_hz']['min'] < expected[1] or v['observations_per_location']['min'] < 8 or v['span_s']['min'] <= 0 or math.ceil(expected[0]*v['span_s']['min']) > math.floor(expected[1]*v['span_s']['min']) for v in audit['stacks'].values()): raise ValueError('observable respiratory subset unsupported')
            continue
        if any(v[branch+'_supported_locations'] != v['unique_locations'] for v in audit['stacks'].values()):
            raise ValueError('temporal observability gate: enabled '+branch+' band unsupported')
