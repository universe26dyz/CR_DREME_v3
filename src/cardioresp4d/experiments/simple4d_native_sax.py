"""S0 input admission for upstream regular-stack SIMPLE-4D.

This is a SAX-only adapter; it does not implement or replace upstream networks.
"""
from __future__ import annotations
import numpy as np


def build_native_sax_series(observations, *, geometry_tolerance_mm=1e-3, direction_tolerance=1e-4):
    """Return an upstream-compatible series and an auditable frame mapping.

    Observations use nominal acquisition geometry. Upstream assumes a regular,
    parallel stack; observed positions may be a subset of its lattice. No image
    interpolation or geometric correction is performed here.
    """
    from third_party.simple4d.simple4d.data import SliceSeries
    frames=[o for o in observations if o.qc_valid and o.view.upper()=='SAX']
    if not frames: raise ValueError('no QC-valid SAX observations')
    groups={}
    for item in frames: groups.setdefault(str(item.slice_id),[]).append(item)
    if len(groups)<2: raise ValueError('native regular-stack admission requires at least two SAX locations')
    first=frames[0]
    def array(value):
        return np.asarray(value.detach().cpu() if hasattr(value,'detach') else value,dtype=np.float64)
    row,column,normal=map(array,(first.row_direction,first.column_direction,first.normal))
    basis=np.stack((row,column,normal),axis=1)
    if basis.shape!=(3,3) or not np.isfinite(basis).all() or not np.allclose(basis.T@basis,np.eye(3),atol=direction_tolerance,rtol=0):
        raise ValueError('SAX basis must be finite and orthonormal')
    spacing=array(first.pixel_spacing_mm); thickness=float(first.slice_thickness_mm)
    if spacing.shape!=(2,) or not np.isfinite(spacing).all() or np.any(spacing<=0) or not np.isfinite(thickness) or thickness<=0:
        raise ValueError('invalid pixel spacing or slice thickness')
    shape=tuple(first.image.shape)
    if len(shape)!=3 or shape[0]!=1: raise ValueError('images must be [1,H,W]')
    locations=[]
    for slice_id,items in groups.items():
        center=array(items[0].center_mm)
        for item in items:
            image=array(item.image); center_i=array(item.center_mm)
            if image.shape!=shape or not np.isfinite(image).all() or np.any((image<0)|(image>1)):
                raise ValueError('SAX images must share shape and be finite in [0,1]')
            if not np.isfinite(item.timestamp_s) or not np.isfinite(center_i).all() or not np.allclose(center_i,center,atol=geometry_tolerance_mm,rtol=0):
                raise ValueError('location frames require finite timestamps and identical nominal centers')
            if any(not np.allclose(array(getattr(item,key)),ref,atol=direction_tolerance,rtol=0) for key,ref in (('row_direction',row),('column_direction',column),('normal',normal))):
                raise ValueError('native SAX requires common nominal orientation')
            if not np.allclose(array(item.pixel_spacing_mm),spacing,atol=geometry_tolerance_mm,rtol=0) or not np.isfinite(float(item.slice_thickness_mm)) or abs(float(item.slice_thickness_mm)-thickness)>geometry_tolerance_mm:
                raise ValueError('native SAX requires constant in-plane spacing and thickness')
        locations.append((float(center@normal),slice_id,center))
    locations.sort(key=lambda value:(value[0],value[1]))
    positions=np.array([v[0] for v in locations]); gaps=np.diff(positions)
    if np.any(gaps<=geometry_tolerance_mm):
        raise ValueError('native SAX requires distinct positions on a uniformly spaced lattice')
    # Estimate the smallest routinely observed gap without averaging missing gaps.
    base_spacing=float(np.median(np.sort(gaps)[:max(1,len(gaps)//2)]))
    lattice_indices=np.rint((positions-positions[0])/base_spacing).astype(np.int64)
    fit_error=np.abs(positions-(positions[0]+lattice_indices*base_spacing))
    gap_error=np.abs(gaps-np.diff(lattice_indices)*base_spacing)
    if np.any(np.diff(lattice_indices)<=0) or np.any(fit_error>geometry_tolerance_mm) or np.any(gap_error>geometry_tolerance_mm):
        raise ValueError('native SAX requires locations on a uniformly spaced lattice')
    num_positions=int(lattice_indices[-1])+1
    reference=locations[0][2]
    for _,_,center in locations:
        delta=center-reference
        if np.linalg.norm(delta-(delta@normal)*normal)>geometry_tolerance_mm:
            raise ValueError('native SAX requires aligned in-plane centers')
    position_for={slice_id:int(index) for (_,slice_id,_),index in zip(locations,lattice_indices)}
    ordered=sorted(frames,key=lambda item:(float(item.timestamp_s),int(item.dynamic_frame_id),str(item.slice_id)))
    ids=[int(item.dynamic_frame_id) for item in ordered]
    if len(set(ids))!=len(ids): raise ValueError('duplicate dynamic frame IDs')
    data=np.stack([array(item.image)[0] for item in ordered]).astype(np.float32)
    indices=np.array([position_for[str(item.slice_id)] for item in ordered],dtype=np.int64)
    series=SliceSeries(data,indices,(base_spacing,float(spacing[0]),float(spacing[1])),num_positions,(0,0))
    mapping=[{'upstream_frame':i,'slice_id':str(item.slice_id),'slice_index':int(indices[i]),'dynamic_frame_id':int(item.dynamic_frame_id),'timestamp_s':float(item.timestamp_s)} for i,item in enumerate(ordered)]
    metadata={'scope':'SAX-only; upstream regular-stack coordinates, not arbitrary LPS slice geometry','geometry_source':'nominal_acquisition_geometry','corrected_pose_applied':False,'corrected_pose_role':'provenance_only_not_applied_to_s0_geometry','base_slice_spacing_mm':base_spacing,'num_nominal_positions':num_positions,'num_observed_positions':len(locations),'observed_slice_indices':lattice_indices.tolist(),'missing_slice_indices':sorted(set(range(num_positions))-set(lattice_indices.tolist())),'interpolation_used':False,'synthetic_slices_created':False,'projected_nominal_positions_mm':positions.tolist(),'lattice_origin_mm':float(positions[0]),'lattice_fit_max_abs_error_mm':float(fit_error.max()),'lattice_fit_median_abs_error_mm':float(np.median(fit_error)),'slice_thickness_mm':thickness,'spacing_mm':list(series.spacing),'image_shape':list(shape[1:]),'slice_order':[v[1] for v in locations],'geometry_tolerance_mm':geometry_tolerance_mm,'direction_tolerance':direction_tolerance,'frame_mapping':mapping}
    return series,metadata


def save_native_npz(series,path):
    np.savez_compressed(path,frames=series.frames,slice_index=series.slice_index,spacing=np.asarray(series.spacing),num_positions=series.num_positions,depth_crop=np.asarray(series.depth_crop))


class NativeSAXRunner:
    """Project checkpoint wrapper around the unmodified upstream training_step.

    A deterministic epoch permutation and cursor are saved to support exact
    resume. This wrapper uses a single-process loader; upstream model/loss and
    its tenth-step evaluation-mode behavior are unchanged.
    """
    def __init__(self,model,series,train_indices,*,batch_size=8,seed=12):
        import torch
        self.model=model; self.series=series; self.indices=np.asarray(train_indices,dtype=np.int64)
        if not len(self.indices) or batch_size<=0: raise ValueError('empty training split or invalid batch size')
        self.batch_size=batch_size; self.generator=torch.Generator().manual_seed(seed); self.order=None; self.cursor=0; self.global_step=0; self.metrics=[]

    def update(self):
        import torch
        if self.order is None or self.cursor>=len(self.order):
            self.order=torch.randperm(len(self.indices),generator=self.generator).tolist(); self.cursor=0
        selection=self.order[self.cursor:self.cursor+self.batch_size]; self.cursor+=len(selection)
        frame_indices=self.indices[selection]
        batch={'gt':torch.from_numpy(self.series.frames[frame_indices]),'slice_position':torch.from_numpy(self.series.slice_index[frame_indices])}
        values=self.model.training_step(batch,self.global_step+1); self.global_step+=1
        row={'step':self.global_step,**values}; self.metrics.append(row); return row

    def state_dict(self):
        from cardioresp4d.training.runtime_state import capture_rng_state
        return {'step':self.global_step,'networks':{k:v.state_dict() for k,v in self.model.networks().items()},'optimizer':self.model.opt.state_dict(),'scheduler':self.model.sched.state_dict(),'permutation':self.order,'cursor':self.cursor,'generator':self.generator.get_state(),'rng':capture_rng_state(),'metrics':self.metrics}

    def load_state_dict(self,state):
        from cardioresp4d.training.runtime_state import restore_rng_state
        if set(state['networks'])!=set(self.model.networks()): raise ValueError('upstream network schema mismatch')
        for name,net in self.model.networks().items(): net.load_state_dict(state['networks'][name],strict=True)
        self.model.opt.load_state_dict(state['optimizer']); self.model.sched.load_state_dict(state['scheduler']); self.global_step=state['step']; self.model.step=self.global_step
        self.order=state['permutation']; self.cursor=state['cursor']; self.generator.set_state(state['generator']); self.metrics=state['metrics']; restore_rng_state(state['rng'])


def validate_native_config(cfg):
    """Reject any change to the upstream paper-like scientific parameters."""
    expected={'iters':10000,'batch_size':8,'num_points':10000,'lr_hash':.01,'lr_template':2e-5,'lr_encoder':5e-5,'lr_deform':5e-5,'lr_decay_steps':5000,'lr_decay_factor':.5,'seed':6,'loader_seed':12,'template':'ngp','encoder':'vae','code_dim':32,'kl_weight':1e-3,'conditioning':'adaptive','deform_layers':[256]*4,'deform_omega':7,'deform_residual':False,'bidirectional':True,'lambda_cycle':.1,'lambda_jac':1e-4,'lambda_mag':1e-2,'thick_slice':True,'ngp_levels':16,'ngp_features_per_level':2,'ngp_log2_hashmap_size':19,'ngp_base_resolution':16,'ngp_max_resolution':512,'ngp_mlp':[64,64]}
    if any(getattr(cfg,key,None)!=value for key,value in expected.items()): raise ValueError('S0 upstream paper-like configuration mismatch')
    if cfg.n_holdout!=64 or cfg.encoder_channels!=32: raise ValueError('S0 upstream holdout or encoder configuration mismatch')


def validate_native_resume(payload,provenance):
    if payload.get('schema')!='simple4d_s0_v1' or payload.get('provenance')!=provenance or not isinstance(payload.get('runner'),dict): raise ValueError('S0 checkpoint provenance mismatch')


def validate_native_backend(requested,has_tcnn,cuda_available):
    if requested=='tcnn' and (not has_tcnn or not cuda_available): raise RuntimeError('BLOCKED_TCNN_UNAVAILABLE')
    if requested=='pytorch_fallback' and has_tcnn and cuda_available: raise ValueError('explicit fallback requested but upstream would select TCNN')
    if requested not in ('tcnn','pytorch_fallback'): raise ValueError('unknown S0 backend')
