"""Read-only cardiorespiratory identity audit primitives.

PC1, non-PC1 and non-rigid motion are descriptive proxies, not cardiac phase.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


def pure_pc1_codes(means, scores, basis):
    """Add only global within-location PC1 to the anchor's posterior mean."""
    means=np.asarray(means,dtype=np.float64); scores=np.asarray(scores,dtype=np.float64); basis=np.asarray(basis,dtype=np.float64)
    if means.ndim!=2 or scores.shape!=(len(means),) or basis.shape!=(means.shape[1],) or not np.isfinite(means).all() or not np.isfinite(scores).all() or not np.isfinite(basis).all() or not np.isclose(np.linalg.norm(basis),1.,atol=1e-6):
        raise ValueError('invalid posterior means, PC1 scores or unit PC1 basis')
    return means.mean(0,keepdims=True)+scores[:,None]*basis[None,:]


def difference_summary(a,b,mask=None):
    delta=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    if mask is not None: delta=delta[np.asarray(mask,dtype=bool)]
    if not delta.size: return None
    absval=np.abs(delta)
    return {'mae':float(absval.mean()),'rmse':float(np.sqrt(np.mean(delta**2))),'p95_abs_difference':float(np.quantile(absval,.95))}


def vector_difference_summary(a,b,mask=None):
    delta=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    if mask is not None: delta=delta[np.asarray(mask,dtype=bool)]
    if not delta.size: return None
    magnitude=np.linalg.norm(delta,axis=-1)
    return {'mean_magnitude_mm':float(magnitude.mean()),'rms_magnitude_mm':float(np.sqrt(np.mean(magnitude**2))),'p95_magnitude_mm':float(np.quantile(magnitude,.95)),'max_magnitude_mm':float(magnitude.max())}


def decompose_motion(dvf,determinant,mask=None):
    field=np.asarray(dvf,dtype=np.float64).reshape(-1,3); det=np.asarray(determinant,dtype=np.float64).reshape(-1)
    if len(field)!=len(det) or not np.isfinite(field).all() or not np.isfinite(det).all(): raise ValueError('invalid field or Jacobian')
    if mask is not None:
        mask=np.asarray(mask,dtype=bool).reshape(-1); field=field[mask]; det=det[mask]
    if not len(det): return None
    translation=field.mean(0); residual=np.linalg.norm(field-translation,axis=1); total=np.sqrt(np.mean(np.sum(field**2,axis=1)))
    return {'tx_mm':float(translation[0]),'ty_mm':float(translation[1]),'tz_mm':float(translation[2]),'translation_magnitude_mm':float(np.linalg.norm(translation)),'total_dvf_rms_mm':float(total),'residual_rms_mm':float(np.sqrt(np.mean(residual**2))),'residual_p95_mm':float(np.quantile(residual,.95)),'residual_max_mm':float(residual.max()),'nonrigid_fraction':float(np.sqrt(np.mean(residual**2))/max(total,1e-12)),'jacobian_mean':float(det.mean()),'jacobian_median':float(np.median(det)),'jacobian_p01':float(np.quantile(det,.01)),'jacobian_p05':float(np.quantile(det,.05)),'jacobian_p95':float(np.quantile(det,.95)),'jacobian_negative_fraction':float(np.mean(det<0)),'jacobian_mean_abs_deviation':float(np.abs(det-1).mean())}


def residual_metrics(target,prediction,mask=None):
    a=np.asarray(target,dtype=np.float64).reshape(-1); b=np.asarray(prediction,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('invalid target/prediction')
    if mask is not None: mask=np.asarray(mask,dtype=bool).reshape(-1); a=a[mask]; b=b[mask]
    if len(a)==0: return None
    error=a-b; aa=a-a.mean(); bb=b-b.mean(); denominator=np.linalg.norm(aa)*np.linalg.norm(bb)
    return {'mse':float(np.mean(error**2)),'mae':float(np.abs(error).mean()),'ncc':float(np.dot(aa,bb)/denominator) if denominator>1e-12 else None,'pixel_count':len(a)}


def timestamp_interpolated_code(previous,following,t_previous,t_target,t_following):
    from .simple4d_resp_v1_identity_audit import interpolate_neighbor_codes
    return interpolate_neighbor_codes(previous,following,t_previous,t_target,t_following)
