"""Transparent statistics for native S0 temporal-collapse diagnostics."""
import numpy as np


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64)**2)))


def ratio(a, b):
    return float(a / b) if b > 0 else None


def latent_stats(codes):
    x = np.asarray(codes, dtype=np.float64)
    mean = x.mean(axis=0)
    dynamic = x - mean
    eig = np.linalg.eigvalsh(dynamic.T @ dynamic / len(x))[::-1].clip(min=0)
    total = eig.sum()
    explained = eig / total if total > 0 else np.zeros_like(eig)
    sample=x[np.linspace(0,len(x)-1,min(len(x),2048),dtype=int)]
    from scipy.spatial.distance import pdist
    distances = pdist(sample)

    return {'mean_norm':float(np.linalg.norm(mean)), 'static_rms':rms(mean), 'dynamic_rms':rms(dynamic), 'dynamic_ratio':ratio(rms(dynamic), rms(mean)), 'covariance_eigenvalues':eig.tolist(), 'explained_variance':explained.tolist(), 'pc1_explained_variance':float(explained[0]), 'pc2_explained_variance':float(explained[1]), 'rank95':int(np.searchsorted(np.cumsum(explained), .95)+1) if total > 0 else 0, 'per_dimension_temporal_std':x.std(axis=0).tolist(), 'pairwise_distance_sample_count':len(sample), 'pairwise_distance_sampling':'evenly spaced frames, maximum 2048', 'pairwise_distance_mean':float(distances.mean()) if distances.size else None, 'pairwise_distance_p95':float(np.percentile(distances,95)) if distances.size else None, 'adjacent_distance_mean':float(np.linalg.norm(np.diff(x,axis=0),axis=-1).mean()) if len(x)>1 else None}


def temporal_stats(observed, predicted):
    a=np.asarray(observed,dtype=np.float64); b=np.asarray(predicted,dtype=np.float64)
    if a.shape != b.shape or len(a)<2: raise ValueError('aligned sequences with at least two frames required')
    ma=a.mean(axis=0); mb=b.mean(axis=0); da=a-ma; db=b-mb
    sa=da.std(axis=0); sb=db.std(axis=0); adja=float(np.abs(np.diff(a,axis=0)).mean()); adjb=float(np.abs(np.diff(b,axis=0)).mean())
    denom=np.sqrt((da**2).sum(axis=0)*(db**2).sum(axis=0)); valid=denom>1e-12
    corr=np.zeros_like(denom); corr[valid]=(da*db).sum(axis=0)[valid]/denom[valid]
    return {'observed_mean_temporal_std':float(sa.mean()),'prediction_mean_temporal_std':float(sb.mean()),'rendered_temporal_std_ratio':ratio(sb.mean(),sa.mean()),'observed_p95_temporal_std':float(np.percentile(sa,95)),'prediction_p95_temporal_std':float(np.percentile(sb,95)),'p95_temporal_std_ratio':ratio(np.percentile(sb,95),np.percentile(sa,95)),'observed_mean_adjacent_abs_diff':adja,'prediction_mean_adjacent_abs_diff':adjb,'adjacent_difference_ratio':ratio(adjb,adja),'temporal_mean_image_mse':float(np.mean((ma-mb)**2)),'dynamic_only_mse':float(np.mean((da-db)**2)),'mean_temporal_correlation':float(corr[valid].mean()) if valid.any() else None,'correlation_pixel_count':int(valid.sum())}
