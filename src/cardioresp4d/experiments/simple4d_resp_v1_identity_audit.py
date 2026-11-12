"""Analysis-only building blocks for the SIMPLE-4D respiratory identity audit.

All PCA directions use the largest absolute loading as the positive pivot.
No image-derived quantity in this module is respiratory ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def global_latent_pca(locations):
    """Fit one PCA to location-centered [frames, features] posterior means.

    Locations must be in deterministic order. The RMS and eigenvalue normalization
    match R0's diagnostic implementation (pooled frames, not pooled locations).
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in locations]
    if not arrays or any(a.ndim != 2 or len(a) == 0 or a.shape[1] != arrays[0].shape[1] or not np.isfinite(a).all() for a in arrays):
        raise ValueError('locations must be nonempty finite [frames, features] arrays')
    centered = [a - a.mean(axis=0, keepdims=True) for a in arrays]
    pooled = np.concatenate(centered, axis=0)
    values, basis = np.linalg.eigh(pooled.T @ pooled)
    order = np.argsort(values)[::-1]
    values, basis = np.maximum(values[order], 0), basis[:, order]
    for col in range(basis.shape[1]):
        pivot = np.argmax(np.abs(basis[:, col]))
        if basis[pivot, col] < 0:
            basis[:, col] *= -1
    variance = values.sum()
    ratios = values / variance if variance > 1e-20 else np.zeros_like(values)
    rank95 = int(np.searchsorted(np.cumsum(ratios), .95) + 1) if variance > 1e-20 else 0
    return {'centered':centered, 'scores':[a @ basis for a in centered], 'basis':basis,
            'eigenvalues':values, 'explained_variance_ratios':ratios, 'rank95':rank95,
            'latent_dynamic_rms':float(np.sqrt(np.mean(pooled**2)))}


def image_temporal_pca(images, *, variance_tolerance=1e-12):
    """Return normalized temporal image PC1, or None for negligible variation."""
    images = np.asarray(images, dtype=np.float64)
    if images.ndim < 2 or images.shape[0] < 2 or not np.isfinite(images).all():
        raise ValueError('expected at least two finite images')
    x = images.reshape(len(images), -1)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T / x.shape[1]
    variance = float(np.trace(gram) / len(x))
    if variance <= variance_tolerance:
        return {'scores':None, 'image_temporal_variance':variance, 'excluded_low_variance':True}
    values, vectors = np.linalg.eigh(gram)
    score = vectors[:, -1]
    pivot = np.argmax(np.abs(score))
    if score[pivot] < 0:
        score = -score
    std = float(score.std())
    if std <= variance_tolerance:
        return {'scores':None, 'image_temporal_variance':variance, 'excluded_low_variance':True}
    return {'scores':(score-score.mean())/std, 'image_temporal_variance':variance, 'excluded_low_variance':False}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('correlation inputs must be equal-length finite vectors')
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return {'pearson':None, 'spearman':None}
    return {'pearson':float(np.corrcoef(x, y)[0,1]), 'spearman':float(np.corrcoef(rankdata(x), rankdata(y))[0,1])}


def align_surrogate(model_score, image_score):
    """For display only; preserve the caller's raw surrogate values."""
    score = np.asarray(image_score, dtype=float)
    r = correlation(model_score, score)['pearson']
    return score.copy() * (-1 if r is not None and r < 0 else 1)


def mutual_nearest_pairs(source_times, target_times, threshold_s):
    """Deterministic mutual-nearest matches, one-to-one, on actual timestamps."""
    a, b = np.asarray(source_times, dtype=float), np.asarray(target_times, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all() or threshold_s < 0:
        raise ValueError('invalid timestamps or threshold')
    if not len(a) or not len(b):
        return []
    # searchsorted avoids a potentially huge all-pairs distance matrix.
    def nearest(x, y):
        order = np.argsort(y, kind='stable'); sorted_y = y[order]
        i = np.searchsorted(sorted_y, x, side='left')
        left, right = np.clip(i-1, 0, len(y)-1), np.clip(i, 0, len(y)-1)
        best = np.where(np.abs(x-sorted_y[left]) <= np.abs(x-sorted_y[right]), left, right)
        return order[best]
    ab, ba = nearest(a, b), nearest(b, a)
    return [(i, int(j), float(abs(a[i]-b[j]))) for i,j in enumerate(ab) if ba[j] == i and abs(a[i]-b[j]) <= threshold_s]


def interpolate_neighbor_codes(previous_code, next_code, previous_time, target_time, next_time):
    """V2 simple and timestamp-aware codes, independent of the target code."""
    if not np.isfinite([previous_time, target_time, next_time]).all() or not (previous_time < target_time < next_time) or next_time-previous_time <= 1e-9:
        raise ValueError('target must lie strictly between distinct finite neighbor timestamps')
    previous, following = np.asarray(previous_code), np.asarray(next_code)
    if previous.shape != following.shape or not np.isfinite(previous).all() or not np.isfinite(following).all():
        raise ValueError('neighbor codes must have matching finite shape')
    alpha = (target_time-previous_time)/(next_time-previous_time)
    return {'alpha':float(alpha), 'simple':(previous+following)*.5, 'time':(1-alpha)*previous+alpha*following}


def representative_locations(locations):
    """Select 25%, central and 75% positions without looking at performance."""
    ordered = sorted(locations, key=lambda x:(float(x[1]), str(x[0])))
    if not ordered:
        return []
    indices = [round((len(ordered)-1)*.25), min(range(len(ordered)), key=lambda i:(abs(ordered[i][1]),str(ordered[i][0]))), round((len(ordered)-1)*.75)]
    return list(dict.fromkeys(ordered[i][0] for i in indices))


def coherence_pairs(locations, threshold_s):
    """Match mutually nearest timestamps, avoiding within-view symmetric duplicates."""
    views = ('SAX', '2CH', '4CH')
    output, coverage = {}, {}
    for i, source_view in enumerate(views):
        for target_view in views[i:]:
            key = source_view + '_' + target_view
            source = sorted((loc for loc in locations if loc['view']==source_view), key=lambda x:str(x['slice_id']))
            target = sorted((loc for loc in locations if loc['view']==target_view), key=lambda x:str(x['slice_id']))
            rows, eligible, matched = [], set(), set()
            for first in source:
                for second in target:
                    if source_view == target_view and str(first['slice_id']) >= str(second['slice_id']):
                        continue
                    a, b = first['frames'], second['frames']
                    for frame in a: eligible.add((str(first['slice_id']), frame['dynamic_frame_id']))
                    for ai, bi, dt in mutual_nearest_pairs([f['timestamp_s'] for f in a],[f['timestamp_s'] for f in b],threshold_s):
                        x,y = a[ai],b[bi]
                        if x['pc1_location_z'] is None or y['pc1_location_z'] is None:
                            continue
                        matched.add((str(first['slice_id']),x['dynamic_frame_id']))
                        rows.append({'source_view':source_view,'target_view':target_view,'source_slice_id':first['slice_id'],'target_slice_id':second['slice_id'],'source_frame_id':x['dynamic_frame_id'],'target_frame_id':y['dynamic_frame_id'],'source_timestamp_s':x['timestamp_s'],'target_timestamp_s':y['timestamp_s'],'delta_t_s':dt,'source_pc1_z':x['pc1_location_z'],'target_pc1_z':y['pc1_location_z'],'source_pc1':x['pc1'],'target_pc1':y['pc1']})
            output[key] = rows
            coverage[key] = {'source_frame_count':len(eligible),'matched_source_fraction':len(matched)/len(eligible) if eligible else None}
    return output, coverage


def coherence_summary(rows, coverage, threshold):
    result = {'time_threshold_s':threshold,'pair_count':len(rows),'unique_location_pair_count':len({(r['source_slice_id'],r['target_slice_id']) for r in rows}),**coverage}
    result['coverage_limited'] = len(rows)<3 or coverage['matched_source_fraction'] is None or coverage['matched_source_fraction'] < .1
    fields = ('pearson','spearman','mean_abs_pc1_z_difference','median_abs_pc1_z_difference','rmse_pc1_z_difference','median_abs_delta_t_s','p95_abs_delta_t_s')
    result.update({k:None for k in fields})
    if rows:
        a = np.asarray([r['source_pc1_z'] for r in rows]); b = np.asarray([r['target_pc1_z'] for r in rows]); diff = a-b
        dt = np.asarray([r['delta_t_s'] for r in rows])
        if len(rows)>=2: result.update(correlation(a,b))
        result.update({'mean_abs_pc1_z_difference':float(np.abs(diff).mean()),'median_abs_pc1_z_difference':float(np.median(np.abs(diff))),'rmse_pc1_z_difference':float(np.sqrt(np.mean(diff**2))),'median_abs_delta_t_s':float(np.median(dt)),'p95_abs_delta_t_s':float(np.quantile(dt,.95))})
    return result


# Audit search window only; it is not a physiological ground-truth range.
PERIODICITY_FREQUENCY_HZ = (0.05, 0.6)
PERIODICITY_MIN_FRAMES = 6


def periodicity_analysis(timestamps, values):
    """Timestamp-aware Lomb–Scargle and interpolated, visualization-only ACF."""
    from scipy.signal import lombscargle, find_peaks, correlate
    t=np.asarray(timestamps,dtype=float)
    if values is None:
        values=np.full(len(t),np.nan)
    y=np.asarray(values,dtype=float)
    if t.ndim != 1 or y.shape != t.shape or not np.isfinite(t).all():
        raise ValueError('timestamps and scores must be matching one-dimensional arrays')
    order=np.argsort(t,kind='stable'); t=t[order]; y=y[order]
    relative=t-t[0] if len(t) else t
    result={'frame_count':len(t),'duration_s':float(relative[-1]) if len(t) else 0.,'median_dt_s':float(np.median(np.diff(t))) if len(t)>1 else None,'dominant_frequency_hz':None,'dominant_period_s':None,'dominant_peak_power':None,'autocorrelation_peak_lag_s':None,'excluded':True,'exclusion_reason':None,'detectable_peak':False}
    empty={'relative_time_s':relative,'frequencies_hz':np.array([]),'power':np.array([]),'acf_lags_s':np.array([]),'acf':np.array([])}
    if len(t)<PERIODICITY_MIN_FRAMES: result['exclusion_reason']='too_few_frames'; return result,empty
    if np.any(np.diff(t)<=0): result['exclusion_reason']='non_increasing_timestamps'; return result,empty
    if not np.isfinite(y).all() or np.std(y)<=1e-12: result['exclusion_reason']='missing_or_low_variance'; return result,empty
    if relative[-1] < 1/PERIODICITY_FREQUENCY_HZ[1]: result['exclusion_reason']='duration_too_short'; return result,empty
    frequencies=np.linspace(*PERIODICITY_FREQUENCY_HZ,1024)
    centered=y-y.mean(); power=lombscargle(relative,centered,2*np.pi*frequencies,precenter=False,normalize=True)
    if not np.isfinite(power).all(): result['exclusion_reason']='invalid_spectrum'; return result,empty
    index=int(np.argmax(power)); frequency=float(frequencies[index])
    result.update({'dominant_frequency_hz':frequency,'dominant_period_s':1/frequency,'dominant_peak_power':float(power[index]),'excluded':False,'detectable_peak':bool(len(find_peaks(power)[0]) > 0 and 0 < index < len(power)-1)})
    n=min(4096,max(2,int(np.floor(relative[-1]/result['median_dt_s']))+1))
    uniform=np.linspace(0,relative[-1],n); interp=np.interp(uniform,relative,centered); interp-=interp.mean()
    acf=correlate(interp,interp,mode='full',method='fft')[n-1:]; acf=acf/acf[0] if acf[0]>1e-20 else np.zeros(n)
    lags=uniform-uniform[0]; peaks,_=find_peaks(acf)
    peaks=[p for p in peaks if lags[p]>=1/PERIODICITY_FREQUENCY_HZ[1] and lags[p]<=min(1/PERIODICITY_FREQUENCY_HZ[0],relative[-1]/2) and acf[p]>0]
    if peaks: result['autocorrelation_peak_lag_s']=float(lags[max(peaks,key=lambda p:acf[p])])
    return result,{'relative_time_s':relative,'frequencies_hz':frequencies,'power':power,'acf_lags_s':lags,'acf':acf}
