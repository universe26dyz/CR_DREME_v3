"""CSV-only respiratory and cardiac-candidate spectral audit."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
from scipy.signal import lombscargle

BANDS={'respiratory':(.10,.60),'cardiac_candidate':(.80,2.00)}
SIGNALS=('pc1','tx_mm','ty_mm','tz_mm','translation_magnitude_mm','residual_rms_mm','jacobian_mean_abs_deviation')


def analyze_trace(times,values):
    t=np.asarray(times,dtype=float); y=np.asarray(values,dtype=float)
    if t.ndim!=1 or y.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(y).all(): raise ValueError('timestamps and values must be matching finite vectors')
    order=np.argsort(t,kind='stable'); t=t[order]; y=y[order]; n=len(t)
    dt=np.diff(t); median=float(np.median(dt)) if len(dt) else None
    result={'n_samples':n,'duration_s':float(t[-1]-t[0]) if n else 0.,'median_dt_s':median,'effective_sampling_hz':1/median if median and median>0 else None,'estimated_nyquist_hz':.5/median if median and median>0 else None,'bands':{}}
    for name,(lo,hi) in BANDS.items():
        out={'available':False,'reason':None,'dominant_frequency_hz':None,'dominant_period_s':None,'peak_power':None,'normalized_peak_power':None,'mean_normalized_band_power':None}
        if n<6: out['reason']='too_few_samples'
        elif np.any(dt<=0): out['reason']='non_increasing_timestamps'
        elif np.std(y)<=1e-12: out['reason']='low_variance'
        elif result['estimated_nyquist_hz']<hi: out['reason']='band_exceeds_estimated_nyquist'
        elif result['duration_s']<1/hi: out['reason']='duration_shorter_than_fastest_band_period'
        else:
            f=np.linspace(lo,hi,1024); centered=y-y.mean(); raw=lombscargle(t-t[0],centered,2*np.pi*f,normalize=False); norm=lombscargle(t-t[0],centered,2*np.pi*f,normalize=True); i=int(np.argmax(norm))
            out.update({'available':True,'dominant_frequency_hz':float(f[i]),'dominant_period_s':float(1/f[i]),'peak_power':float(raw[i]),'normalized_peak_power':float(norm[i]),'mean_normalized_band_power':float(norm.mean())})
        result['bands'][name]=out
    r=result['bands']['respiratory']['normalized_peak_power']; c=result['bands']['cardiac_candidate']['normalized_peak_power']
    result['cardiac_candidate_to_respiratory_peak_power_ratio']=c/r if c is not None and r is not None and r>0 else None
    return result


def frequency_stage(audit_root,steps,*,overwrite=False):
    """Read existing motion CSV only; never load a model or checkpoint."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(audit_root)/'07_cardioresp_identity'; collected={}
    targets=[root/f'step_{step}'/'motion_decomposition/frequency_band_summary.json' for step in steps]+[root/'comparisons/frequency_band_comparison.json']
    if not overwrite and any(path.exists() for path in targets): raise FileExistsError('frequency output exists; use --overwrite-stage')
    for step in steps:
        folder=root/f'step_{step}'/'motion_decomposition'; source=folder/'frame_motion_decomposition.csv'; target=folder/'frequency_band_summary.json'
        if target.exists() and not overwrite: raise FileExistsError(f'{target} exists; use --overwrite-stage')
        with source.open(newline='') as f: rows=list(csv.DictReader(f))
        traces={}
        for region in sorted({r['region'] for r in rows}):
            region_rows=sorted((r for r in rows if r['region']==region),key=lambda r:float(r['timestamp_s']))
            keys=[(r['slice_id'],r['dynamic_frame_id'],r['timestamp_s']) for r in region_rows]
            if len(keys)!=len(set(keys)) or len({r['slice_id'] for r in region_rows})!=1: raise ValueError('ambiguous location/frame trace')
            traces[region]={}
            for signal in SIGNALS:
                if region_rows and signal in region_rows[0] and all(r.get(signal,'')!='' for r in region_rows):
                    traces[region][signal]=analyze_trace([float(r['timestamp_s']) for r in region_rows],[float(r[signal]) for r in region_rows])
        result={'checkpoint_step':step,'source_csv':str(source.resolve()),'source_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'audit_root':str(Path(audit_root).resolve()),'settings':{'method':'scipy.signal.lombscargle','normalized':True,'frequency_bins':1024,'timestamp_column':'timestamp_s'},'bands_hz':BANDS,'interpretation':{'respiratory_band':'physiologically plausible respiratory-frequency search band','cardiac_candidate_band':'candidate faster-motion search band; a peak does not establish cardiac identity'},'regions':traces}
        target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); collected[step]=result
        flat=[]
        for region,signals in traces.items():
            for signal,trace in signals.items():
                row={'region':region,'signal':signal,'source_csv':result['source_csv'],'source_csv_sha256':result['source_csv_sha256'],'checkpoint_step':step,**{k:trace[k] for k in ('n_samples','duration_s','median_dt_s','effective_sampling_hz','estimated_nyquist_hz','cardiac_candidate_to_respiratory_peak_power_ratio')}}
                for name,band in trace['bands'].items():
                    for key,value in band.items(): row[name+'_'+key]=value
                flat.append(row)
        if flat:
            with (folder/'frequency_band_summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        for label,signals in [('pc1',('pc1',)),('translation',('tz_mm','translation_magnitude_mm')),('nonrigid',('residual_rms_mm','jacobian_mean_abs_deviation'))]:
            fig,ax=plt.subplots()
            group=[r for r in rows if r['region']=='anatomy_support']
            for signal in signals:
                if len(group)<6 or signal not in group[0]: continue
                t=np.array([float(r['timestamp_s']) for r in group]); y=np.array([float(r[signal]) for r in group]); analysis=analyze_trace(t,y)
                if np.std(y)<=1e-12 or np.any(np.diff(np.sort(t))<=0): continue
                upper=min(2.,analysis['estimated_nyquist_hz'] or 0.)
                if upper<=.1: continue
                f=np.linspace(.1,upper,1200); power=lombscargle(t-t.min(),y-y.mean(),2*np.pi*f,normalize=True); ax.plot(f,power,label=signal)
                for band in BANDS:
                    data=analysis['bands'][band]
                    if data['available']: ax.scatter(data['dominant_frequency_hz'],data['normalized_peak_power'],s=18)
            ax.axvspan(.1,.6,alpha=.12,color='C0'); ax.axvspan(.8,2.,alpha=.12,color='C1'); ax.set(xlim=(.1,2.),xlabel='Frequency (Hz)',ylabel='Normalized Lomb–Scargle power',title=f'{step} — {label}')
            if ax.lines: ax.legend();
            fig.savefig(folder/f'frequency_bands_{label}.png',dpi=180,bbox_inches='tight',metadata={'SourceCSV':result['source_csv'],'SourceSHA256':result['source_csv_sha256']}); plt.close(fig)
    comparison=root/'comparisons'; comparison.mkdir(parents=True,exist_ok=True)
    table=[]
    for step,data in collected.items():
        for region,signals in data['regions'].items():
            for signal,result in signals.items():
                row={'step':step,'region':region,'signal':signal,**{k:result[k] for k in ('n_samples','duration_s','median_dt_s','effective_sampling_hz','estimated_nyquist_hz','cardiac_candidate_to_respiratory_peak_power_ratio')}}
                for name,values in result['bands'].items():
                    for key,value in values.items(): row[name+'_'+key]=value
                table.append(row)
    sources={str(step):{'path':str((root/f'step_{step}'/'motion_decomposition/frequency_band_summary.json').resolve()),'sha256':hashlib.sha256((root/f'step_{step}'/'motion_decomposition/frequency_band_summary.json').read_bytes()).hexdigest()} for step in collected}
    for row in table:
        source=sources[str(row['step'])]; row['source_summary_path']=source['path']; row['source_summary_sha256']=source['sha256']
    comparison_file=comparison/'frequency_band_comparison.json'
    if comparison_file.exists() and not overwrite: raise FileExistsError(f'{comparison_file} exists; use --overwrite-stage')
    comparison_file.write_text(json.dumps({'sources':sources,'rows':table,'composite_score':None},indent=2,allow_nan=False)+'\n')
    if table:
        with (comparison/'frequency_band_comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    fig,axes=plt.subplots(5,1,figsize=(10,18))
    fields=('respiratory_dominant_frequency_hz','respiratory_normalized_peak_power','cardiac_candidate_dominant_frequency_hz','cardiac_candidate_normalized_peak_power','cardiac_candidate_to_respiratory_peak_power_ratio')
    titles=('Respiratory peak frequency (Hz)','Respiratory normalized peak power','Cardiac-candidate peak frequency (Hz)','Cardiac-candidate normalized peak power','Candidate / respiratory peak power')
    for signal in SIGNALS:
        rows=[row for row in table if row['region']=='anatomy_support' and row['signal']==signal]
        if not rows: continue
        for ax,field,title in zip(axes,fields,titles):
            ax.plot([r['step'] for r in rows],[r[field] if r[field] is not None else np.nan for r in rows],marker='o',label=signal)
            ax.set_title(title); ax.set_xlabel('Checkpoint step')
    for ax in axes:
        if ax.lines: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(comparison/'frequency_band_comparison.png',dpi=180,bbox_inches='tight',metadata={'SourceSummaries':json.dumps(sources,sort_keys=True)}); plt.close(fig)
    return collected
