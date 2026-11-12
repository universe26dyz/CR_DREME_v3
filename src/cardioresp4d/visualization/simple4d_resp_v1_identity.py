"""Headless figures for the SIMPLE-4D respiratory identity audit."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import representative_locations


def save_summary_figures(rows, output):
    steps = [r['step'] for r in rows]
    panels = [
        ('01_reconstruction_vs_step.png', [('static_baseline_sampled_mse_full','static MSE'),('diagnostic_sampled_reconstruction_mse_full','motion MSE')], [('motion_vs_static_mse_improvement_percent','improvement %')]),
        ('02_latent_structure_vs_step.png', [('latent_dynamic_pc1_explained_variance_ratio','PC1'),('latent_dynamic_pc2_explained_variance_ratio','PC2')], [('latent_dynamic_rank_95pct','rank95'),('latent_mu_static_dynamic_ratio','static/dynamic')]),
        ('03_dvf_components_vs_step.png', [('respiratory_dvf_total_rms_mm','total mm'),('respiratory_dvf_static_rms_mm','static mm'),('respiratory_dvf_dynamic_rms_mm','dynamic mm')], [('respiratory_dvf_static_dynamic_ratio','static/dynamic')]),
        ('04_regularity_vs_step.png', [('jacobian_det_p01','Jacobian p01'),('jacobian_det_min','Jacobian min')], [('cycle_error_mean_mm','cycle mean mm'),('cycle_error_p95_mm','cycle p95 mm')]),
    ]
    for filename, left, right in panels:
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        for ax, curves in zip(axes, (left,right)):
            for key,label in curves:
                ax.plot(steps,[r.get(key) for r in rows],marker='o',label=label)
            ax.set_xlabel('checkpoint step'); ax.grid(alpha=.25); ax.legend()
        fig.savefig(output/filename,dpi=180,bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, key, title in zip(axes, ('cycle_error_mean_mm','jacobian_det_p01'), ('Cycle mean (mm)','Jacobian p01')):
        for row in rows:
            ax.scatter(row['motion_vs_static_mse_improvement_percent'],row[key]); ax.annotate(str(row['step']),(row['motion_vs_static_mse_improvement_percent'],row[key]))
        ax.set(xlabel='Motion vs static MSE improvement (%)',ylabel=title); ax.grid(alpha=.25)
    fig.savefig(output/'05_reconstruction_regularity_tradeoff.png',dpi=180,bbox_inches='tight'); plt.close(fig)


def save_latent_figures(rows, ratios, output):
    fig, ax = plt.subplots(); ax.bar(np.arange(1,len(ratios)+1),ratios); ax.set(xlabel='Principal component', ylabel='Explained variance ratio'); fig.savefig(output/'01_pca_scree.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    views = ('SAX','2CH','4CH')
    for pc in (1,2):
        fig, axes = plt.subplots(3,1,figsize=(10,9),sharex=True)
        for ax, view in zip(axes, views):
            selected = [r for r in rows if r['view'] == view]
            location_ids = sorted({r['slice_id'] for r in selected})
            positions = [(location, next(r['normalized_slice_position'] for r in selected if r['slice_id']==location)) for location in location_ids]
            highlighted = set(representative_locations(positions))
            for location in location_ids:
                group = sorted((r for r in selected if r['slice_id']==location),key=lambda r:r['timestamp_s'])
                is_highlight = location in highlighted
                ax.plot([r['timestamp_s'] for r in group],[r[f'pc{pc}'] for r in group],'.-',alpha=.9 if is_highlight else .2,lw=1.2 if is_highlight else .5,label=location if is_highlight else None)
            if highlighted: ax.legend(fontsize=7)
            ax.set_ylabel(f'{view} PC{pc}'); ax.grid(alpha=.2)
        axes[-1].set_xlabel('Acquisition timestamp (s)')
        fig.savefig(output/f'0{pc+1}_pc{pc}_time_by_view.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    groups = []
    for view in views:
        for location in sorted({r['slice_id'] for r in rows if r['view']==view}):
            group = sorted((r for r in rows if r['view']==view and r['slice_id']==location),key=lambda r:r['timestamp_s'])
            values = [r['pc1'] for r in group]
            groups.append(np.interp(np.linspace(0,1,100),np.linspace(0,1,len(values)),values))
    fig,ax=plt.subplots(figsize=(10,6)); im=ax.imshow(np.asarray(groups),aspect='auto',origin='lower',extent=(0,1,0,len(groups)),cmap='coolwarm'); fig.colorbar(im,ax=ax,label='PC1'); ax.set(xlabel='Normalized within-location time (not simultaneity)',ylabel='Location index'); fig.savefig(output/'04_pc1_location_heatmap.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots()
    for view,marker in zip(views,('o','s','^')):
        selected=[r for r in rows if r['view']==view]; ax.scatter([r['pc1'] for r in selected],[r['pc2'] for r in selected],s=8,marker=marker,alpha=.5,label=view)
    ax.set(xlabel='Centered latent PC1',ylabel='Centered latent PC2'); ax.legend(); fig.savefig(output/'05_pc1_pc2_scatter.png',dpi=180,bbox_inches='tight'); plt.close(fig)


def save_periodicity_figures(rows, output):
    """Write deterministic location-level temporal audit and figure backing data."""
    import csv, json
    from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import periodicity_analysis, PERIODICITY_FREQUENCY_HZ
    groups={}
    for row in rows: groups.setdefault((row['view'],row['slice_id']),[]).append(row)
    details=[]; series={}
    for (view,location),group in sorted(groups.items()):
        group=sorted(group,key=lambda r:r['timestamp_s']); t=[r['timestamp_s'] for r in group]
        z=[float(r['pc1_location_z']) if r['pc1_location_z'] is not None else float('nan') for r in group]
        result,arrays=periodicity_analysis(t,z)
        details.append({'view':view,'slice_id':location,'normalized_slice_position':group[0]['normalized_slice_position'],**result})
        series[(view,location)]=(group,result,arrays)
    with (output/'periodicity_by_location.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    summaries={}
    for view in ('SAX','2CH','4CH','ALL'):
        selected=[r for r in details if view=='ALL' or r['view']==view]; valid=[r for r in selected if not r['excluded']]
        def stat(key,q):
            vals=[r[key] for r in valid if r[key] is not None]
            return float(np.quantile(vals,q)) if vals else None
        summaries[view]={'location_count':len(selected),'valid_location_count':len(valid),'median_dominant_frequency_hz':stat('dominant_frequency_hz',.5),'frequency_iqr_hz':[stat('dominant_frequency_hz',.25),stat('dominant_frequency_hz',.75)],'median_dominant_period_s':stat('dominant_period_s',.5),'period_iqr_s':[stat('dominant_period_s',.25),stat('dominant_period_s',.75)],'median_peak_strength':stat('dominant_peak_power',.5),'fraction_with_detectable_peak':sum(r['detectable_peak'] for r in valid)/len(valid) if valid else None}
    (output/'periodicity_summary.json').write_text(json.dumps({'search_frequency_hz':list(PERIODICITY_FREQUENCY_HZ),'interpretation':'periodicity evidence only; not respiratory ground truth','views':summaries},indent=2,allow_nan=False)+'\n')
    for view_index,view in enumerate(('SAX','2CH','4CH')):
        choices=representative_locations([(r['slice_id'],r['normalized_slice_position']) for r in details if r['view']==view])
        for plot_index,kind in enumerate(('relative_time','periodogram','autocorrelation')):
            fig,axes=plt.subplots(max(1,len(choices)),1,figsize=(9,3*max(1,len(choices))),squeeze=False)
            for ax,location in zip(axes[:,0],choices):
                group,info,arrays=series[(view,location)]
                if kind=='relative_time':
                    values=[r['pc1_location_z'] if r['pc1_location_z'] is not None else np.nan for r in group]
                    ax.plot(arrays['relative_time_s'],values,'.-'); ax.set(xlabel='Relative time (s)',ylabel='PC1 location z')
                elif kind=='periodogram':
                    ax.plot(arrays['frequencies_hz'],arrays['power']); ax.set(xlabel='Frequency (Hz)',ylabel='Normalized Lomb–Scargle power')
                    if info['dominant_frequency_hz'] is not None:
                        ax.axvline(info['dominant_frequency_hz'],color='C3',linestyle='--',label=f"{info['dominant_frequency_hz']:.3f} Hz / {info['dominant_period_s']:.2f} s"); ax.legend()
                else:
                    ax.plot(arrays['acf_lags_s'],arrays['acf']); ax.set(xlabel='Lag (s)',ylabel='Normalized ACF')
                    if info['autocorrelation_peak_lag_s'] is not None: ax.axvline(info['autocorrelation_peak_lag_s'],color='C3',linestyle='--')
                ax.set_title(f"{view}/{location} position={group[0]['normalized_slice_position']:.3f} N={info['frame_count']} duration={info['duration_s']:.1f}s" + (f" — excluded: {info['exclusion_reason']}" if info['excluded'] else ''))
                ax.grid(alpha=.2)
                np.savez_compressed(output/f'periodicity_{view}_{str(location).replace("/","_")}.npz',**arrays)
            number=6+view_index+3*plot_index
            fig.savefig(output/f'{number:02d}_pc1_{kind}_{view}.png',dpi=180,bbox_inches='tight'); plt.close(fig)
