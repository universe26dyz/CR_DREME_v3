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
