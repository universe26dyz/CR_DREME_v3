"""Comparison of available audit-stage results without running inference."""
from pathlib import Path
import csv
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .simple4d_resp_v1_identity_stages import write_csv,json_write


METRICS=[('reconstruction_improvement','Reconstruction improvement %','higher'),('surrogate','Image-surrogate median |Pearson|','higher'),('coherence','Cross-view coherence at 0.5 s','higher'),('withheld','Target-code-withheld improvement %','higher'),('dynamic_dvf','Dynamic DVF RMS (mm)','neutral'),('static_dynamic','Static/dynamic DVF ratio','lower'),('jacobian','Jacobian p01','higher'),('cycle','Cycle error mean (mm)','lower'),('rank95','Latent rank95','neutral')]


def comparison_report(root,steps,overwrite=False):
    root=Path(root); dest=root/'comparison'
    if not overwrite and (dest.exists() or (root/'audit_report.md').exists() or (root/'audit_manifest.json').exists()): raise FileExistsError('report output exists; pass --overwrite-stage')
    manifest_paths=[p for p in sorted(root.glob('*/stage_manifest.json')) if p.parent.name != 'comparison']
    manifests=[json.loads(p.read_text()) for p in manifest_paths]
    for path,manifest in zip(manifest_paths,manifests):
        for name,digest in manifest.get('output_sha256',{}).items():
            if name.endswith(('.csv','.json')):
                file=path.parent/name
                if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest: raise ValueError('stage output hash mismatch: '+str(file))
    if manifests:
        hashes=manifests[0]['input_artifact_sha256']; checkpoints={}
        for manifest in manifests:
            if manifest['input_artifact_sha256'] != hashes: raise ValueError('stage input provenance mismatch')
            for step, value in manifest['checkpoints'].items():
                if step in checkpoints and checkpoints[step] != value['sha256']: raise ValueError('stage checkpoint provenance mismatch')
                checkpoints[step]=value['sha256']
    import shutil
    if dest.exists(): shutil.rmtree(dest)
    dest.mkdir(parents=True,exist_ok=True)
    available={p.parent.name for p in manifest_paths}
    summary_path=root/'00_summary/checkpoint_summary.json'; summary=json.loads(summary_path.read_text()) if '00_summary' in available and summary_path.exists() else []
    summaries={int(r['step']):r for r in summary}; rows=[]
    for step in steps:
        base=summaries.get(step,{}); row={'step':step,'reconstruction_improvement':base.get('motion_vs_static_mse_improvement_percent'),'dynamic_dvf':base.get('respiratory_dvf_dynamic_rms_mm'),'static_dynamic':base.get('respiratory_dvf_static_dynamic_ratio'),'jacobian':base.get('jacobian_det_p01'),'cycle':base.get('cycle_error_mean_mm'),'rank95':base.get('latent_dynamic_rank_95pct'),'pc1':base.get('latent_dynamic_pc1_explained_variance_ratio'),'pc2':base.get('latent_dynamic_pc2_explained_variance_ratio'),'latent_static_dynamic':base.get('latent_mu_static_dynamic_ratio'),'surrogate':None,'coherence':None,'withheld':None,'dvf_state_rms':None}
        path=root/f'02_surrogate/step_{step}/view_summary.json'
        if '02_surrogate' in available and path.exists(): row['surrogate']=json.loads(path.read_text())['ALL']['median_abs_pearson']
        path=root/f'03_coherence/step_{step}/coherence_summary.json'
        if '03_coherence' in available and path.exists():
            values=[r['pearson'] for r in json.loads(path.read_text()) if r['pair_group'] in ('SAX_2CH','SAX_4CH','2CH_4CH') and r['time_threshold_s']==.5 and not r['coverage_limited'] and r['pearson'] is not None]
            row['coherence']=float(np.median(values)) if values else None
        path=root/f'05_withheld/step_{step}/summary.json'
        if '05_withheld' in available and path.exists(): row['withheld']=json.loads(path.read_text())['ALL']['median_v2_time_vs_static_improvement_percent']
        path=root/f'04_dvf/step_{step}/dvf_state_summary.csv'
        if '04_dvf' in available and path.exists():
            with path.open(newline='') as f: values=[float(r['rms_dvf_mm']) for r in csv.DictReader(f) if r['view']=='SAX']
            row['dvf_state_rms']=float(np.mean(values)) if values else None
        rows.append(row)
    write_csv(dest/'checkpoint_comparison.csv',rows); json_write(dest/'checkpoint_comparison.json',{'rows':rows,'metric_direction':{k:d for k,_,d in METRICS},'composite_score':None,'coherence_policy':'median available cross-view Pearson with sufficient coverage at 0.5s'})
    fig,ax=plt.subplots(figsize=(max(7,len(steps)*2),7)); matrix=np.full((len(METRICS),len(steps)),np.nan)
    for i,(key,_,direction) in enumerate(METRICS):
        vals=np.array([float(r[key]) if r[key] is not None else np.nan for r in rows]); valid=np.isfinite(vals)
        if valid.any():
            lo,hi=np.nanmin(vals),np.nanmax(vals); scaled=np.full(len(vals),.5) if hi==lo else (vals-lo)/(hi-lo)
            if direction=='lower': scaled=1-scaled
            if direction=='neutral': scaled[:]=.5
            matrix[i,valid]=scaled[valid]
    ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap='RdYlGn',vmin=0,vmax=1)
    for i,(key,_,_) in enumerate(METRICS):
        for j,row in enumerate(rows): ax.text(j,i,'—' if row[key] is None else f"{row[key]:.3g}",ha='center',va='center')
    ax.set(xticks=range(len(steps)),xticklabels=steps,yticks=range(len(METRICS)),yticklabels=[m[1] for m in METRICS]); fig.savefig(dest/'01_checkpoint_decision_matrix.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    for filename,key,label in [('02_surrogate_correlation_comparison.png','surrogate','Median image surrogate |r|'),('03_cross_view_coherence_comparison.png','coherence','Median cross-view Pearson'),('04_target_withheld_comparison.png','withheld','V2 vs static improvement (%)'),('05_dvf_state_comparison.png','dvf_state_rms','Mean SAX state DVF RMS (mm)')]:
        fig,ax=plt.subplots(); vals=[r[key] if r[key] is not None else np.nan for r in rows]; ax.plot(steps,vals,marker='o'); ax.set(xlabel='Checkpoint step',ylabel=label); ax.grid(alpha=.2); fig.savefig(dest/filename,dpi=180,bbox_inches='tight'); plt.close(fig)
    lines=['# SIMPLE-4D respiratory identity audit','', 'Measured values below describe the available stages; image PCA is not respiratory ground truth.', '', '| Metric | '+' | '.join(map(str,steps))+' |','|---|'+'---|'*len(steps)]
    for key,label,_ in METRICS: lines.append('| '+label+' | '+' | '.join('—' if r[key] is None else f"{r[key]:.5g}" for r in rows)+' |')
    def best(key,maximum=True):
        valid=[r for r in rows if r[key] is not None]
        return str((max if maximum else min)(valid,key=lambda r:r[key])['step']) if valid else 'insufficient data'
    lines += ['', '## Decision support', '',f"Reconstruction-optimal checkpoint: {best('reconstruction_improvement')}",f"Image-surrogate-optimal checkpoint: {best('surrogate')}",f"Cross-view-coherence-optimal checkpoint: {best('coherence')}",f"Target-code-withheld-optimal checkpoint: {best('withheld')}",f"Lowest cycle error: {best('cycle',False)}; highest Jacobian p01: {best('jacobian')}",'Respiratory-manifold-optimal checkpoint: not automatically assigned; inspect PC1, rank95, latent static/dynamic, surrogate and coherence separately.','Regularity-optimal checkpoint: not automatically assigned; inspect Jacobian and cycle independently.','', '**Recommended authority checkpoint: NOT AUTOMATICALLY ASSIGNED**', '', 'Target-code-withheld is an in-training-set diagnostic, not held-out generalization. Temporal coherence is limited by observed overlap.']
    (root/'audit_report.md').write_text('\n'.join(lines)+'\n')
    files=sorted(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file())
    json_write(dest/'stage_manifest.json',{'stage':'report','timestamp':datetime.now(timezone.utc).isoformat(),'git_sha':manifests[0].get('git_sha') if manifests else None,'input_artifact_sha256':manifests[0]['input_artifact_sha256'] if manifests else {},'checkpoints':{step:{'sha256':digest} for step,digest in checkpoints.items()} if manifests else {},'output_files':files+['stage_manifest.json','stage_report.json'],'output_sha256':{name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in files},'analysis_policy':'available validated stage data; no composite score'})
    json_write(dest/'stage_report.json',{'stage':'report','available_checkpoints':steps,'recommended_authority_checkpoint':None})
    json_write(root/'audit_manifest.json',{'steps':steps,'available_stage_manifests':{p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/stage_manifest.json'))},'comparison':str(dest),'recommended_authority_checkpoint':None})
    return rows
