"""Stage-level surrogate and temporal coherence analyses."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .simple4d_resp_v1_identity_audit import (image_temporal_pca, correlation, align_surrogate, representative_locations, coherence_pairs, coherence_summary)


def write_csv(path, rows, fields=None):
    if fields is None:
        if not rows: raise ValueError('empty CSV requires an explicit schema')
        fields = list(rows[0])
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def json_write(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def latent_locations(rows):
    locations={}
    for row in rows:
        key=(row['view'],row['slice_id'])
        locations.setdefault(key,[]).append(row)
    return [{'view':v,'slice_id':s,'frames':sorted(group,key=lambda r:(r['timestamp_s'],r['dynamic_frame_id']))} for (v,s),group in sorted(locations.items())]


def read_latent(path, cohort):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    by_id={(view,str(slice_id),int(frame.dynamic_frame_id)):frame for view,slice_id,frames in cohort for frame in frames}
    result=[]
    for row in rows:
        key=(row['view'],row['slice_id'],int(row['dynamic_frame_id']))
        if key not in by_id: raise ValueError('latent frame does not match input cohort')
        frame=by_id.pop(key)
        if abs(float(row['timestamp_s'])-frame.timestamp_s)>1e-8: raise ValueError('latent timestamp mismatch')
        item={**row,'dynamic_frame_id':key[2],'timestamp_s':float(row['timestamp_s']),'normalized_slice_position':float(row['normalized_slice_position']),'pc1':float(row['pc1']),'pc1_location_z':float(row['pc1_location_z']) if row['pc1_location_z'] else None,'observation':frame}
        result.append(item)
    if by_id: raise ValueError('latent frame cohort is incomplete')
    return result


def surrogate_stage(rows, dest):
    dest.mkdir(parents=True)
    locations=latent_locations(rows); records=[]; waveforms={}
    for loc in locations:
        frames=loc['frames']; obs=[x['observation'] for x in frames]
        pca=image_temporal_pca(np.stack([o.image.detach().cpu().numpy() for o in obs])) if len(obs)>=2 else {'scores':None,'image_temporal_variance':0.,'excluded_low_variance':True}
        scores=pca['scores']; model=np.array([f['pc1'] for f in frames]); corr=correlation(model,scores) if scores is not None else {'pearson':None,'spearman':None}
        record={'view':loc['view'],'slice_id':loc['slice_id'],'normalized_slice_position':frames[0]['normalized_slice_position'],'frame_count':len(frames),'image_temporal_variance':pca['image_temporal_variance'],'excluded_low_variance':pca['excluded_low_variance'],'pearson_r':corr['pearson'],'abs_pearson_r':abs(corr['pearson']) if corr['pearson'] is not None else None,'spearman_rho':corr['spearman'],'abs_spearman_rho':abs(corr['spearman']) if corr['spearman'] is not None else None}
        records.append(record)
        if scores is not None:
            waveforms[(loc['view'],loc['slice_id'])]=(frames,scores,model)
    write_csv(dest/'location_correlations.csv',records)
    wave_rows=[]
    for (view,slice_id),(frames,scores,model) in waveforms.items():
        for frame,image_pc,pc in zip(frames,scores,model):
            wave_rows.append({'view':view,'slice_id':slice_id,'dynamic_frame_id':frame['dynamic_frame_id'],'timestamp_s':frame['timestamp_s'],'image_pc1_z_raw':float(image_pc),'model_pc1':float(pc),'model_pc1_location_z':frame['pc1_location_z']})
    write_csv(dest/'frame_surrogate_scores.csv',wave_rows,['view','slice_id','dynamic_frame_id','timestamp_s','image_pc1_z_raw','model_pc1','model_pc1_location_z'])
    summary={}
    for view in ('SAX','2CH','4CH','ALL'):
        selected=[r for r in records if view=='ALL' or r['view']==view]; valid=[r for r in selected if r['abs_pearson_r'] is not None]
        p=[r['abs_pearson_r'] for r in valid]; s=[r['abs_spearman_rho'] for r in valid if r['abs_spearman_rho'] is not None]
        q=lambda a,x:float(np.quantile(a,x)) if a else None
        summary[view]={'location_count':len(selected),'valid_location_count':len(valid),'excluded_low_variance_count':sum(r['excluded_low_variance'] for r in selected),'median_abs_pearson':q(p,.5),'q25_abs_pearson':q(p,.25),'q75_abs_pearson':q(p,.75),'median_abs_spearman':q(s,.5),'q25_abs_spearman':q(s,.25),'q75_abs_spearman':q(s,.75),'fraction_abs_pearson_ge_0p5':sum(x>=.5 for x in p)/len(p) if p else None,'fraction_abs_pearson_ge_0p7':sum(x>=.7 for x in p)/len(p) if p else None}
    json_write(dest/'view_summary.json',summary)
    fig,ax=plt.subplots(figsize=(7,4))
    for i,v in enumerate(('SAX','2CH','4CH'),1):
        vals=[r['abs_pearson_r'] for r in records if r['view']==v and r['abs_pearson_r'] is not None]
        if vals:
            ax.boxplot([vals],positions=[i],widths=.4); ax.scatter(np.full(len(vals),i)+np.linspace(-.14,.14,len(vals)),vals,s=12,alpha=.6)
    ax.set(xticks=[1,2,3],xticklabels=['SAX','2CH','4CH'],ylabel='Absolute Pearson r',ylim=(0,1.02)); fig.savefig(dest/'01_abs_correlation_by_view.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ordered=sorted((r for r in records if r['abs_pearson_r'] is not None),key=lambda x:x['abs_pearson_r'])
    for i,r in enumerate(ordered): ax.scatter(i,r['abs_pearson_r'],c={'SAX':'C0','2CH':'C1','4CH':'C2'}[r['view']],s=10)
    ax.set(xlabel='Location rank',ylabel='Absolute Pearson r',ylim=(0,1.02)); fig.savefig(dest/'02_location_correlation_rank.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    for num,view in enumerate(('SAX','2CH','4CH'),3):
        choices=representative_locations([(r['slice_id'],r['normalized_slice_position']) for r in records if r['view']==view])
        fig,axes=plt.subplots(max(1,len(choices)),1,figsize=(9,3*max(1,len(choices))),squeeze=False)
        for ax,location in zip(axes[:,0],choices):
            item=waveforms.get((view,location))
            if item is None: ax.text(.5,.5,'Insufficient image variation',ha='center'); continue
            frames,image,model=item; zstd=np.std(model)
            if zstd<=1e-12: ax.text(.5,.5,'Insufficient latent variation',ha='center'); continue
            ax.plot([f['timestamp_s'] for f in frames],(model-model.mean())/zstd,label='model PC1 z')
            ax.plot([f['timestamp_s'] for f in frames],align_surrogate(model,image),label='image PC1 z (display-aligned)'); ax.set(title=f'{view} / {location}',xlabel='Timestamp (s)'); ax.legend()
        fig.savefig(dest/f'0{num}_waveform_overlays_{view}.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    return summary


def coherence_stage(rows, dest):
    dest.mkdir(parents=True); locations=latent_locations(rows); summaries=[]; all_pairs={}
    fields=['source_view','target_view','source_slice_id','target_slice_id','source_frame_id','target_frame_id','source_timestamp_s','target_timestamp_s','delta_t_s','source_pc1_z','target_pc1_z','source_pc1','target_pc1']
    for threshold,label in ((.25,'0p25'),(.5,'0p50'),(1.,'1p00')):
        pairs,coverage=coherence_pairs(locations,threshold); all_pairs[threshold]=pairs
        write_csv(dest/f'pairs_{label}s.csv',[r for group in pairs.values() for r in group],fields)
        for name, group in pairs.items(): summaries.append({'pair_group':name,**coherence_summary(group,coverage[name],threshold)})
    write_csv(dest/'coherence_summary.csv',summaries); json_write(dest/'coherence_summary.json',summaries)
    for num,field,title in ((1,'pearson','Pearson r'),(2,'mean_abs_pc1_z_difference','Mean absolute PC1 z difference'),(6,'matched_source_fraction','Matched source fraction')):
        fig,ax=plt.subplots(figsize=(7,4))
        for group in ('SAX_2CH','SAX_4CH','2CH_4CH'):
            data=[r for r in summaries if r['pair_group']==group]; ax.plot([r['time_threshold_s'] for r in data],[r[field] if r[field] is not None else np.nan for r in data],marker='o',label=group)
        ax.set(xlabel='Max |Δt| (s)',ylabel=title); ax.legend(); ax.grid(alpha=.2)
        name={1:'01_cross_view_correlation_vs_dt.png',2:'02_cross_view_abs_difference_vs_dt.png',6:'06_pair_coverage.png'}[num]; fig.savefig(dest/name,dpi=180,bbox_inches='tight'); plt.close(fig)
    for number,group in enumerate(('SAX_2CH','SAX_4CH','2CH_4CH'),3):
        pairs=all_pairs[.5][group]; info=next(r for r in summaries if r['pair_group']==group and r['time_threshold_s']==.5)
        fig,ax=plt.subplots(figsize=(5,5))
        if len(pairs)<3:
            ax.text(.5,.5,'Insufficient temporal overlap',ha='center',transform=ax.transAxes)
        else:
            ax.scatter([r['source_pc1_z'] for r in pairs],[r['target_pc1_z'] for r in pairs],s=8,alpha=.5)
            bound=max(abs(ax.get_xlim()[0]),abs(ax.get_xlim()[1]),abs(ax.get_ylim()[0]),abs(ax.get_ylim()[1])); ax.plot([-bound,bound],[-bound,bound],'k--',lw=1)
        ax.set(xlabel='Source PC1 z',ylabel='Target PC1 z',title=f"{group} N={len(pairs)} r={info['pearson']} rho={info['spearman']} median Δt={info['median_abs_delta_t_s']}")
        fig.savefig(dest/f'0{number}_{group}_scatter_0p5s.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    return summaries
