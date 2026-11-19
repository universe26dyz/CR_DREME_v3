#!/usr/bin/env python3
"""Audit timestamp support in the corrected S05PROP manifest/QC groups."""
import argparse, csv, json, hashlib, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cardioresp4d.training.sampler import proportional_view_counts
from cardioresp4d.training.runtime_state import is_hard_invalid_reason

def stats(a):
    return {k: float(v) for k, v in zip(('min','p10','median','p90','max'), np.percentile(a, [0,10,50,90,100]))} if a else None

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--qc-table',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    with a.qc_table.open() as f: qc={r['source_file_token']:r for r in csv.DictReader(f)}
    groups=defaultdict(list); missing=defaultdict(int)
    with a.manifest.open() as f:
        for r in csv.DictReader(f):
            q=qc[r['source_file_token']]
            if q['qc_valid'].lower() in ('0','false') or is_hard_invalid_reason(q['qc_reason']): continue
            key=(r['view'].upper(),r['slice_id'])
            if not r['timestamp_s'].strip(): missing[key]+=1; continue
            groups[key].append((int(r['frame_index']),float(r['timestamp_s'])))
    loc_counts={v:sum(k[0]==v for k in groups) for v in ('SAX','2CH','4CH')}
    quotas=proportional_view_counts(loc_counts,24); out=[]
    for (view,loc), frames in sorted(groups.items()):
        seq=[t for _,t in sorted(frames)]; dt=np.diff(seq); positives=dt[dt>0]; span=max(seq)-min(seq); med=float(np.median(positives)) if len(positives) else None
        nyquist=1/(2*med) if med else None
        # >= one period at the lower band edge, >= 8 samples, and Nyquist above upper edge.
        def supported(lo,hi): return len(seq)>=8 and span>=1/lo and nyquist is not None and nyquist>=hi and bool(np.all(dt>0))
        out.append(dict(view=view,slice_id=loc,observations=len(seq),span_s=span,dt_min_s=float(np.min(dt)) if len(dt) else None,dt_median_s=med,dt_p90_s=float(np.percentile(dt,90)) if len(dt) else None,dt_max_s=float(np.max(dt)) if len(dt) else None,sampling_hz=1/med if med else None,nyquist_hz=nyquist,duplicates=int(np.sum(dt==0)),non_monotonic=int(np.sum(dt<0)),missing_timestamps=missing[(view,loc)],resp_support=supported(.1,.6),card_support=supported(.8,2.0)))
    result={'location_rule':'QC-valid observations grouped by (upper(view), slice_id), as in ViewLocationBalancedSampler; normalized-position assignment checks common physical center within 1e-4 mm and aligned unit normals within 1e-4','timestamp_source':'DICOM AcquisitionTime in seconds; frame_index ordering','quota_policy':'24 observations allocated by Hamilton largest remainder using QC-valid location counts; uniform location with replacement, then uniform frame', 'location_counts':loc_counts,'quotas':quotas,'support_rule':'at least 8 frames, span >= one period at lower band edge, median-dt Nyquist >= upper band edge, strictly increasing timestamps; necessary screening, not proof of physiological identifiability','stacks':{}}
    for view in ('SAX','2CH','4CH'):
        rr=[r for r in out if r['view']==view]
        all_dt=[float(x) for (v,_), fs in groups.items() if v==view for x in np.diff([t for _,t in sorted(fs)])]
        result['stacks'][view]={'valid_observations':sum(r['observations'] for r in rr),'unique_locations':len(rr),'observations_per_location':stats([r['observations'] for r in rr]),'span_s':stats([r['span_s'] for r in rr]),'timestamp_spacing_s':stats(all_dt),'sampling_hz':stats([r['sampling_hz'] for r in rr if r['sampling_hz'] is not None]),'nyquist_hz':stats([r['nyquist_hz'] for r in rr if r['nyquist_hz'] is not None]),'missing_timestamps':sum(r['missing_timestamps'] for r in rr),'duplicate_intervals':sum(r['duplicates'] for r in rr),'non_monotonic_intervals':sum(r['non_monotonic'] for r in rr),'resp_supported_locations':sum(r['resp_support'] for r in rr),'card_supported_locations':sum(r['card_support'] for r in rr)}
    result['input_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('manifest',a.manifest),('qc_table',a.qc_table)]}
    for view in result['stacks']:
        st=result['stacks'][view]; st['resp_supported_fraction']=st['resp_supported_locations']/st['unique_locations']; st['card_supported_fraction']=st['card_supported_locations']/st['unique_locations']; st['sampling_regularity']='approximately uniform at 1 ms timestamp resolution' if st['timestamp_spacing_s']['max']-st['timestamp_spacing_s']['min']<=0.00101 else 'irregular'
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'temporal_observability.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output_dir/'temporal_observability.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]),delimiter='\t'); w.writeheader(); w.writerows(out)
    (a.output_dir/'temporal_observability.md').write_text('# Temporal observability\n\n'+result['location_rule']+'\n\n'+result['timestamp_source']+'\n\n'+result['support_rule']+'\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
