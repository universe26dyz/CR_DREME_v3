from types import SimpleNamespace
import json
import numpy as np
import torch
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import coherence_pairs, coherence_summary, representative_locations
from cardioresp4d.experiments.simple4d_resp_v1_identity_stages import surrogate_stage, coherence_stage


def fixture_rows():
    rows=[]
    for vi,view in enumerate(('SAX','2CH','4CH')):
        for li,location in enumerate(('a','b')):
            for frame in range(6):
                t=frame*.5+vi*.05
                signal=float(np.sin(frame*.7))
                image=torch.tensor(np.ones((1,4,4))*signal,dtype=torch.float32)
                rows.append({'view':view,'slice_id':location,'dynamic_frame_id':100*vi+10*li+frame,'timestamp_s':t,'normalized_slice_position':-.5+li,'pc1':signal,'pc1_location_z':signal,'observation':SimpleNamespace(image=image)})
    return rows


def test_selection_and_pairs():
    assert representative_locations([('a',-.8),('b',0.),('c',.8)]) == ['a','b','c']
    rows=fixture_rows()
    locations=[]
    for view in ('SAX','2CH','4CH'):
        for location in ('a','b'):
            locations.append({'view':view,'slice_id':location,'frames':[r for r in rows if r['view']==view and r['slice_id']==location]})
    pairs,coverage=coherence_pairs(locations,.25)
    assert len(pairs['SAX_2CH'])==24
    assert all(p['source_slice_id'] != p['target_slice_id'] for p in pairs['SAX_SAX'])
    summary=coherence_summary(pairs['SAX_2CH'],coverage['SAX_2CH'],.25)
    assert summary['pearson'] > .999 and summary['pair_count']==24
    no,cover=coherence_pairs(locations,.01)
    missing=coherence_summary(no['SAX_2CH'],cover['SAX_2CH'],.01)
    assert missing['pearson'] is None and missing['coverage_limited']


def test_stage_outputs(tmp_path):
    rows=fixture_rows()
    sur=surrogate_stage(rows,tmp_path/'surrogate')
    coh=coherence_stage(rows,tmp_path/'coherence')
    assert sur['ALL']['valid_location_count']==6 and sur['ALL']['median_abs_pearson']>.999
    assert len(coh)==18
    assert len(list((tmp_path/'surrogate').glob('*.png')))==5
    assert len(list((tmp_path/'coherence').glob('*.png')))==6
    assert json.loads((tmp_path/'surrogate/view_summary.json').read_text())['SAX']['location_count']==2
