import csv
import json
from pathlib import Path
import numpy as np
import pytest
from cardioresp4d.experiments.simple4d_dual_band_periodicity import analyze_trace, frequency_stage


@pytest.mark.parametrize('resp,cardiac',[(1,0),(0,1),(1,1)])
def test_band_recovery(resp,cardiac):
    rng=np.random.default_rng(2); t=np.cumsum(rng.uniform(.15,.19,180)); y=resp*np.sin(2*np.pi*.3*t)+cardiac*np.sin(2*np.pi*1.1*t)
    out=analyze_trace(t,y); a=out['bands']['respiratory']; b=out['bands']['cardiac_candidate']
    if resp: assert a['dominant_frequency_hz']==pytest.approx(.3,abs=.02)
    if cardiac: assert b['dominant_frequency_hz']==pytest.approx(1.1,abs=.02)
    if not cardiac: assert b['normalized_peak_power']<.1
    if not resp: assert a['normalized_peak_power']<.1


def test_insufficient_nyquist():
    t=np.arange(40)*.5; out=analyze_trace(t,np.sin(2*np.pi*.3*t))
    assert out['bands']['respiratory']['available'] and out['bands']['cardiac_candidate']['reason']=='band_exceeds_estimated_nyquist'


def test_csv_only(tmp_path):
    path=tmp_path/'07_cardioresp_identity/step_2000/motion_decomposition'; path.mkdir(parents=True)
    with (path/'frame_motion_decomposition.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['region','slice_id','dynamic_frame_id','timestamp_s','pc1','tz_mm']); writer.writeheader()
        for i in range(80): writer.writerow({'region':'anatomy_support','slice_id':'a','dynamic_frame_id':i,'timestamp_s':i*.17,'pc1':np.sin(2*np.pi*.3*i*.17),'tz_mm':np.sin(2*np.pi*1.1*i*.17)})
    frequency_stage(tmp_path,[2000]); result=json.loads((path/'frequency_band_summary.json').read_text())
    assert result['regions']['anatomy_support']['pc1']['bands']['respiratory']['available']
    assert (path/'frequency_bands_pc1.png').exists() and not (path/'frequency_summary.json').exists()


def test_cross_checkpoint_provenance(tmp_path):
    import hashlib
    for step in (2000,5000,10000):
        folder=tmp_path/f'07_cardioresp_identity/step_{step}/motion_decomposition'; folder.mkdir(parents=True)
        with (folder/'frame_motion_decomposition.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['region','slice_id','dynamic_frame_id','timestamp_s','pc1']); writer.writeheader()
            for i in range(80): writer.writerow({'region':'anatomy_support','slice_id':'a','dynamic_frame_id':i,'timestamp_s':i*.17,'pc1':np.sin(2*np.pi*.3*i*.17)})
    frequency_stage(tmp_path,[2000,5000,10000])
    comparison=json.loads((tmp_path/'07_cardioresp_identity/comparisons/frequency_band_comparison.json').read_text())
    assert len(comparison['rows'])==3 and comparison['composite_score'] is None
    for step,source in comparison['sources'].items():
        path=Path(source['path']); assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
