import json
import pytest
from cardioresp4d.experiments.simple4d_resp_v1_identity_report import comparison_report, METRICS


def test_partial_report_and_directions(tmp_path):
    rows=comparison_report(tmp_path,[2000,5000])
    assert all(r['surrogate'] is None for r in rows)
    assert (tmp_path/'audit_report.md').exists()
    values=json.loads((tmp_path/'comparison/checkpoint_comparison.json').read_text())
    assert values['composite_score'] is None
    assert values['metric_direction']['cycle']=='lower' and values['metric_direction']['jacobian']=='higher'
    assert 'NOT AUTOMATICALLY ASSIGNED' in (tmp_path/'audit_report.md').read_text()
    assert len(list((tmp_path/'comparison').glob('*.png')))==5
    with pytest.raises(FileExistsError): comparison_report(tmp_path,[2000,5000])
    comparison_report(tmp_path,[2000,5000],overwrite=True)


def test_mismatched_stage_provenance(tmp_path):
    for directory,digest in (('00_summary','a'),('01_latent','b')):
        path=tmp_path/directory; path.mkdir()
        (path/'stage_manifest.json').write_text(json.dumps({'input_artifact_sha256':{'manifest':digest},'checkpoints':{},'git_sha':'x'}))
    with pytest.raises(ValueError): comparison_report(tmp_path,[2000])
