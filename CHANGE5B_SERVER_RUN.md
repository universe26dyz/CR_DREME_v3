# Change5B GPU Stage3a run (do not run on the local CPU host)

This is the first PCA-waveform weak-supervision arm. It resumes the common
Change4 Stage2c checkpoint; it does not rerun the completed `w003` control and
does not enter Stage3b or Stage3c.

```bash
cd /data/dengyz/code/CR_DREME_v3
git pull --ff-only origin dev/cardioresp4d
git rev-parse HEAD
conda activate cr_dreme

PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
OUT=/data/dengyz/dataset/CR_DREME_v3/v1_change5b
CKPT=/data/dengyz/dataset/CR_DREME_v3/v1/train_stage2c_100/source_first_last.pt
CONFIG=configs/source_first_change5b.yaml
FREQ=${PHASE1}/frequency/frequency_bands.json
MANIFEST=${PHASE1}/dicom_manifest.csv
QC=${PHASE1}/acquisition_qc/acquisition_qc.csv
DOMAIN=${PHASE1}/domain/canonical_domain.json
```

First perform a no-step resume/config probe (it must only construct and save a
resumed checkpoint):

```bash
python scripts/train_source_first.py --source-config ${CONFIG} --frequency-bands ${FREQ} \
  --manifest ${MANIFEST} --qc-table ${QC} --canonical-domain ${DOMAIN} \
  --output-dir ${OUT}/probe --device cuda --pixel-samples 256 --seed 0 --resume ${CKPT} \
  --stage1-steps 0 --stage2a-steps 0 --stage2b-steps 0 --stage2c-steps 0 \
  --stage3a-steps 0 --stage3b-steps 0 --stage3c-steps 0
```

Then run exactly the Change5B arm: `cardiac_target_concentration=0.003`,
`cardiac_pca_waveform=0.001`, Stage3a=1000, and Stage3b/3c=0.

```bash
python scripts/train_source_first.py --source-config ${CONFIG} --frequency-bands ${FREQ} \
  --manifest ${MANIFEST} --qc-table ${QC} --canonical-domain ${DOMAIN} \
  --output-dir ${OUT}/stage3a1000 --device cuda --pixel-samples 256 --seed 0 --resume ${CKPT} \
  --stage1-steps 0 --stage2a-steps 0 --stage2b-steps 0 --stage2c-steps 0 \
  --stage3a-steps 1000 --stage3b-steps 0 --stage3c-steps 0

python scripts/diagnose_change4_checkpoint.py --source-config ${CONFIG} --frequency-bands ${FREQ} \
  --manifest ${MANIFEST} --qc-table ${QC} --canonical-domain ${DOMAIN} \
  --checkpoint ${OUT}/stage3a1000/source_first_last.pt --device cuda \
  --output-json ${OUT}/stage3a1000/change5b_diagnostic.json
```

Print only comparison metrics (PCA R², target fraction, target/wrong relation,
peak degeneracy, ablation, DVF and score spread):

```bash
python - <<'PY'
import json
p = json.load(open('/data/dengyz/dataset/CR_DREME_v3/v1_change5b/stage3a1000/change5b_diagnostic.json'))
r = p['frequency_semantics']['records']; mean = p['frequency_semantics']['summary_mean']; median = p['frequency_semantics']['summary_median']
print('pca_r2 mean/median', mean['cardiac_pca_waveform_r2'], median['cardiac_pca_waveform_r2'])
print('target_fraction mean/median', mean['cardiac_target_fraction'], median['cardiac_target_fraction'])
print('card_target/wrong', sum(x['card_target_amp'] for x in r) / sum(x['card_wrong_amp'] for x in r))
print('target>wrong', sum(x['card_target_amp'] > x['card_wrong_amp'] for x in r), '/', len(r))
print('same_peak_exact', sum(x['card_peak_hz'] == x['resp_peak_hz'] for x in r), '/', len(r))
print('same_peak_within_0.02Hz', sum(abs(x['card_peak_hz']-x['resp_peak_hz']) <= .02 for x in r), '/', len(r))
print('cardiac_ablation', p['cardiac_ablation'])
print('card_dvf', p['motion_statistics'])
print('card_score_std mean/median', mean['card_score_std'], median['card_score_std'])
PY
```

Interpret jointly: higher PCA R² alone is insufficient unless local target-band
semantics improve, peak degeneracy drops, cardiac ablation remains useful, and
cardiac DVF stays non-pathological.
