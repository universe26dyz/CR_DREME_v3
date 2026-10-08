# C4 paper-aligned server run

Run only on CMRServer04. No command below is started by this checkout.

## A. update repo and verify HEAD

```bash
cd /data/dengyz/code/CR_DREME_v3
git fetch origin
git checkout dev/cardioresp4d
git pull --ff-only origin dev/cardioresp4d
git rev-parse HEAD
conda activate cr_dreme

PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
FREQ=/data/dengyz/dataset/CR_DREME_v3/v1/phase1/frequency/frequency_bands.json
MANIFEST=/data/dengyz/dataset/CR_DREME_v3/v1/phase1/dicom_manifest.csv
QC=/data/dengyz/dataset/CR_DREME_v3/v1/phase1/acquisition_qc/acquisition_qc.csv
DOMAIN=/data/dengyz/dataset/CR_DREME_v3/v1/phase1/canonical_domain/canonical_domain.json
TRAIN_OUT=/data/dengyz/dataset/CR_DREME_v3/v1_change4_paperaligned_full
PREFLIGHT_OUT=/data/dengyz/dataset/CR_DREME_v3/c4_paperaligned_loss_scale_preflight.json
LAUNCH_LOG=/data/dengyz/dataset/CR_DREME_v3/c4_paperaligned_full_launcher.log
CFG=configs/source_first_change4_paperaligned.yaml
```

## B. one-batch read-only loss-scale preflight

```bash
test ! -e ${PREFLIGHT_OUT}
python scripts/preflight_c4_paperaligned_loss_scale.py \
  --source-config ${CFG} --frequency-bands ${FREQ} --manifest ${MANIFEST} \
  --qc-table ${QC} --canonical-domain ${DOMAIN} --segment s3b_full \
  --device cuda --seed 0 --output-json ${PREFLIGHT_OUT}
```

Inspect warnings before launching. The command never performs `optimizer.step()` and never changes weights.

## C. launch full C4 6250-update run with nohup

```bash
test ! -e ${TRAIN_OUT}/c4_paperaligned_run_manifest.json
nohup python scripts/run_c4_paperaligned_full.py \
  --source-config ${CFG} --frequency-bands ${FREQ} --manifest ${MANIFEST} \
  --qc-table ${QC} --canonical-domain ${DOMAIN} --output-dir ${TRAIN_OUT} \
  --device cuda --pixel-samples 256 --seed 0 \
  > ${LAUNCH_LOG} 2>&1 &
```

The launcher runs: `s1a_500`, `s1b_1800`, `s2a_init_1850`, `s2a_joint_2050`, `s2b_init_2100`, `s2b_joint_2300`, `s2c_init_2350`, `s2c_joint_2550`, `s3a_2600`, and `s3b_full_6250`.

## D. concise status command

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/data/dengyz/dataset/CR_DREME_v3/v1_change4_paperaligned_full/c4_paperaligned_run_manifest.json')
data = json.loads(p.read_text())
print({'status': data['status'], 'completed': [(x['segment'], x['status']) for x in data['completed']]})
PY
```

## E. resume command

```bash
python scripts/run_c4_paperaligned_full.py \
  --source-config ${CFG} --frequency-bands ${FREQ} --manifest ${MANIFEST} \
  --qc-table ${QC} --canonical-domain ${DOMAIN} --output-dir ${TRAIN_OUT} \
  --device cuda --pixel-samples 256 --seed 0 --resume
```

The launcher verifies contiguous checkpoint lineage and starts after the last completed segment; it never replays a verified segment.

## F. completion gate

```bash
test -f ${TRAIN_OUT}/s3b_full_6250/source_first_last.pt
python - <<'PY'
import json
from pathlib import Path
p = Path('/data/dengyz/dataset/CR_DREME_v3/v1_change4_paperaligned_full/c4_paperaligned_run_manifest.json')
data = json.loads(p.read_text())
assert data['status'] == 'completed', data['status']
assert len(data['completed']) == 10, len(data['completed'])
print({'status': data['status'], 'final_checkpoint': data['completed'][-1]['checkpoint']})
PY
```

## G. final foundation/dynamics evaluation command

```bash
FINAL=${TRAIN_OUT}/s3b_full_6250/source_first_last.pt
EVAL=${TRAIN_OUT}/final_evaluation
mkdir -p ${EVAL}
python scripts/audit_foundation_reconstruction.py \
  --source-config ${CFG} --manifest ${MANIFEST} --qc-table ${QC} \
  --canonical-domain ${DOMAIN} --checkpoint ${FINAL} --device cuda \
  --location SAX/SAX_s026 --location 2CH/2CH_s001 --location 4CH/4CH_s001 \
  --frames 8 --seed 0 --slice-chunk-size 1024 --output-dir ${EVAL}/foundation
python scripts/diagnose_change4_checkpoint.py \
  --source-config ${CFG} --frequency-bands ${FREQ} --manifest ${MANIFEST} \
  --qc-table ${QC} --canonical-domain ${DOMAIN} --checkpoint ${FINAL} --device cuda \
  --all-locations --all-locations-dir ${EVAL}/semantic --output-json ${EVAL}/semantic/legacy_diagnostic.json
python scripts/audit_all_location_cardiac_ablation.py \
  --source-config ${CFG} --manifest ${MANIFEST} --qc-table ${QC} \
  --canonical-domain ${DOMAIN} --checkpoint ${FINAL} --device cuda --seed 0 \
  --output-json ${EVAL}/ablation.json --output-csv ${EVAL}/ablation.csv
```

The foundation audit uses one shared deterministic PSF realization for every temporal frame within each fixed `view/slice_id`; it reports acquired, canonical-direct, canonical+PSF, Resp-only, Resp+Card, residual/edge/temporal measures, and cardiac-effect amplitude. The semantic and ablation outputs retain score-spectrum/frequency and DVF/Jacobian diagnostics. These are observation-conditioned outputs, not globally synchronized physiological cine.
