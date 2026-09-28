# Change5C closure — GPU server command

Run this read-only package on the GPU server after replacing only the clearly
marked checkpoint/config variables. It validates every input before launch,
does not train or modify checkpoints, and writes verbose progress to `logs/`.

```bash
cd /data/dengyz/code/CR_DREME_v3
git fetch origin && git checkout dev/cardioresp4d && git pull --ff-only origin dev/cardioresp4d
conda activate svr4d

PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
FREQ=${PHASE1}/frequency/frequency_bands.json
MANIFEST=${PHASE1}/dicom_manifest.csv
QC=${PHASE1}/acquisition_qc/acquisition_qc.csv
DOMAIN=${PHASE1}/canonical_domain/canonical_domain.json
OUT=/data/dengyz/dataset/CR_DREME_v3/change5c_closure

CTRL_CKPT=/REPLACE/with/CTRL_2500/source_first_last.pt
CTRL_CFG=/REPLACE/with/CTRL_2500/source_first.yaml
LATE_CKPT=/REPLACE/with/LATE_1e4/source_first_last.pt
LATE_CFG=/REPLACE/with/LATE_1e4/source_first.yaml
C5B_CKPT=/REPLACE/with/C5B_2500/source_first_last.pt
C5B_CFG=/REPLACE/with/C5B_2500/source_first.yaml

python scripts/run_change5c_closure.py \
  --experiment CTRL_2500=${CTRL_CFG},${CTRL_CKPT} \
  --experiment LATE_1e4=${LATE_CFG},${LATE_CKPT} \
  --experiment C5B_2500=${C5B_CFG},${C5B_CKPT} \
  --frequency-bands ${FREQ} --manifest ${MANIFEST} --qc-table ${QC} \
  --canonical-domain ${DOMAIN} --output-dir ${OUT} --device cuda --seed 0
```

For a genuinely interrupted run whose existing output provenance has been
checked manually, append `--reuse-valid`; otherwise the runner refuses to
silently overwrite output paths. Confirm `checkpoint_contract_audit.json`:
CTRL has PCA weight `0`, LATE has `1e-4`, and C5B has concentration/PCA weights
`0.003`/`0.001`; directory names are not treated as evidence.
