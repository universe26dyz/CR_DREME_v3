# Change5C server-only audit, diagnostic, and visualization commands

Run on the GPU server only. These commands evaluate checkpoints read-only; they
do not train or alter checkpoints. Do not interpret any exported dynamic volume
as a globally synchronized physiological 4D cine.

```bash
cd /data/dengyz/code/CR_DREME_v3
git fetch origin
git checkout dev/cardioresp4d
git pull --ff-only origin dev/cardioresp4d
git rev-parse HEAD
conda activate cr_dreme

PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
FREQ=${PHASE1}/frequency/frequency_bands.json
MANIFEST=${PHASE1}/dicom_manifest.csv
QC=${PHASE1}/acquisition_qc/acquisition_qc.csv
DOMAIN=${PHASE1}/canonical_domain/canonical_domain.json
OUT=/data/dengyz/dataset/CR_DREME_v3/v1_change5c_audit
mkdir -p ${OUT}

CTRL_2000=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/control_2000/source_first_last.pt
CTRL_2500=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/control_2500/source_first_last.pt
LATE_1e4=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/late_pca_1e4_2500/source_first_last.pt
LATE_3e4=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/late_pca_3e4_2500/source_first_last.pt
EARLY_5B_2500=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/change5b_2500/source_first_last.pt
CTRL_CFG=/data/dengyz/dataset/CR_DREME_v3/v1_change5a_weight_sweep/configs/source_first_change5a_w003.yaml
LATE_1e4_CFG=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/configs/late_pca_1e4.yaml
LATE_3e4_CFG=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/configs/late_pca_3e4.yaml
EARLY_CFG=configs/source_first_change5b.yaml
```

## Full-location diagnostics

```bash
for LABEL in CTRL_2500 LATE_1e4 LATE_3e4 EARLY_5B_2500; do
  case ${LABEL} in
    CTRL_2500) CKPT=${CTRL_2500}; CFG=${CTRL_CFG};;
    LATE_1e4) CKPT=${LATE_1e4}; CFG=${LATE_1e4_CFG};;
    LATE_3e4) CKPT=${LATE_3e4}; CFG=${LATE_3e4_CFG};;
    EARLY_5B_2500) CKPT=${EARLY_5B_2500}; CFG=${EARLY_CFG};;
  esac
  mkdir -p ${OUT}/${LABEL}
  python scripts/diagnose_change4_checkpoint.py \
    --source-config ${CFG} --frequency-bands ${FREQ} --manifest ${MANIFEST} \
    --qc-table ${QC} --canonical-domain ${DOMAIN} --checkpoint ${CKPT} \
    --device cuda --all-locations --all-locations-dir ${OUT}/${LABEL} \
    --output-json ${OUT}/${LABEL}/legacy_diagnostic.json
done
```

`motion_statistics` in `legacy_diagnostic.json` remains the historical first
valid-observation probe. Use `all_location_diagnostic.json`'s explicit
`aggregate_motion_statistics` for aggregate interpretation.

## Paired comparisons

```bash
for LABEL in LATE_1e4 LATE_3e4 EARLY_5B_2500; do
  python scripts/compare_all_location_diagnostics.py \
    --input CTRL_2500=${OUT}/CTRL_2500/all_location_diagnostic.json \
    --input ${LABEL}=${OUT}/${LABEL}/all_location_diagnostic.json \
    --output-json ${OUT}/compare_CTRL_2500_vs_${LABEL}.json
done
```

These reports are descriptive paired deltas only; do not make significance
claims from this single-subject comparison.

## Read-only loss-gradient audits

```bash
for LABEL in CTRL_2500 LATE_1e4 EARLY_5B_2500; do
  case ${LABEL} in
    CTRL_2500) CKPT=${CTRL_2500}; CFG=${CTRL_CFG};;
    LATE_1e4) CKPT=${LATE_1e4}; CFG=${LATE_1e4_CFG};;
    EARLY_5B_2500) CKPT=${EARLY_5B_2500}; CFG=${EARLY_CFG};;
  esac
  python scripts/audit_stage3a_loss_gradients.py \
    --source-config ${CFG} --frequency-bands ${FREQ} --manifest ${MANIFEST} \
    --qc-table ${QC} --canonical-domain ${DOMAIN} --checkpoint ${CKPT} \
    --view SAX --slice-id SAX_s026 --device cuda \
    --output-json ${OUT}/${LABEL}/stage3a_gradient_audit.json
done
```

## Observation-conditioned visualization packages

```bash
for LABEL in CTRL_2000 CTRL_2500 LATE_1e4 EARLY_5B_2500; do
  case ${LABEL} in
    CTRL_2000) CKPT=${CTRL_2000}; CFG=${CTRL_CFG};;
    CTRL_2500) CKPT=${CTRL_2500}; CFG=${CTRL_CFG};;
    LATE_1e4) CKPT=${LATE_1e4}; CFG=${LATE_1e4_CFG};;
    EARLY_5B_2500) CKPT=${EARLY_5B_2500}; CFG=${EARLY_CFG};;
  esac
  for LOC in SAX/SAX_s026 SAX/SAX_s001 2CH/2CH_s001 4CH/4CH_s001; do
    VIEW=${LOC%%/*}; SLICE=${LOC#*/}
    python scripts/visualize_checkpoint_dynamics.py \
      --source-config ${CFG} --manifest ${MANIFEST} --qc-table ${QC} \
      --canonical-domain ${DOMAIN} --checkpoint ${CKPT} --view ${VIEW} \
      --slice-id ${SLICE} --device cuda --frames 12 --grid-shape 64 64 64 \
      --chunk-size 65536 --seed 0 --output-dir ${OUT}/${LABEL}/visualize_${VIEW}_${SLICE}
  done
done
```

The package exports deterministic reprojection arrays/PNGs and CSV MSEs plus
direct pullback volume, DVF, and observation-to-reference Jacobian arrays in
NPZ. It does not use thick-slice PSF for voxel-grid volume queries.
