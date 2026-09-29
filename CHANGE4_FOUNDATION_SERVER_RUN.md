# C4 foundation audit and server-only runs

Run these commands on CMRServer04 only. They do not start automatically from this checkout. The C4 configuration is a DREME-style negative-crossover-only baseline within CR_DREME_v3, not an exact DREME-MR reproduction.

## Common setup

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
STAGE2C=/data/dengyz/dataset/CR_DREME_v3/v1/train_stage2c_100/source_first_last.pt
C4_CFG=configs/source_first_change4_negonly.yaml
C4_ROOT=/data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly
C4_JOINT=/data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly_joint
mkdir -p "${C4_ROOT}" "${C4_JOINT}"
```

## A. Read-only foundation audit

Run this once for the common Stage2c foundation and once for each final Stage3a checkpoint. The Stage2c JSON explicitly records its documented short state (`global_step=400`, `stage_step=100`) and does not draw a convergence conclusion.

```bash
python scripts/audit_foundation_reconstruction.py \
  --source-config ${C4_CFG} --manifest ${MANIFEST} --qc-table ${QC} \
  --canonical-domain ${DOMAIN} --checkpoint ${STAGE2C} --device cuda \
  --location SAX/SAX_s026 --location 2CH/2CH_s001 --location 4CH/4CH_s001 \
  --frames 8 --seed 0 --slice-chunk-size 1024 \
  --output-dir ${C4_ROOT}/foundation_stage2c
```

## B. C4_NEGONLY_2500: Stage3a 1000+500+500+500

The following single `nohup` chain refuses pre-existing output directories and verifies the serialized parent checkpoint's stage and stage step before every segment.

```bash
nohup bash -lc '
set -euo pipefail
cd /data/dengyz/code/CR_DREME_v3
source /etc/profile
conda activate cr_dreme
PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
FREQ=${PHASE1}/frequency/frequency_bands.json; MANIFEST=${PHASE1}/dicom_manifest.csv
QC=${PHASE1}/acquisition_qc/acquisition_qc.csv; DOMAIN=${PHASE1}/canonical_domain/canonical_domain.json
CFG=configs/source_first_change4_negonly.yaml
ROOT=/data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly
PARENT=/data/dengyz/dataset/CR_DREME_v3/v1/train_stage2c_100/source_first_last.pt
check_parent() { python - "$1" "$2" "$3" <<"PY"
import sys, torch
path, stage, stage_step = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)["training_state"]
assert state["current_stage"] == stage, (state["current_stage"], stage)
assert int(state["stage_step"]) == int(stage_step), (state["stage_step"], stage_step)
print({"parent": path, "stage": stage, "stage_step": stage_step, "global_step": state["global_step"]})
PY
}
run() { OUT=$1; RESUME=$2; STEPS=$3; EXPECTED_STAGE=$4; EXPECTED_STEP=$5
  test -f "${RESUME}"; test ! -e "${OUT}"; check_parent "${RESUME}" "${EXPECTED_STAGE}" "${EXPECTED_STEP}"
  python scripts/train_source_first.py --source-config "${CFG}" --frequency-bands "${FREQ}" \
    --manifest "${MANIFEST}" --qc-table "${QC}" --canonical-domain "${DOMAIN}" \
    --output-dir "${OUT}" --device cuda --pixel-samples 256 --seed 0 --resume "${RESUME}" \
    --stage1-steps 0 --stage2a-steps 0 --stage2b-steps 0 --stage2c-steps 0 \
    --stage3a-steps "${STEPS}" --stage3b-steps 0 --stage3c-steps 0
}
run "${ROOT}/stage3a1000" "${PARENT}" 1000 stage2c 100
run "${ROOT}/stage3a1500" "${ROOT}/stage3a1000/source_first_last.pt" 500 stage3a 1000
run "${ROOT}/stage3a2000" "${ROOT}/stage3a1500/source_first_last.pt" 500 stage3a 500
run "${ROOT}/stage3a2500" "${ROOT}/stage3a2000/source_first_last.pt" 500 stage3a 500
' > /data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly/c4_stage3a2500.log 2>&1 &
```

Each resumed `run_stage` restarts `stage_step` at zero, which is why the three continuation parents are verified at `500`, not cumulative `1500/2000`.

## C. C4 Stage3b joint rescue: 100+400

```bash
nohup bash -lc '
set -euo pipefail
cd /data/dengyz/code/CR_DREME_v3
source /etc/profile
conda activate cr_dreme
PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
FREQ=${PHASE1}/frequency/frequency_bands.json; MANIFEST=${PHASE1}/dicom_manifest.csv
QC=${PHASE1}/acquisition_qc/acquisition_qc.csv; DOMAIN=${PHASE1}/canonical_domain/canonical_domain.json
CFG=configs/source_first_change4_negonly.yaml
ROOT=/data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly_joint
PARENT=/data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly/stage3a2500/source_first_last.pt
check_parent() { python - "$1" "$2" "$3" <<"PY"
import sys, torch
path, stage, stage_step = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)["training_state"]
assert state["current_stage"] == stage, (state["current_stage"], stage)
assert int(state["stage_step"]) == int(stage_step), (state["stage_step"], stage_step)
print({"parent": path, "stage": stage, "stage_step": stage_step, "global_step": state["global_step"]})
PY
}
run() { OUT=$1; RESUME=$2; STEPS=$3; EXPECTED_STAGE=$4; EXPECTED_STEP=$5
  test -f "${RESUME}"; test ! -e "${OUT}"; check_parent "${RESUME}" "${EXPECTED_STAGE}" "${EXPECTED_STEP}"
  python scripts/train_source_first.py --source-config "${CFG}" --frequency-bands "${FREQ}" \
    --manifest "${MANIFEST}" --qc-table "${QC}" --canonical-domain "${DOMAIN}" \
    --output-dir "${OUT}" --device cuda --pixel-samples 256 --seed 0 --resume "${RESUME}" \
    --stage1-steps 0 --stage2a-steps 0 --stage2b-steps 0 --stage2c-steps 0 \
    --stage3a-steps 0 --stage3b-steps "${STEPS}" --stage3c-steps 0
}
run "${ROOT}/stage3b100" "${PARENT}" 100 stage3a 500
run "${ROOT}/stage3b500" "${ROOT}/stage3b100/source_first_last.pt" 400 stage3b 100
' > /data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly_joint/c4_stage3b500.log 2>&1 &
```

## D. Four-arm unified read-only evaluation

This closure expects Stage3a checkpoints; keep the Stage3b rescue as its separate mechanism comparison. It runs semantic, full-location ablation, gradients, foundation reconstruction, and fixed-scale visualization without optimizer steps.

```bash
C4=/data/dengyz/dataset/CR_DREME_v3/v1_change4_negonly/stage3a2500/source_first_last.pt
C5A=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/control_2500/source_first_last.pt
C5B_LATE=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/late_pca_1e4_2500/source_first_last.pt
C5B=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_length_probe/change5b_2500/source_first_last.pt
C5A_CFG=/data/dengyz/dataset/CR_DREME_v3/v1_change5a_weight_sweep/configs/source_first_change5a_w003.yaml
C5B_LATE_CFG=/data/dengyz/dataset/CR_DREME_v3/v1_change5b_late_pca/configs/late_pca_1e4.yaml
python scripts/run_change5c_closure.py \
  --baseline C4_NEGONLY_2500 \
  --experiment C4_NEGONLY_2500=${C4_CFG},${C4} \
  --experiment C5A_W003_2500=${C5A_CFG},${C5A} \
  --experiment C5B_LATE1E4_2500=${C5B_LATE_CFG},${C5B_LATE} \
  --experiment C5B_1E3_2500=configs/source_first_change5b.yaml,${C5B} \
  --frequency-bands ${FREQ} --manifest ${MANIFEST} --qc-table ${QC} \
  --canonical-domain ${DOMAIN} --device cuda --seed 0 \
  --output-dir /data/dengyz/dataset/CR_DREME_v3/v1_change5c_c4_comparison
```
