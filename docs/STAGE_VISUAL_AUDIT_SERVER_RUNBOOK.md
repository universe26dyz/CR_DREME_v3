# C4 stage visual audit: CMRServer04 runbook

This is read-only diagnostic work. It never trains, rewrites checkpoints, or
changes optimizer state. All output is placed below a new audit directory.

```bash
cd /data/dengyz/code/CR_DREME_v3
git fetch origin
git checkout dev/cardioresp4d
git pull --ff-only origin dev/cardioresp4d
git rev-parse HEAD
conda activate cr_dreme

PHASE1=/data/dengyz/dataset/CR_DREME_v3/v1/phase1
CFG=configs/source_first_change4_paperaligned.yaml
MANIFEST=${PHASE1}/dicom_manifest.csv
QC=${PHASE1}/acquisition_qc/acquisition_qc.csv
DOMAIN=${PHASE1}/canonical_domain/canonical_domain.json
RUN=/data/dengyz/dataset/CR_DREME_v3/v1_change4_paperaligned_full
FINAL=${RUN}/s3b_full_6250/source_first_last.pt
AUDIT=${RUN}/stage_visual_audit_$(git rev-parse --short HEAD)
test -f "${FINAL}"
python scripts/export_stage_visual_audit.py --help >/dev/null
```

## 1. Lightweight final-checkpoint preflight

```bash
python scripts/preflight_c4_paperaligned_loss_scale.py \
  --source-config "${CFG}" --frequency-bands "${PHASE1}/frequency/frequency_bands.json" \
  --manifest "${MANIFEST}" --qc-table "${QC}" --canonical-domain "${DOMAIN}" \
  --device cuda --seed 0 --output-json "${AUDIT}/preflight.json"
```

## 2. SAX_s026 smoke audit

The location spelling is `VIEW/SLICE_ID`; this smoke run uses all valid frames
at only the requested fixed location and a bounded 1024-pixel slice chunk.

```bash
python scripts/export_stage_visual_audit.py \
  --source-config "${CFG}" --manifest "${MANIFEST}" --qc-table "${QC}" \
  --canonical-domain "${DOMAIN}" --checkpoint "s3b_full=${FINAL}" \
  --location SAX/SAX_s026 --output-dir "${AUDIT}/smoke_sax_s026" \
  --device cuda --seed 0 --slice-chunk-size 1024 --volume-chunk-size 65536
```

Inspect `temporal_std/temporal_std_stats.json` and the two temporal-std PNGs
before launching the all-location job. The PNG maps are rendered directly from
the float arrays saved in `temporal_std_arrays.npz`.

## 3. Full final-checkpoint decomposition and demeaned-score audit

```bash
nohup python scripts/export_stage_visual_audit.py \
  --source-config "${CFG}" --manifest "${MANIFEST}" --qc-table "${QC}" \
  --canonical-domain "${DOMAIN}" --checkpoint "s3b_full=${FINAL}" \
  --output-dir "${AUDIT}/final_s3b_full" --device cuda --seed 0 \
  --slice-chunk-size 1024 --volume-chunk-size 65536 \
  > "${AUDIT}/final_s3b_full.log" 2>&1 &
```

## 4. Retrospective available-stage audit

Missing historical checkpoints are explicitly recorded as `unavailable` in
`provenance.json`; this command does not retrain or fabricate them.

```bash
nohup python scripts/export_stage_visual_audit.py \
  --source-config "${CFG}" --manifest "${MANIFEST}" --qc-table "${QC}" \
  --canonical-domain "${DOMAIN}" --run-dir "${RUN}" \
  --output-dir "${AUDIT}/retrospective" --device cuda --seed 0 \
  --slice-chunk-size 1024 --volume-chunk-size 65536 \
  > "${AUDIT}/retrospective.log" 2>&1 &
```

## 5. Concise inspection and recovery

```bash
python - <<PY
import json
from pathlib import Path
p = Path("${AUDIT}/retrospective/provenance.json")
print(json.loads(p.read_text())["checkpoints"])
PY

find "${AUDIT}" -name temporal_std_stats.json -print | head
```

On failure, inspect the corresponding `*.log` and retain all partial audit
outputs. Re-run the same command with a fresh `AUDIT` directory, or re-run a
single checkpoint/location with `--checkpoint LABEL=PATH --location VIEW/SLICE_ID`.
Do not delete or overwrite checkpoints, training logs, or prior final-evaluation
outputs.

## Optional future-training hook

The formal launcher keeps this disabled by default. Supplying the option runs
the exporter after each completed major checkpoint in a separate process; a
visualization failure is recorded in `c4_paperaligned_run_manifest.json` and
does not alter or replay training.

```bash
python scripts/run_c4_paperaligned_full.py \
  --source-config "${CFG}" --frequency-bands "${PHASE1}/frequency/frequency_bands.json" \
  --manifest "${MANIFEST}" --qc-table "${QC}" --canonical-domain "${DOMAIN}" \
  --output-dir /path/to/a-new-training-run --device cuda --pixel-samples 256 --seed 0 \
  --visual-audit-output-dir /path/to/a-new-training-run/stage_visual_audit
```

The current training entry point produces segment-end checkpoints only. It
therefore audits S3B at its final checkpoint; it does not fabricate +500,
+1500, or +3000 snapshots. Add explicit training checkpoint milestones before
requesting those intermediate audits.
