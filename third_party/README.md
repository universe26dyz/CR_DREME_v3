# Pinned third-party source

This directory vendors unmodified, pinned NeSVoR, SINR and FiLM source trees
used by the v3 source-first adapters. See `SOURCE_LOCK.json` and
`../SOURCE_PROVENANCE.md` for immutable repository, commit, key-file SHA256,
symbol, license and adapter mappings. These are ordinary directories rather
than gitlinks; do not replace them with a floating branch or copy their
algorithms into local modules.

SIMPLE-4D is a pinned upstream snapshot. Do not modify it directly; project
adaptations belong in `src/cardioresp4d/simple4d/`. Its upstream commit and
license are recorded in `simple4d_VENDOR.json`.
