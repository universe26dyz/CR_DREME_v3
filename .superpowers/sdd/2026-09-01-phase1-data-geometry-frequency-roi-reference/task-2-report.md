# Task 2 report — DICOM patient-world geometry, normalization, and QC

## Status

Implemented Task 2 only. The new `cardioresp4d.geometry` package supplies a strict DICOM PS3.3 `DicomPlane`, an invertible subject-world `WorldNormalizer`, and JSON/PNG manifest-driven geometry QC. No Task 3 or later files were changed.

## Coordinate convention

The public pixel API is `(column, row)` and follows the required mapping:

`world = origin + column * PixelSpacing[1] * IOP[:3] + row * PixelSpacing[0] * IOP[3:]`

The plane normal is `cross(IOP[:3], IOP[3:])`. `world_to_pixel` uses a two-by-two Gram solve of the physical in-plane step vectors. This preserves an exact in-plane inverse despite normal DICOM direction-cosine decimal rounding; it also intentionally orthogonally projects off-plane points for the planned shared-3D-box ROI projection.

## Delivered files

- `src/cardioresp4d/geometry/world_geometry.py`: `DicomPlane`, validation, scalar/batch coordinate transforms, corners and bounds.
- `src/cardioresp4d/geometry/coordinate_normalization.py`: all-plane-corner subject bounding box, serializable 4x4 forward/inverse matrices, scalar and batch transforms.
- `src/cardioresp4d/geometry/geometry_qc.py`: CSV-manifest loading, evenly sampled per-view representative planes, numerical round trips, JSON and 3D PNG rendering.
- `tests/test_geometry.py`: hand-derived anisotropic-spacing, inverse, rounded-IOP regression, orthonormality, bounds/corners, normalizer, and QC integration coverage.

## TDD evidence

1. Initial `python -m unittest tests.test_geometry -v` failed with the expected `ModuleNotFoundError: No module named 'cardioresp4d.geometry'` before the production module existed.
2. The initial implementation made the test suite green.
3. Real QC found `3.44e-6 mm` error due to rounded non-exactly-orthogonal IOP. A regression test was added and observed failing (`6.375e-7 px` coordinate deviation), then passed after replacing independent dot-product inversion with the Gram-system solve.

## Real-data QC

Generated ignored artifacts:

- `results/geometry_qc/geometry_qc.json`
- `results/geometry_qc/geometry_qc.png`

Input: `results/dicom_manifest.csv`.

- Unique physical planes: 144.
- Rendered samples: 9 (three each from SAX, 2CH, and 4CH).
- Pixel → world → pixel maximum error: `5.684341886080802e-14 px`.
- World → pixel → world maximum in-plane error: `1.1368683772161603e-13 mm` (well below `1e-6 mm`).
- The PNG was visually inspected: translucent view-specific plane polygons, centres, and normal arrows agree with the expected SAX/2CH/4CH geometry.

## Verification commands

All commands use `conda run --no-capture-output -n knesvr_torch`; QC also uses `MPLCONFIGDIR=/tmp/cardioresp-mpl`, and source-layout CLI calls use `PYTHONPATH=src`.

- `python -m unittest tests.test_geometry -v`
- `python -m unittest discover -v`
- `python -m compileall -q src tests`
- `python -m cardioresp4d.geometry.world_geometry --help`
- `python -m cardioresp4d.geometry.coordinate_normalization --help`
- `python -m cardioresp4d.geometry.geometry_qc --help`
- `python -m cardioresp4d.geometry.geometry_qc --manifest results/dicom_manifest.csv --output-dir results/geometry_qc --max-per-view 3`
- `git diff --check`

## Concerns

The real 3D plot deliberately labels all nine selected planes, so centre labels are dense in the current viewing angle. The plane polygons, centres and normal arrows remain legible; JSON retains unambiguous per-plane values. Real DICOM IOP values are rounded, hence the Gram-system inverse is necessary to satisfy the sub-micrometre physical round-trip requirement without altering PS3.3 forward mapping.
