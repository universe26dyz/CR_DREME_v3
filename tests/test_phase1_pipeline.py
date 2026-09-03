"""功能：验证 Phase-1 runner 与模块化配置契约。
论文来源：工程编排 necessary adaptation。
输入：临时路径、完整 nested mapping 和 mock 公开 API。
输出：阶段顺序、范围、依赖错误及固定 v1 参数断言。
主要步骤：先构造配置，再隔离验证控制流；科学计算仍属于各公开模块。
是否属于原论文直接实现 / 必要适配 / 可选实验：necessary adaptation。
命令行：python -m unittest tests.test_phase1_pipeline -v
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.config import config_from_mapping, validate_config  # noqa: E402
from cardioresp4d.config import AppConfig  # noqa: E402


def nested_mapping(root: Path) -> dict:
    """Return a complete portable Phase-1 configuration for control-flow tests."""
    return {
        "project": {"results_dir": str(root / "results")},
        "data": {
            "dicom_root": str(root / "dicom"),
            "views": ["SAX", "2CH", "4CH"],
            "expected_frames_per_slice": 50,
            "manifest_stem": "dicom_manifest",
            "inspection_filename": "inspection.json",
            "expected_series_per_view": {"SAX": 50, "2CH": 52, "4CH": 42},
        },
        "geometry": {"required_views": ["SAX", "2CH", "4CH"], "max_qc_planes_per_view": 3},
        "frequency": {
            "respiratory_band_hz": [0.1, 0.6],
            "cardiac_band_hz": [0.8, 2.0],
            "dominance_threshold": 2.2,
            "uniform_relative_tolerance": 0.001,
            "uniform_absolute_tolerance_s": 0.0011,
            "pca_components": 50,
            "consensus_min_slice_fraction": 0.5,
        },
        "roi": {
            "cardiac_box_center_mm": [1.0, 2.0, 3.0],
            "cardiac_box_size_mm": [4.0, 5.0, 6.0],
            "required_views": ["SAX", "2CH", "4CH"],
        },
        "reference": {
            "source_view": "SAX",
            "orientation_tolerance": 1e-5,
            "spacing_tolerance_mm": 1e-6,
            "duplicate_tolerance_mm": 1e-3,
            "origin_affine_tolerance_mm": 0.5,
        },
        "outlier_qc": {"ncc_mad_threshold": 6.0, "scale_mad_threshold": 6.0,
                       "residual_mad_threshold": 6.0, "mad_floor": 1e-6,
                       "output_subdir": "acquisition_qc"},
    }


def load_runner_module():
    """Load the source-tree script without requiring package installation."""
    path = PROJECT_ROOT / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("phase1_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModularConfigTest(unittest.TestCase):
    def test_nested_configuration_is_the_contract_and_v1_constants_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_from_mapping(nested_mapping(root), root)
            self.assertEqual(root / "dicom", config.data.dicom_root)
            self.assertEqual(root / "results", config.project.results_dir)
            self.assertEqual(50, config.data.expected_frames_per_slice)
            self.assertEqual((0.1, 0.6), config.frequency.respiratory_band_hz)
            for section, field, value in (
                ("data", "expected_frames_per_slice", 49),
                ("geometry", "required_views", ["SAX"]),
                ("frequency", "respiratory_band_hz", [0.2, 0.5]),
                ("frequency", "uniform_relative_tolerance", -1.0),
                ("reference", "source_view", "4CH"),
                ("roi", "cardiac_box_size_mm", [4.0, 0.0, 6.0]),
            ):
                broken = nested_mapping(root)
                broken[section][field] = value
                with self.subTest(section=section, field=field):
                    with self.assertRaises(ValueError):
                        config_from_mapping(broken, root)

    def test_flat_configuration_is_rejected_instead_of_silently_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "module-scoped"):
                config_from_mapping(
                    {"dicom_root": str(root / "dicom"), "views": ["SAX"], "results_dir": str(root / "results")},
                    root,
                )

    def test_direct_python_configuration_cannot_bypass_fixed_acquisition_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(root / "dicom", ("SAX", "2CH", "4CH"), root / "results")
            with self.assertRaisesRegex(ValueError, "expected series SAX:50"):
                validate_config(config)


class Phase1RunnerTest(unittest.TestCase):
    def test_full_run_calls_public_apis_in_dependency_order(self) -> None:
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_from_mapping(nested_mapping(root), root)
            config.project.results_dir.mkdir()
            manifest = config.manifest_csv_path
            calls = []

            def record(name, result):
                def wrapped(*args, **kwargs):
                    calls.append(name)
                    if name == "manifest":
                        manifest.write_text("header\n", encoding="utf-8")
                    return result
                return wrapped

            with patch.object(runner, "write_inspection", record("inspect", config.inspection_path)), \
                 patch.object(runner, "build_manifest", record("manifest", (manifest, config.manifest_json_path))), \
                 patch.object(runner, "validate_manifest_artifacts", lambda *args: (manifest, config.manifest_json_path)), \
                 patch.object(runner, "run_acquisition_qc", record("qc", (Path("q.csv"), Path("q.json"), Path("q.png")))), \
                 patch.object(runner, "validate_qc_table_coverage", lambda *args: None), \
                 patch.object(runner, "run_geometry_qc", record("geometry", (Path("g.json"), Path("g.png")))), \
                 patch.object(runner, "_validate_geometry_tolerances", lambda *args: None), \
                 patch.object(runner, "analyze_manifest", record("frequency", {"slice_count": 3})), \
                 patch.object(runner, "run_roi_qc", record("roi", (Path("r.json"), {}))), \
                 patch.object(runner, "build_initial_reference", record("reference", (Path("v.nii.gz"), Path("v.json"), Path("v.png")))), \
                 patch.object(runner, "_write_run_summary", lambda *args: Path("summary.json")), \
                 patch.object(Path, "is_file", lambda self: True):
                summary = runner.run_pipeline(config)

            self.assertEqual(list(runner.STAGES), calls)
            self.assertEqual(list(runner.STAGES), list(summary))

    def test_stage_range_and_missing_manifest_dependency_are_explicit(self) -> None:
        runner = load_runner_module()
        self.assertEqual(("manifest", "qc", "geometry", "frequency"), runner.selected_stages("manifest", "frequency"))
        self.assertEqual(("reference",), runner.selected_stages("reference", "reference"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_from_mapping(nested_mapping(root), root)
            config.project.results_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "manifest"):
                runner.run_pipeline(config, from_stage="geometry", to_stage="geometry")
            with self.assertRaisesRegex(ValueError, "stage order"):
                runner.run_pipeline(config, from_stage="roi", to_stage="manifest")
            unsafe = config_from_mapping(nested_mapping(root), root)
            object.__setattr__(unsafe.project, "results_dir", unsafe.dicom_root / "results")
            with self.assertRaisesRegex(ValueError, "results_dir"):
                runner.run_pipeline(unsafe, to_stage="inspect")

    def test_inspect_only_range_writes_summary_without_requiring_manifest(self) -> None:
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_from_mapping(nested_mapping(root), root)
            config.dicom_root.mkdir()
            config.results_dir.mkdir()
            with patch.object(runner, "write_inspection", return_value=config.inspection_path):
                summary = runner.run_pipeline(config, to_stage="inspect")
            self.assertEqual(["inspect"], list(summary))
            payload = (config.results_dir / "pipeline_run_summary.json").read_text()
            self.assertIn('"csv_sha256": null', payload)


if __name__ == "__main__":
    unittest.main()
