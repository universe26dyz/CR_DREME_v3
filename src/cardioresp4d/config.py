"""功能：加载按 Phase-1 模块分组的 CardioResp 4D MRI 配置。
论文来源：图像域本地 DICOM pipeline 的 necessary adaptation。
输入：含 project/data/geometry/frequency/roi/reference 的 YAML。
输出：不可变 AppConfig 及各模块配置对象。
主要步骤：解析相对路径、校验 v1 科学常量与数值范围、保护只读 DICOM 树。
是否属于原论文直接实现 / 必要适配 / 可选实验：necessary adaptation。
命令行：python -m cardioresp4d.config --config configs/subject_local.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

EXPECTED_FRAMES_PER_SLICE = 50
REQUIRED_VIEWS = ("SAX", "2CH", "4CH")
RESPIRATORY_BAND_HZ = (0.10, 0.60)
CARDIAC_BAND_HZ = (0.80, 2.00)
DOMINANCE_THRESHOLD = 2.2
UNIFORM_ATOL_S = 1.1e-3


@dataclass(frozen=True)
class ProjectConfig:
    results_dir: Path
    name: str = "cardioresp4d"


@dataclass(frozen=True)
class DataConfig:
    dicom_root: Path
    views: tuple[str, ...] = REQUIRED_VIEWS
    expected_frames_per_slice: int = EXPECTED_FRAMES_PER_SLICE
    manifest_stem: str = "dicom_manifest"
    inspection_filename: str = "inspection.json"


@dataclass(frozen=True)
class GeometryConfig:
    required_views: tuple[str, ...] = REQUIRED_VIEWS
    max_qc_planes_per_view: int = 3
    pixel_round_trip_tolerance: float = 1e-6
    world_round_trip_tolerance_mm: float = 1e-6
    output_subdir: str = "geometry_qc"


@dataclass(frozen=True)
class FrequencyConfig:
    respiratory_band_hz: tuple[float, float] = RESPIRATORY_BAND_HZ
    cardiac_band_hz: tuple[float, float] = CARDIAC_BAND_HZ
    dominance_threshold: float = DOMINANCE_THRESHOLD
    uniform_relative_tolerance: float = 1e-3
    uniform_absolute_tolerance_s: float = UNIFORM_ATOL_S
    pca_components: int = EXPECTED_FRAMES_PER_SLICE
    output_subdir: str = "frequency"


@dataclass(frozen=True)
class RoiConfig:
    cardiac_box_center_mm: tuple[float, float, float] | None = None
    cardiac_box_size_mm: tuple[float, float, float] | None = None
    required_views: tuple[str, ...] = REQUIRED_VIEWS
    output_subdir: str = "roi_qc"


@dataclass(frozen=True)
class ReferenceConfig:
    source_view: str = "SAX"
    orientation_tolerance: float = 1e-5
    spacing_tolerance_mm: float = 1e-6
    duplicate_tolerance_mm: float = 1e-3
    origin_affine_tolerance_mm: float = 0.5
    output_subdir: str = "reference"


@dataclass(frozen=True, init=False)
class AppConfig:
    """Validated settings with old Python constructor aliases for existing tests.

    YAML loading accepts only the nested contract and cannot mix flat/nested keys.
    """
    project: ProjectConfig
    data: DataConfig
    geometry: GeometryConfig
    frequency: FrequencyConfig
    roi: RoiConfig
    reference: ReferenceConfig

    def __init__(self, dicom_root: str | Path | None = None,
                 views: Sequence[str] = REQUIRED_VIEWS,
                 results_dir: str | Path | None = None,
                 manifest_stem: str = "dicom_manifest",
                 expected_frames_per_slice: int = EXPECTED_FRAMES_PER_SLICE, *,
                 project: ProjectConfig | None = None, data: DataConfig | None = None,
                 geometry: GeometryConfig | None = None, frequency: FrequencyConfig | None = None,
                 roi: RoiConfig | None = None, reference: ReferenceConfig | None = None) -> None:
        if project is None or data is None:
            if dicom_root is None or results_dir is None:
                raise ValueError("AppConfig requires project/data sections or dicom_root/results_dir")
            project = ProjectConfig(Path(results_dir))
            data = DataConfig(Path(dicom_root), tuple(str(v).upper() for v in views),
                              expected_frames_per_slice, manifest_stem)
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "geometry", geometry or GeometryConfig())
        object.__setattr__(self, "frequency", frequency or FrequencyConfig())
        object.__setattr__(self, "roi", roi or RoiConfig())
        object.__setattr__(self, "reference", reference or ReferenceConfig())

    @property
    def dicom_root(self) -> Path: return self.data.dicom_root
    @property
    def views(self) -> tuple[str, ...]: return self.data.views
    @property
    def results_dir(self) -> Path: return self.project.results_dir
    @property
    def manifest_stem(self) -> str: return self.data.manifest_stem
    @property
    def expected_frames_per_slice(self) -> int: return self.data.expected_frames_per_slice
    @property
    def manifest_csv_path(self) -> Path: return self.results_dir / f"{self.manifest_stem}.csv"
    @property
    def manifest_json_path(self) -> Path: return self.results_dir / f"{self.manifest_stem}.json"
    @property
    def inspection_path(self) -> Path: return self.results_dir / self.data.inspection_filename


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_mapping(raw, config_path.parent)


def config_from_mapping(raw: Mapping[str, Any], base_dir: Path) -> AppConfig:
    """Build nested configuration; deliberately reject old flat YAML."""
    sections = ("project", "data", "geometry", "frequency", "roi", "reference")
    if any(k in raw for k in ("dicom_root", "views", "results_dir", "manifest_stem")):
        raise ValueError("Configuration must use the module-scoped project/data/geometry/frequency/roi/reference contract")
    missing = [k for k in sections if not isinstance(raw.get(k), Mapping)]
    if missing:
        raise ValueError(f"Missing required module-scoped configuration section(s): {', '.join(missing)}")
    p, d, g, f, r, ref = (raw[k] for k in sections)
    if not p.get("results_dir") or not d.get("dicom_root"):
        raise ValueError("project.results_dir and data.dicom_root are required")
    config = AppConfig(
        project=ProjectConfig(_resolve_path(p["results_dir"], base_dir), str(p.get("name", "cardioresp4d"))),
        data=DataConfig(_resolve_path(d["dicom_root"], base_dir), _strings(d.get("views", REQUIRED_VIEWS), "data.views"),
                        int(d.get("expected_frames_per_slice", 50)), str(d.get("manifest_stem", "dicom_manifest")),
                        str(d.get("inspection_filename", "inspection.json"))),
        geometry=GeometryConfig(_strings(g.get("required_views", REQUIRED_VIEWS), "geometry.required_views"),
                                int(g.get("max_qc_planes_per_view", 3)), float(g.get("pixel_round_trip_tolerance", 1e-6)),
                                float(g.get("world_round_trip_tolerance_mm", 1e-6)), str(g.get("output_subdir", "geometry_qc"))),
        frequency=FrequencyConfig(_pair(f.get("respiratory_band_hz", RESPIRATORY_BAND_HZ), "frequency.respiratory_band_hz"),
                                  _pair(f.get("cardiac_band_hz", CARDIAC_BAND_HZ), "frequency.cardiac_band_hz"),
                                  float(f.get("dominance_threshold", 2.2)), float(f.get("uniform_relative_tolerance", 1e-3)),
                                  float(f.get("uniform_absolute_tolerance_s", UNIFORM_ATOL_S)), int(f.get("pca_components", 50)),
                                  str(f.get("output_subdir", "frequency"))),
        roi=RoiConfig(_triple(r.get("cardiac_box_center_mm"), "roi.cardiac_box_center_mm"),
                      _triple(r.get("cardiac_box_size_mm"), "roi.cardiac_box_size_mm"),
                      _strings(r.get("required_views", REQUIRED_VIEWS), "roi.required_views"), str(r.get("output_subdir", "roi_qc"))),
        reference=ReferenceConfig(str(ref.get("source_view", "SAX")).upper(), float(ref.get("orientation_tolerance", 1e-5)),
                                  float(ref.get("spacing_tolerance_mm", 1e-6)), float(ref.get("duplicate_tolerance_mm", 1e-3)),
                                  float(ref.get("origin_affine_tolerance_mm", 0.5)), str(ref.get("output_subdir", "reference"))),
    )
    validate_config(config)
    if config.views != REQUIRED_VIEWS or config.geometry.required_views != REQUIRED_VIEWS or config.roi.required_views != REQUIRED_VIEWS:
        raise ValueError("Phase 1 v1 nested configuration requires views exactly SAX, 2CH, 4CH in that order")
    return config


def validate_config(config: AppConfig) -> None:
    validate_expected_frames_per_slice(config.expected_frames_per_slice)
    validate_output_path(config.dicom_root, config.results_dir, "project.results_dir")
    if config.frequency.respiratory_band_hz != RESPIRATORY_BAND_HZ or config.frequency.cardiac_band_hz != CARDIAC_BAND_HZ:
        raise ValueError("Phase 1 v1 frequency bands are fixed")
    if config.frequency.dominance_threshold != DOMINANCE_THRESHOLD or config.frequency.uniform_absolute_tolerance_s != UNIFORM_ATOL_S:
        raise ValueError("Phase 1 v1 dominance and DICOM quantisation tolerances cannot be overridden")
    if config.frequency.pca_components != 50 or config.reference.source_view != "SAX":
        raise ValueError("Phase 1 v1 requires all 50 PCA components and SAX reference")
    if config.geometry.max_qc_planes_per_view <= 0:
        raise ValueError("geometry.max_qc_planes_per_view must be positive")
    positive_values = (
        (config.geometry.pixel_round_trip_tolerance, "geometry.pixel_round_trip_tolerance"),
        (config.geometry.world_round_trip_tolerance_mm, "geometry.world_round_trip_tolerance_mm"),
        (config.frequency.uniform_relative_tolerance, "frequency.uniform_relative_tolerance"),
        (config.reference.orientation_tolerance, "reference.orientation_tolerance"),
        (config.reference.spacing_tolerance_mm, "reference.spacing_tolerance_mm"),
        (config.reference.duplicate_tolerance_mm, "reference.duplicate_tolerance_mm"),
        (config.reference.origin_affine_tolerance_mm, "reference.origin_affine_tolerance_mm"),
    )
    for value, name in positive_values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not config.manifest_stem or Path(config.manifest_stem).name != config.manifest_stem:
        raise ValueError("data.manifest_stem must be a filename stem")
    components = (config.data.inspection_filename, config.geometry.output_subdir, config.frequency.output_subdir,
                  config.roi.output_subdir, config.reference.output_subdir)
    if any(not x or Path(x).name != x for x in components):
        raise ValueError("output filenames/subdirectories must be single path components")
    if (config.roi.cardiac_box_center_mm is None) != (config.roi.cardiac_box_size_mm is None):
        raise ValueError("roi cardiac box center and size must be supplied together")
    if config.roi.cardiac_box_center_mm is not None:
        if not all(math.isfinite(value) for value in config.roi.cardiac_box_center_mm):
            raise ValueError("roi cardiac box center must be finite")
        if not all(math.isfinite(value) and value > 0.0 for value in config.roi.cardiac_box_size_mm or ()):
            raise ValueError("roi cardiac box size must be finite and positive")


def validate_expected_frames_per_slice(value: int) -> None:
    if value != EXPECTED_FRAMES_PER_SLICE:
        raise ValueError(f"expected_frames_per_slice must equal exactly {EXPECTED_FRAMES_PER_SLICE} for Phase 1 v1")


def validate_output_path(dicom_root: str | Path, output_path: str | Path, name: str) -> None:
    root, output = Path(dicom_root).expanduser().resolve(), Path(output_path).expanduser().resolve()
    try: output.relative_to(root)
    except ValueError: return
    raise ValueError(f"{name} must not equal or be nested under dicom_root")


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str): raise ValueError(f"{name} must be a YAML list")
    return tuple(str(x).upper() for x in value)


def _pair(value: Any, name: str) -> tuple[float, float]:
    result = tuple(float(x) for x in value) if isinstance(value, Sequence) and not isinstance(value, str) else ()
    if len(result) != 2: raise ValueError(f"{name} must contain two numbers")
    return result  # type: ignore[return-value]


def _triple(value: Any, name: str) -> tuple[float, float, float] | None:
    if value is None: return None
    result = tuple(float(x) for x in value) if isinstance(value, Sequence) and not isinstance(value, str) else ()
    if len(result) != 3: raise ValueError(f"{name} must contain three numbers")
    return result  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a module-scoped CardioResp 4D Phase-1 configuration.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    c = load_config(args.config)
    print(json.dumps({"dicom_root": str(c.dicom_root), "results_dir": str(c.results_dir), "views": list(c.views)}, indent=2))


if __name__ == "__main__": main()
