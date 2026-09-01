"""
功能：加载 CardioResp 4D MRI Phase 1 的便携式数据配置。
论文来源：Necessary adaptation: configuration for local image-domain DICOM inputs.
输入：YAML 文件，包含 dicom_root、views、results_dir 和 manifest_stem。
输出：不可变的 AppConfig 配置对象。
主要步骤：读取 YAML，校验字段，并将相对输出路径解析为配置文件所在目录。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.config --config configs/subject_local.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Paths and view selection required to create a DICOM manifest."""

    dicom_root: Path
    views: tuple[str, ...]
    results_dir: Path
    manifest_stem: str = "dicom_manifest"
    expected_frames_per_slice: int = 50


def load_config(path: str | Path) -> AppConfig:
    """Load and validate one portable YAML configuration file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw: Mapping[str, Any] = yaml.safe_load(handle) or {}
    return config_from_mapping(raw, config_path.parent)


def config_from_mapping(raw: Mapping[str, Any], base_dir: Path) -> AppConfig:
    """Construct an AppConfig from YAML-like values, resolving relative paths."""
    required = ("dicom_root", "views", "results_dir")
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise ValueError(f"Missing required configuration field(s): {', '.join(missing)}")
    views_value = raw["views"]
    if not isinstance(views_value, Sequence) or isinstance(views_value, str):
        raise ValueError("views must be a YAML list of view names")
    dicom_root = _resolve_path(raw["dicom_root"], base_dir)
    results_dir = _resolve_path(raw["results_dir"], base_dir)
    manifest_stem = str(raw.get("manifest_stem", "dicom_manifest"))
    if not manifest_stem or Path(manifest_stem).name != manifest_stem:
        raise ValueError("manifest_stem must be a filename stem without directories")
    try:
        expected_frames_per_slice = int(raw.get("expected_frames_per_slice", 50))
    except (TypeError, ValueError) as error:
        raise ValueError("expected_frames_per_slice must be a positive integer") from error
    config = AppConfig(
        dicom_root,
        tuple(str(view) for view in views_value),
        results_dir,
        manifest_stem,
        expected_frames_per_slice,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Reject invalid frame expectations and output paths that can touch source data."""
    if config.expected_frames_per_slice <= 0:
        raise ValueError("expected_frames_per_slice must be a positive integer")
    root = config.dicom_root.expanduser().resolve()
    results = config.results_dir.expanduser().resolve()
    try:
        results.relative_to(root)
    except ValueError:
        return
    raise ValueError("results_dir must not equal or be nested under dicom_root")


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def main() -> None:
    """Print a validated configuration as JSON for command-line inspection."""
    parser = argparse.ArgumentParser(description="Validate a CardioResp 4D data configuration.")
    parser.add_argument("--config", required=True, type=Path, help="YAML configuration path")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps({
        "dicom_root": str(config.dicom_root),
        "views": list(config.views),
        "results_dir": str(config.results_dir),
        "manifest_stem": config.manifest_stem,
        "expected_frames_per_slice": config.expected_frames_per_slice,
    }, indent=2))


if __name__ == "__main__":
    main()
