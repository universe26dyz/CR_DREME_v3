"""Portable JSON provenance and self-contained HTML audit reports."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

from .contract import sha256


def write_audit_manifest(output: str | Path, *, adapter: str, sources: Mapping[str, Path], stages: Mapping[str, Mapping[str, Any]], display_policies: Mapping[str, Any], render_configuration: Mapping[str, Any] | None = None) -> Path:
    payload = {"schema_version": 1, "experiment_adapter": adapter, "source_artifacts": {name: {"source_token": name, "artifact_name": path.name, "sha256": sha256(path)} for name, path in sources.items() if path.is_file()}, "render_configuration": dict(render_configuration or {}), "display_window_policies": dict(display_policies), "stage_status": {name: dict(value) for name, value in stages.items()}}
    path = Path(output); path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return path


def write_html_report(output: str | Path, manifest: Mapping[str, Any], stages: Mapping[str, Mapping[str, Any]]) -> Path:
    path = Path(output); root = path.parent
    def assets(name: str) -> str:
        directory = root / name
        if not directory.is_dir(): return ""
        items = []
        for asset in sorted(directory.rglob("*")):
            if not asset.is_file() or asset.suffix not in {".png", ".csv", ".json", ".gz"}: continue
            relative = asset.relative_to(root).as_posix(); link = f"<a href='{html.escape(relative)}'>{html.escape(relative)}</a>"
            items.append(f"<figure><img src='{html.escape(relative)}' loading='lazy' style='max-width:260px'><figcaption>{link}</figcaption></figure>" if asset.suffix == ".png" else f"<li>{link}</li>")
        return "<div class='assets'>" + "".join(items) + "</div>" if items else ""
    sections = "".join(f"<section><h2>{html.escape(name)}</h2><pre>{html.escape(json.dumps(value, indent=2, sort_keys=True))}</pre>{assets(name)}</section>" for name, value in stages.items())
    scientific = manifest.get("scientific_status", {})
    content = f"<!doctype html><html><head><meta charset='utf-8'><title>Experiment audit</title><style>body{{font-family:sans-serif;margin:2rem;max-width:1100px}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem}}section{{border-top:1px solid #ccc}}</style></head><body><h1>Read-only experiment audit</h1><pre>{html.escape(json.dumps(scientific, indent=2))}</pre>{sections}</body></html>"
    path.write_text(content, encoding="utf-8"); return path
