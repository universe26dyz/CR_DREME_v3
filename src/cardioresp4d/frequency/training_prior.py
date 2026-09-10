"""Validated adapter from Phase-1 aggregate frequency JSON to training bands."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

Band = tuple[float, float]
@dataclass(frozen=True)
class TrainingFrequencyPrior:
    respiratory_bands_hz: list[Band]; cardiac_bands_hz: list[Band]; baseline_bands_hz: list[Band]
    source_path: str; source_schema: str; provenance: dict

def _bands(value: object, label: str) -> list[Band]:
    if value is None: return []
    if not isinstance(value,list): raise ValueError(f'{label} must be a list of [low, high] bins')
    result=[(float(x[0]),float(x[1])) for x in value]
    if any(lo < 0 or hi <= lo for lo,hi in result): raise ValueError(f'{label} has invalid bins')
    return result

def load_training_frequency_prior(path: str|Path, *, allow_template_fallback: bool=False) -> TrainingFrequencyPrior:
    source=Path(path).resolve(); payload=json.loads(source.read_text(encoding='utf-8'))
    respiratory=_bands(payload.get('respiratory',{}).get('verified_band_hz'),'respiratory.verified_band_hz')
    if not respiratory and not allow_template_fallback: raise ValueError('Phase-1 frequency prior has no verified respiratory band')
    cardiac=_bands(payload.get('cardiac',{}).get('union_resolution_bins_hz'),'cardiac.union_resolution_bins_hz')
    if not cardiac: raise ValueError('Phase-1 frequency prior has no cardiac resolution bins')
    # Paper-derived necessary baseline: the nearest equally wide adjacent bin
    # outside every physiological band and DC, evaluated at the same precision.
    baseline=[]
    occupied=respiratory+cardiac
    for lo,hi in cardiac:
        width=hi-lo
        candidate=(max(width,lo-width),lo) if lo-width>0 else (hi,hi+width)
        if not any(candidate[0] < b and candidate[1] > a for a,b in occupied): baseline.append(candidate)
    if not baseline: raise ValueError('cannot construct non-physiological baseline bins')
    return TrainingFrequencyPrior(respiratory,cardiac,baseline,str(source),'phase1_aggregate_v1',{'baseline_rule':'paper-derived adjacent equal-width non-DC non-physiological bins'})
