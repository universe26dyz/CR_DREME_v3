"""Read-only experiment-audit visualization framework."""
from .adapters.exp1_reference import Exp1ReferenceAdapter
from .adapters.exp2a_reference_gauge import Exp2AReferenceGaugeAdapter


def adapter_for(name: str):
    if name == "exp1_reference":
        return Exp1ReferenceAdapter()
    if name == "exp2a_reference_gauge":
        return Exp2AReferenceGaugeAdapter()
    raise ValueError(f"unknown experiment adapter: {name}")


__all__ = ["adapter_for", "Exp1ReferenceAdapter", "Exp2AReferenceGaugeAdapter"]
