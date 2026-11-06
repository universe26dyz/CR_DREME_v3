"""Read-only experiment-audit visualization framework."""
from .adapters.exp1_reference import Exp1ReferenceAdapter


def adapter_for(name: str):
    if name == "exp1_reference":
        return Exp1ReferenceAdapter()
    raise ValueError(f"unknown experiment adapter: {name}")


__all__ = ["adapter_for", "Exp1ReferenceAdapter"]
