"""Thin domain adapters around pinned third-party source implementations.

Import concrete modules directly.  Keeping this namespace lazy lets Phase-1
coverage load its NeSVoR primitive without also importing optional FiLM code.
"""
