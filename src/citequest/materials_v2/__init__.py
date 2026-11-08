"""S04 pure material contracts; no database, network or readiness promotion."""

from .text import MaterialError, MaterialLimits, TextMaterial, load_limits, prepare_text

__all__ = ["MaterialError", "MaterialLimits", "TextMaterial", "load_limits", "prepare_text"]
