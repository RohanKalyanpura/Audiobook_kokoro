"""Kokoro Audiobook Maker package exports."""

from __future__ import annotations

from .app import main
from .converter import AudiobookConverter, ConversionOptions, ConversionResult

__all__ = [
    "main",
    "AudiobookConverter",
    "ConversionOptions",
    "ConversionResult",
]

__version__ = "0.1.0"
