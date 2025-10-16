"""Core package for the Kokoro Audiobook Maker."""

from __future__ import annotations

from .converter import AudiobookConverter, ConversionOptions, ConversionResult

__all__ = [
    "AudiobookConverter",
    "ConversionOptions",
    "ConversionResult",
]

__version__ = "0.1.0"
