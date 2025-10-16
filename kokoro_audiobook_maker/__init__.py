"""Kokoro Audiobook Maker package."""

from .app import main

__all__ = ["main"]
"""Core package for the Kokoro Audiobook Maker."""

from __future__ import annotations

from .converter import AudiobookConverter, ConversionOptions, ConversionResult

__all__ = [
    "AudiobookConverter",
    "ConversionOptions",
    "ConversionResult",
]

__version__ = "0.1.0"
