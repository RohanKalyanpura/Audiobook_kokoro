"""Content ingestion helpers for Kokoro Audiobook Maker."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chapter:
    """Representation of a logical chapter extracted from a book."""

    index: int
    title: str
    text: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None


__all__ = ["Chapter"]
