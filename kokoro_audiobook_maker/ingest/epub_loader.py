"""EPUB ingestion utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Iterable, List

import ebooklib
from ebooklib import epub

from . import Chapter

LOGGER = logging.getLogger(__name__)


@dataclass
class TocEntry:
    title: str
    href: str


class EpubLoader:
    """Extract chapters from EPUB files based on their table of contents."""

    def __init__(self, *, strip_empty: bool = True) -> None:
        self.strip_empty = strip_empty

    def load(self, path: str) -> List[Chapter]:
        """Load chapters from the EPUB at *path*."""

        book = epub.read_epub(path)
        toc_entries = list(self._flatten_toc(book.get_toc()))
        if not toc_entries:
            LOGGER.warning("EPUB has no explicit TOC, falling back to spine order.")
            toc_entries = self._spine_entries(book)

        chapters: List[Chapter] = []
        for idx, entry in enumerate(toc_entries):
            item = book.get_item_with_href(entry.href)
            if item is None:
                LOGGER.debug("Skipping TOC entry without href: %s", entry)
                continue
            text = self._html_to_text(item.get_content().decode("utf-8", errors="ignore"))
            if self.strip_empty and not text.strip():
                LOGGER.debug("Skipping empty chapter: %s", entry.title)
                continue
            chapters.append(
                Chapter(
                    index=idx,
                    title=entry.title.strip() or f"Chapter {idx + 1}",
                    text=text,
                )
            )
        return chapters

    def _flatten_toc(self, toc: Iterable) -> Iterable[TocEntry]:
        for node in toc:
            if isinstance(node, (list, tuple)) and node:
                first, *rest = node
                if hasattr(first, "title") and hasattr(first, "href"):
                    yield TocEntry(title=self._safe_title(first.title), href=first.href)
                if rest:
                    yield from self._flatten_toc(rest)
            elif hasattr(node, "title") and hasattr(node, "href"):
                yield TocEntry(title=self._safe_title(node.title), href=node.href)

    def _spine_entries(self, book: epub.EpubBook) -> List[TocEntry]:
        entries: List[TocEntry] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):  # type: ignore[name-defined]
            entries.append(TocEntry(title=item.get_name(), href=item.get_name()))
        return entries

    def _safe_title(self, value: str) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return re.sub(r"\s+", " ", value or "").strip()

    def _html_to_text(self, html: str) -> str:
        # Remove scripts/styles
        html = re.sub(r"<(script|style)[^>]*>.*?</\\1>", "", html, flags=re.S | re.I)
        # Replace breaks with newline
        html = re.sub(r"<br[^>]*>", "\n", html, flags=re.I)
        html = re.sub(r"</p>", "\n", html, flags=re.I)
        # Strip tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


__all__ = ["EpubLoader"]
