"""PDF ingestion utilities."""

from __future__ import annotations

import collections
import logging
import re
from typing import Dict, Iterable, List, Optional

from pypdf import PdfReader

from . import Chapter

LOGGER = logging.getLogger(__name__)


class PdfLoader:
    """Extract chapters from PDFs using heading heuristics."""

    def __init__(
        self,
        *,
        min_heading_length: int = 6,
        heading_patterns: Optional[Iterable[str]] = None,
        common_threshold: float = 0.4,
    ) -> None:
        self.min_heading_length = min_heading_length
        self.heading_patterns = [re.compile(p, re.I) for p in heading_patterns or []]
        self.common_threshold = common_threshold

    def load(self, path: str) -> List[Chapter]:
        reader = PdfReader(path)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        headers, footers = self._detect_repeated_lines(page_texts)

        chapters: List[Chapter] = []
        current_lines: List[str] = []
        current_title: Optional[str] = None
        current_start_page: Optional[int] = None

        def flush(page_index: int) -> None:
            nonlocal current_lines, current_title, current_start_page
            if current_title and current_lines:
                chapters.append(
                    Chapter(
                        index=len(chapters),
                        title=current_title,
                        text="\n".join(current_lines).strip(),
                        start_page=current_start_page,
                        end_page=page_index,
                    )
                )
            current_lines = []
            current_title = None
            current_start_page = None

        for page_index, text in enumerate(page_texts):
            filtered = self._strip_common_lines(text, headers, footers)
            lines = [line.strip() for line in filtered.splitlines() if line.strip()]
            for line in lines:
                if self._is_heading(line):
                    flush(page_index)
                    current_title = line
                    current_start_page = page_index + 1
                    continue
                current_lines.append(line)
            if current_title and current_start_page is None:
                current_start_page = page_index + 1
        flush(len(page_texts))

        if not chapters and page_texts:
            LOGGER.warning("PDF heading detection failed; creating single chapter.")
            chapters.append(
                Chapter(index=0, title="Full Book", text="\n\n".join(page_texts))
            )
        return chapters

    def _detect_repeated_lines(self, pages: List[str]) -> tuple[Dict[str, int], Dict[str, int]]:
        header_counts: Dict[str, int] = collections.Counter()
        footer_counts: Dict[str, int] = collections.Counter()
        for text in pages:
            lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
            if not lines:
                continue
            header_counts[lines[0]] += 1
            footer_counts[lines[-1]] += 1
        threshold = max(1, int(len(pages) * self.common_threshold))
        headers = {line: count for line, count in header_counts.items() if count >= threshold}
        footers = {line: count for line, count in footer_counts.items() if count >= threshold}
        return headers, footers

    def _strip_common_lines(
        self, text: str, headers: Dict[str, int], footers: Dict[str, int]
    ) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in headers or stripped in footers:
                continue
            lines.append(stripped)
        return "\n".join(lines)

    def _is_heading(self, line: str) -> bool:
        if len(line) < self.min_heading_length:
            return False
        if any(pattern.search(line) for pattern in self.heading_patterns):
            return True
        if line.isupper():
            return True
        if re.match(r"^(chapter|book|part|section)\b", line, flags=re.I):
            return True
        if re.match(r"^[IVXLCM]+\.?(\s+.+)?$", line):
            return True
        if re.match(r"^\d+\.?(\s+.+)?$", line):
            return True
        return False


__all__ = ["PdfLoader"]
