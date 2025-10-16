"""Text normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping, MutableMapping

DEFAULT_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "æ": "ae",
    "œ": "oe",
}

SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
}


@dataclass
class NormalizationOptions:
    """Configuration toggles for text normalization."""

    fix_hyphenation: bool = True
    normalize_quotes: bool = True
    replace_ligatures: bool = True
    strip_page_numbers: bool = True
    collapse_whitespace: bool = True
    custom_replacements: MutableMapping[str, str] = field(default_factory=dict)
    pronunciation_dict: MutableMapping[str, str] = field(default_factory=dict)


class Normalizer:
    """Normalize chapter text according to configured options."""

    def __init__(self, options: NormalizationOptions | None = None) -> None:
        self.options = options or NormalizationOptions()

    def normalize(self, text: str) -> str:
        if self.options.fix_hyphenation:
            text = self._fix_hyphenation(text)
        if self.options.normalize_quotes:
            text = self._normalize_quotes(text)
        if self.options.replace_ligatures:
            text = self._replace_ligatures(text)
        if self.options.strip_page_numbers:
            text = self._strip_page_numbers(text)
        if self.options.custom_replacements:
            text = self._apply_mapping(text, self.options.custom_replacements)
        if self.options.pronunciation_dict:
            text = self._apply_pronunciation(text, self.options.pronunciation_dict)
        if self.options.collapse_whitespace:
            text = self._collapse_whitespace(text)
        return text.strip()

    def _fix_hyphenation(self, text: str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"-\s+", "-", text)
        return text

    def _normalize_quotes(self, text: str) -> str:
        for src, dst in SMART_QUOTES.items():
            text = text.replace(src, dst)
        return text

    def _replace_ligatures(self, text: str) -> str:
        for src, dst in DEFAULT_LIGATURES.items():
            text = text.replace(src, dst)
        return text

    def _strip_page_numbers(self, text: str) -> str:
        lines = [line for line in text.splitlines() if not re.fullmatch(r"\d+", line.strip())]
        return "\n".join(lines)

    def _apply_mapping(self, text: str, mapping: Mapping[str, str]) -> str:
        pattern = re.compile("|".join(re.escape(k) for k in mapping.keys()))

        def repl(match: re.Match[str]) -> str:
            return mapping[match.group(0)]

        return pattern.sub(repl, text)

    def _apply_pronunciation(self, text: str, mapping: Mapping[str, str]) -> str:
        result = text
        for pattern, replacement in mapping.items():
            result = re.sub(pattern, replacement, result, flags=re.I)
        return result

    def _collapse_whitespace(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\t ]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text


__all__ = ["Normalizer", "NormalizationOptions"]
