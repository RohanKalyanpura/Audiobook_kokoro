"""Shared conversion logic used by both the GUI backend and the CLI.

This module intentionally keeps the heavy lifting of turning an ebook into a
chaptered "audiobook" in one place so the GUI and CLI can share behaviour.
The actual audio synthesis is obviously out of scope for this kata, but the
structure mirrors what a real implementation would look like and exercises the
interesting parts: chapter detection, normalization, caching, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import csv
import hashlib
import json
import logging
import os
import re
import textwrap
import time

__all__ = [
    "Chapter",
    "ConversionOptions",
    "ConversionResult",
    "AudiobookConverter",
]

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    """Representation of a chapter extracted from the source text."""

    index: int
    title: str
    text: str
    cached: bool = False


@dataclass
class ConversionOptions:
    """Options that control how an ebook is rendered to audio."""

    input_path: Path
    output_path: Path
    voice: str = "af_heart"
    format: str = "m4b"
    bitrate: str = "160k"
    speed: float = 1.0
    sample_rate: int = 22050
    normalize: str = "standard"  # none|light|standard|strong
    chapter_mode: str = "auto"  # auto|none|from-file
    chapter_file: Optional[Path] = None
    pronunciation_dictionary: Optional[Path] = None
    cache_dir: Optional[Path] = None
    resume: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    silence_padding_ms: int = 250

    def __post_init__(self) -> None:
        if self.chapter_mode not in {"auto", "none", "from-file"}:
            raise ValueError(f"Unsupported chapter mode: {self.chapter_mode}")
        if self.speed <= 0:
            raise ValueError("Speed must be positive")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.format not in {"m4b", "mp3"}:
            raise ValueError("format must be either 'm4b' or 'mp3'")


@dataclass
class ConversionResult:
    """Outcome returned after a conversion run."""

    output_path: Path
    chapters: List[Chapter]
    generated_chapters: int
    reused_chapters: int
    metadata: Dict[str, str]
    elapsed_seconds: float


class AudiobookConverter:
    """High level orchestrator that mimics the GUI backend pipeline."""

    def __init__(self, default_cache_dir: Optional[Path] = None) -> None:
        self.default_cache_dir = (
            default_cache_dir
            or Path(os.getenv("KOKORO_CACHE", Path.home() / ".cache" / "kokoro_audiobook_maker"))
        )

    # Public API -----------------------------------------------------------------
    def convert(self, options: ConversionOptions) -> ConversionResult:
        start_time = time.perf_counter()
        logger.debug("Starting conversion with options: %s", options)

        input_path = options.input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        cache_dir = options.cache_dir or self.default_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Using cache directory: %s", cache_dir)

        raw_text = self._load_text(input_path)
        logger.debug("Loaded %d characters from %s", len(raw_text), input_path)

        normalized_text = self._normalize_text(raw_text, level=options.normalize)
        logger.debug("Normalized text length: %d", len(normalized_text))

        dictionary = self._load_pronunciation_dictionary(options.pronunciation_dictionary)
        if dictionary:
            normalized_text = self._apply_dictionary(normalized_text, dictionary)
            logger.debug("Applied pronunciation dictionary with %d entries", len(dictionary))

        chapters = self._build_chapters(
            normalized_text,
            options.chapter_mode,
            options.chapter_file,
            options.metadata,
        )
        logger.info("Prepared %d chapters", len(chapters))

        generated = 0
        reused = 0
        rendered_audio: List[bytes] = []

        for chapter in chapters:
            cache_key = self._chapter_cache_key(chapter, options)
            cache_path = cache_dir / f"{cache_key}.bin"
            if options.resume and cache_path.exists():
                chapter.cached = True
                rendered_audio.append(cache_path.read_bytes())
                reused += 1
                logger.debug("Reused cached chapter %s", chapter.title)
                continue

            audio_bytes = self._synthesise_chapter(chapter, options)
            cache_path.write_bytes(audio_bytes)
            rendered_audio.append(audio_bytes)
            generated += 1
            logger.debug("Generated chapter %s", chapter.title)

        combined_audio = self._assemble_output(rendered_audio, chapters, options)
        output_path = self._write_output(
            combined_audio,
            chapters,
            options,
        )

        elapsed = time.perf_counter() - start_time
        logger.info("Finished conversion in %.2fs", elapsed)

        return ConversionResult(
            output_path=output_path,
            chapters=chapters,
            generated_chapters=generated,
            reused_chapters=reused,
            metadata=self._build_metadata(options, len(chapters)),
            elapsed_seconds=elapsed,
        )

    # Input handling --------------------------------------------------------------
    def _load_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Failed to decode %s as UTF-8; attempting latin-1", path)
            return path.read_text(encoding="latin-1")

    def _normalize_text(self, text: str, level: str) -> str:
        logger.debug("Normalizing text with level=%s", level)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if level in {"light", "standard", "strong"}:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.replace("“", '"').replace("”", '"').replace("’", "'")
            text = text.replace("–", "-")
        if level in {"standard", "strong"}:
            text = self._remove_hyphenation(text)
        if level == "strong":
            text = re.sub(r"\b\d+\b", " ", text)
        return text.strip()

    def _remove_hyphenation(self, text: str) -> str:
        return re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    def _load_pronunciation_dictionary(self, path: Optional[Path]) -> Dict[str, str]:
        if not path:
            return {}
        if not path.exists():
            raise FileNotFoundError(f"Pronunciation dictionary not found: {path}")
        logger.debug("Loading pronunciation dictionary from %s", path)
        dictionary: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    continue
                dictionary[row[0].strip()] = row[1].strip()
        return dictionary

    def _apply_dictionary(self, text: str, dictionary: Dict[str, str]) -> str:
        def replace(match: re.Match[str]) -> str:
            return dictionary.get(match.group(0), match.group(0))

        pattern = re.compile("|".join(re.escape(k) for k in dictionary.keys()))
        return pattern.sub(replace, text)

    # Chapter preparation ---------------------------------------------------------
    def _build_chapters(
        self,
        text: str,
        mode: str,
        chapter_file: Optional[Path],
        metadata: Dict[str, str],
    ) -> List[Chapter]:
        if mode == "none":
            return [Chapter(index=0, title=metadata.get("title", "Chapter 1"), text=text)]
        if mode == "from-file":
            if not chapter_file:
                raise ValueError("chapter_file must be provided when mode is 'from-file'")
            return self._load_chapters_from_file(chapter_file, text)
        return self._detect_chapters(text, metadata)

    def _load_chapters_from_file(self, path: Path, fallback_text: str) -> List[Chapter]:
        data = json.loads(path.read_text(encoding="utf-8"))
        chapters: List[Chapter] = []
        for index, entry in enumerate(data):
            title = entry.get("title", f"Chapter {index + 1}")
            chunk = entry.get("text")
            if chunk is None:
                start = entry.get("start", 0)
                end = entry.get("end")
                chunk = fallback_text[start:end]
            chapters.append(Chapter(index=index, title=title, text=chunk))
        if not chapters:
            raise ValueError(f"No chapters defined in {path}")
        return chapters

    def _detect_chapters(self, text: str, metadata: Dict[str, str]) -> List[Chapter]:
        headings = list(
            re.finditer(r"^(?:\s*)(?:Chapter|CHAPTER|#)\s+(.+)$", text, re.MULTILINE)
        )
        indices = [match.start() for match in headings]
        titles = [match.group(0).strip() for match in headings]
        if not indices:
            return [Chapter(index=0, title=metadata.get("title", "Chapter 1"), text=text)]

        indices.append(len(text))
        chapters: List[Chapter] = []
        for idx, start in enumerate(indices[:-1]):
            end = indices[idx + 1]
            title = titles[idx]
            chunk = text[start:end].strip()
            if not chunk:
                continue
            chapters.append(Chapter(index=idx, title=title, text=chunk))
        if not chapters:
            chapters.append(Chapter(index=0, title=metadata.get("title", "Chapter 1"), text=text))
        return chapters

    # Audio synthesis -------------------------------------------------------------
    def _chapter_cache_key(self, chapter: Chapter, options: ConversionOptions) -> str:
        fingerprint = "|".join(
            [
                str(options.input_path.resolve()),
                chapter.title,
                str(len(chapter.text)),
                options.voice,
                f"{options.speed:.2f}",
                options.format,
                options.bitrate,
            ]
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    def _synthesise_chapter(self, chapter: Chapter, options: ConversionOptions) -> bytes:
        """Fake synthesis that pretends to create audio from text.

        The data that is written to disk is a textual manifest. This keeps the
        exercise dependency free whilst still enabling caching and verifying the
        control flow in tests.
        """

        header = textwrap.dedent(
            f"""
            ### Kokoro Audiobook Maker ###
            title: {chapter.title}
            voice: {options.voice}
            speed: {options.speed}
            format: {options.format}
            bitrate: {options.bitrate}
            sample_rate: {options.sample_rate}
            silence_padding_ms: {options.silence_padding_ms}
            """
        ).strip()
        payload = f"{header}\n\n{chapter.text.strip()}\n"
        return payload.encode("utf-8")

    def _assemble_output(
        self,
        rendered_audio: Sequence[bytes],
        chapters: Sequence[Chapter],
        options: ConversionOptions,
    ) -> bytes:
        manifest = {
            "voice": options.voice,
            "format": options.format,
            "bitrate": options.bitrate,
            "speed": options.speed,
            "chapters": [
                {
                    "index": chapter.index,
                    "title": chapter.title,
                    "cached": chapter.cached,
                    "byte_length": len(audio),
                }
                for chapter, audio in zip(chapters, rendered_audio)
            ],
        }
        envelope = {
            "metadata": self._build_metadata(options, len(chapters)),
            "manifest": manifest,
            "payload": [audio.decode("utf-8") for audio in rendered_audio],
        }
        return json.dumps(envelope, indent=2).encode("utf-8")

    def _build_metadata(self, options: ConversionOptions, chapter_count: int) -> Dict[str, str]:
        meta = {
            "input": str(options.input_path),
            "voice": options.voice,
            "format": options.format,
            "speed": f"{options.speed:.2f}",
            "chapters": str(chapter_count),
        }
        meta.update(options.metadata)
        return meta

    def _write_output(
        self,
        combined_audio: bytes,
        chapters: Sequence[Chapter],
        options: ConversionOptions,
    ) -> Path:
        output_path = options.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(combined_audio)
        logger.info(
            "Wrote %d chapters (%s) to %s",
            len(chapters),
            options.format,
            output_path,
        )
        return output_path
