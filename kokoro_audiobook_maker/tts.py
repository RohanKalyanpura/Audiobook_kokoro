"""Kokoro synthesis helpers and audio assembly utilities."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, ID3NoHeaderError, TIT2, TRCK
from mutagen.mp4 import MP4, MP4Cover
from pydub import AudioSegment

try:  # pragma: no cover - kokoro is optional for docs
    from kokoro import pipeline, voices as kokoro_voices
except Exception:  # pragma: no cover - fallback when kokoro not present
    pipeline = None
    kokoro_voices = None

from .ingest import Chapter

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


def list_available_voices() -> List[str]:
    """Return a list of available Kokoro voices (best-effort)."""

    if kokoro_voices is None:
        return ["af_heart"]
    if hasattr(kokoro_voices, "list_voices"):
        return sorted(kokoro_voices.list_voices())  # type: ignore[call-arg]
    if hasattr(kokoro_voices, "VOICES"):
        voices = getattr(kokoro_voices, "VOICES")
        if isinstance(voices, dict):
            return sorted(voices.keys())
        return sorted(voices)
    return ["af_heart"]


@dataclass
class ChapterAudio:
    """A synthesized chapter along with metadata."""

    chapter: Chapter
    path: Path
    duration_ms: int


class ChapterCache:
    """Filesystem cache that stores per-chapter renders."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        base = Path(base_dir or Path.home() / ".kokoro_audiobook_cache")
        base.mkdir(parents=True, exist_ok=True)
        self.base_dir = base

    def key(self, chapter: Chapter, voice: str, speed: float, pitch: float) -> str:
        payload = json.dumps(
            {
                "title": chapter.title,
                "voice": voice,
                "speed": speed,
                "pitch": pitch,
                "text_hash": hashlib.sha1(chapter.text.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def path_for(self, chapter: Chapter, voice: str, speed: float, pitch: float) -> Path:
        return self.base_dir / f"{self.key(chapter, voice, speed, pitch)}.wav"


class KokoroSynthesizer:
    """Render chapters using Kokoro with caching."""

    def __init__(
        self,
        *,
        cache: Optional[ChapterCache] = None,
        silence_padding_ms: int = 750,
    ) -> None:
        self.cache = cache or ChapterCache()
        self.silence_padding_ms = silence_padding_ms

    def synthesize_chapter(
        self,
        chapter: Chapter,
        *,
        voice: str,
        speed: float = 1.0,
        pitch: float = 0.0,
        progress: Optional[ProgressCallback] = None,
    ) -> ChapterAudio:
        cache_path = self.cache.path_for(chapter, voice, speed, pitch)
        if cache_path.exists():
            LOGGER.info("Using cached render for chapter '%s'", chapter.title)
            audio = AudioSegment.from_file(cache_path)
            if progress:
                progress(chapter.title, 1.0)
            return ChapterAudio(chapter=chapter, path=cache_path, duration_ms=len(audio))

        if pipeline is None:
            raise RuntimeError("kokoro pipeline is not available; install the kokoro package")

        LOGGER.info("Rendering chapter '%s' with voice %s", chapter.title, voice)
        generator = pipeline(
            chapter.text,
            voice=voice,
            speed=speed,
            pitch=pitch,
        )
        segments: List[AudioSegment] = []
        total_chunks = 0
        for total_chunks, (_, sample_rate, audio_chunk) in enumerate(generator, start=1):
            segments.append(_segment_from_chunk(audio_chunk, sample_rate))
            if progress and total_chunks:
                progress(chapter.title, min(0.95, total_chunks / max(total_chunks, 1)))
        audio = AudioSegment.silent(duration=self.silence_padding_ms)
        for segment in segments:
            audio += segment
        audio += AudioSegment.silent(duration=self.silence_padding_ms)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(cache_path, format="wav")
        if progress:
            progress(chapter.title, 1.0)
        return ChapterAudio(chapter=chapter, path=cache_path, duration_ms=len(audio))

    def synthesize_book(
        self,
        chapters: Sequence[Chapter],
        *,
        voice: str,
        speed: float = 1.0,
        pitch: float = 0.0,
        progress: Optional[ProgressCallback] = None,
    ) -> List[ChapterAudio]:
        rendered: List[ChapterAudio] = []
        chapter_list = list(chapters)
        total = len(chapter_list) or 1
        for idx, chapter in enumerate(chapter_list, start=1):
            chapter_progress: Optional[ProgressCallback]
            if progress:
                chapter_progress = lambda title, value, base=idx - 1: progress(
                    title, (base + value) / total
                )
            else:
                chapter_progress = None
            rendered.append(
                self.synthesize_chapter(
                    chapter,
                    voice=voice,
                    speed=speed,
                    pitch=pitch,
                    progress=chapter_progress,
                )
            )
        return rendered


def _segment_from_chunk(chunk, sample_rate: Optional[int]) -> AudioSegment:
    """Convert a Kokoro chunk to an AudioSegment."""

    import numpy as np  # Local import to keep dependency optional

    if isinstance(chunk, bytes):
        data = chunk
        sample_width = 2
    elif isinstance(chunk, np.ndarray):
        data = chunk.astype("int16").tobytes()
        sample_width = chunk.dtype.itemsize if chunk.dtype.itemsize <= 4 else 2
    else:  # pragma: no cover - best effort
        data = bytes(chunk)
        sample_width = 2
    return AudioSegment(
        data=data,
        frame_rate=sample_rate or 22050,
        sample_width=sample_width,
        channels=1,
    )


@dataclass
class AssemblyResult:
    path: Path
    chapter_offsets: List[Tuple[str, float]]


def assemble_audiobook(
    chapter_audio: Sequence[ChapterAudio],
    *,
    output_path: Path,
    output_format: str = "mp3",
    bitrate: str = "192k",
    sample_rate: int = 44100,
    metadata: Optional[Dict[str, str]] = None,
    silence_ms: int = 500,
    cover_art: Optional[bytes] = None,
) -> AssemblyResult:
    """Combine chapter audio into a single audiobook and write metadata."""

    combined = AudioSegment.silent(duration=0)
    chapter_offsets: List[Tuple[str, float]] = []
    elapsed = 0
    for chapter_audio_item in chapter_audio:
        audio = AudioSegment.from_file(chapter_audio_item.path)
        audio = audio.set_frame_rate(sample_rate)
        chapter_offsets.append((chapter_audio_item.chapter.title, elapsed / 1000))
        combined += audio
        elapsed += len(audio)
        combined += AudioSegment.silent(duration=silence_ms)
        elapsed += silence_ms
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_format = "mp3" if output_format.lower() == "mp3" else "mp4"
    combined.export(output_path, format=export_format, bitrate=bitrate)
    _write_metadata(
        output_path,
        output_format,
        metadata or {},
        chapter_offsets,
        cover_art,
    )
    return AssemblyResult(path=output_path, chapter_offsets=chapter_offsets)


def _write_metadata(
    output_path: Path,
    fmt: str,
    metadata: Dict[str, str],
    chapter_offsets: List[Tuple[str, float]],
    cover_art: Optional[bytes],
) -> None:
    if fmt.lower() == "mp3":
        _write_mp3_metadata(output_path, metadata, chapter_offsets)
    else:
        _write_m4b_metadata(output_path, metadata, chapter_offsets, cover_art)


def _write_mp3_metadata(
    output_path: Path,
    metadata: Dict[str, str],
    chapter_offsets: List[Tuple[str, float]],
) -> None:
    try:
        tags = EasyID3(output_path)
    except ID3NoHeaderError:
        tags = EasyID3()
    for key, value in metadata.items():
        tags[key] = value
    tags.save(output_path)
    if chapter_offsets:
        try:
            id3 = ID3(output_path)
        except ID3NoHeaderError:
            id3 = ID3()
        first_title, _ = chapter_offsets[0]
        id3.add(TIT2(encoding=3, text=first_title))
        id3.add(TRCK(encoding=3, text=str(len(chapter_offsets))))
        id3.save(output_path)


def _write_m4b_metadata(
    output_path: Path,
    metadata: Dict[str, str],
    chapter_offsets: List[Tuple[str, float]],
    cover_art: Optional[bytes],
) -> None:
    file = MP4(output_path)
    itunes_map = {
        "title": "\xa9nam",
        "artist": "\xa9ART",
        "album": "\xa9alb",
        "genre": "\xa9gen",
        "year": "\xa9day",
        "comment": "\xa9cmt",
    }
    for key, value in metadata.items():
        tag_key = itunes_map.get(key)
        if tag_key:
            file[tag_key] = [value]
    if cover_art:
        file["covr"] = [MP4Cover(cover_art, imageformat=MP4Cover.FORMAT_JPEG)]
    if chapter_offsets:
        chapters = []
        for title, offset_seconds in chapter_offsets:
            chapters.append((offset_seconds, title))
        file["chpl"] = [(title, offset) for title, offset in chapters]
    file.save()


__all__ = [
    "AssemblyResult",
    "ChapterAudio",
    "ChapterCache",
    "KokoroSynthesizer",
    "assemble_audiobook",
    "list_available_voices",
]
