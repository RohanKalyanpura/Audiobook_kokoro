"""Kokoro synthesis helpers and audio assembly utilities."""

from __future__ import annotations

import hashlib
import importlib
import inspect
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
    from kokoro import pipeline as _kokoro_pipeline  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - fallback when kokoro not present
    _kokoro_pipeline = None
    _pipeline_import_error: Optional[BaseException] = exc
except Exception as exc:  # pragma: no cover - unexpected runtime import error
    _kokoro_pipeline = None
    _pipeline_import_error = exc
else:  # pragma: no cover - pipeline successfully imported
    _pipeline_import_error = None

try:  # pragma: no cover - kokoro exposes optional helpers
    from kokoro import KPipeline as _kokoro_class  # type: ignore[attr-defined]
except ImportError:
    _kokoro_class = None
except Exception:  # pragma: no cover - defensive
    _kokoro_class = None

try:  # pragma: no cover - some releases expose a separate kpipeline helper
    from kokoro import kpipeline as _kokoro_kpipeline  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - helper not present
    _kokoro_kpipeline = None
except Exception:  # pragma: no cover - defensive
    _kokoro_kpipeline = None
else:  # pragma: no cover - helper successfully imported
    # ``kokoro.kpipeline`` may export either a callable or a class, so we
    # normalise it below together with the main pipeline import.
    pass

def _normalise_exports(module_like):
    """Return callable pipeline and class exports from a module-like object."""

    pipeline_candidate = getattr(module_like, "pipeline", None)
    kpipeline_candidate = getattr(module_like, "kpipeline", None)
    class_candidates = [
        getattr(module_like, name, None) for name in ("KPipeline", "Kpipeline", "kpipeline")
    ]

    pipeline_callable = None
    for candidate in (pipeline_candidate, kpipeline_candidate):
        if callable(candidate):
            pipeline_callable = candidate
            break

    class_export = next((c for c in class_candidates if inspect.isclass(c)), None)

    if pipeline_callable is None:
        pipeline_callable = next((c for c in class_candidates if callable(c)), None)

    return pipeline_callable, class_export


def _import_module_attribute(module_path: str, attr: str):
    """Attempt to import ``attr`` from ``module_path`` returning ``None`` on failure."""

    try:  # pragma: no cover - defensive import helper
        module = importlib.import_module(module_path)
    except Exception:
        return None
    return getattr(module, attr, None)


if _kokoro_pipeline is not None and not callable(_kokoro_pipeline):
    pipeline_callable, class_export = _normalise_exports(_kokoro_pipeline)
    _kokoro_pipeline = pipeline_callable
    if _kokoro_class is None:
        _kokoro_class = class_export

if _kokoro_pipeline is None and _kokoro_kpipeline is not None:
    if callable(_kokoro_kpipeline):
        _kokoro_pipeline = _kokoro_kpipeline
    else:
        pipeline_callable, class_export = _normalise_exports(_kokoro_kpipeline)
        _kokoro_pipeline = pipeline_callable
        if _kokoro_class is None:
            _kokoro_class = class_export

if _kokoro_pipeline is None:
    _kokoro_pipeline = _import_module_attribute("kokoro.pipeline", "pipeline")
    if callable(_kokoro_pipeline):
        _pipeline_import_error = None

if _kokoro_pipeline is None:
    _kokoro_pipeline = _import_module_attribute("kokoro.pipeline", "kpipeline")
    if callable(_kokoro_pipeline):
        _pipeline_import_error = None

if _kokoro_class is None:
    _kokoro_class = _import_module_attribute("kokoro", "KPipeline")

if _kokoro_class is None:
    _kokoro_class = _import_module_attribute("kokoro.pipeline", "KPipeline")

if _kokoro_pipeline is not None and not callable(_kokoro_pipeline):
    pipeline_callable, class_export = _normalise_exports(_kokoro_pipeline)
    _kokoro_pipeline = pipeline_callable
    if _kokoro_class is None:
        _kokoro_class = class_export

try:  # pragma: no cover
    from kokoro import voices as kokoro_voices  # type: ignore[attr-defined]
except ImportError:
    kokoro_voices = None
except Exception:
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
        lang_code: str = "a",
        device: Optional[str] = None,
        default_sample_rate: int = 24000,
    ) -> None:
        self.cache = cache or ChapterCache()
        self.silence_padding_ms = silence_padding_ms
        self.lang_code = lang_code
        self.device = device
        self.default_sample_rate = default_sample_rate
        self._pipeline_instance: Optional[Callable[..., Iterable[tuple]]] = None

    def synthesize_chapter(
        self,
        chapter: Chapter,
        *,
        voice: str,
        speed: float = 1.0,
        pitch: float = 0.0,
        use_cache: bool = True,
        progress: Optional[ProgressCallback] = None,
    ) -> ChapterAudio:
        cache_path = self.cache.path_for(chapter, voice, speed, pitch)
        if cache_path.exists():
            if use_cache:
                LOGGER.info("Using cached render for chapter '%s'", chapter.title)
                audio = AudioSegment.from_file(cache_path)
                if progress:
                    progress(chapter.title, 1.0)
                return ChapterAudio(chapter=chapter, path=cache_path, duration_ms=len(audio))
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass

        generator_factory = self._resolve_pipeline()

        LOGGER.info("Rendering chapter '%s' with voice %s", chapter.title, voice)
        generator = generator_factory(
            text=chapter.text,
            voice=voice,
            speed=speed,
            pitch=pitch,
        )
        segments: List[AudioSegment] = []
        total_chunks = 0
        for total_chunks, chunk in enumerate(generator, start=1):
            audio_chunk, sample_rate = _extract_audio_chunk(chunk, self.default_sample_rate)
            segments.append(_segment_from_chunk(audio_chunk, sample_rate))
            if progress and total_chunks:
                progress(
                    chapter.title,
                    min(0.95, total_chunks / (total_chunks + 1)),
                )
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
        use_cache: bool = True,
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
                    use_cache=use_cache,
                    progress=chapter_progress,
                )
            )
        return rendered

    def _resolve_pipeline(self) -> Callable[..., Iterable[tuple]]:
        """Return a callable that yields Kokoro audio chunks.

        Kokoro has shipped multiple public APIs (`pipeline`, `KPipeline`), and we
        support both to reduce friction for users. This helper lazily
        initialises whichever interface is available and raises a detailed
        error if neither can be located.
        """

        if callable(_kokoro_pipeline):
            signature_params = _signature_parameters(_kokoro_pipeline)

            def _call_pipeline(*, text: str, voice: str, speed: float, pitch: float):
                kwargs = _build_pipeline_kwargs(
                    signature_params,
                    voice=voice,
                    speed=speed,
                    pitch=pitch,
                    lang_code=self.lang_code,
                    device=self.device,
                    default_sample_rate=self.default_sample_rate,
                )
                return _kokoro_pipeline(text, **kwargs)  # type: ignore[misc]

            return _call_pipeline

        if _kokoro_class is not None:
            if self._pipeline_instance is None:
                init_params = _signature_parameters(_kokoro_class)
                init_kwargs = {}
                if self.lang_code is not None and "lang_code" in init_params:
                    init_kwargs["lang_code"] = self.lang_code
                if self.device is not None and "device" in init_params:
                    init_kwargs["device"] = self.device
                try:
                    self._pipeline_instance = _kokoro_class(**init_kwargs)
                except Exception as exc:  # pragma: no cover - runtime guard
                    raise RuntimeError(
                        "failed to initialise kokoro.KPipeline; check your kokoro installation"
                    ) from exc
            call_params = _signature_parameters(self._pipeline_instance)  # type: ignore[arg-type]

            def _call_instance(*, text: str, voice: str, speed: float, pitch: float):
                kwargs = _build_pipeline_kwargs(
                    call_params,
                    voice=voice,
                    speed=speed,
                    pitch=pitch,
                    lang_code=self.lang_code,
                    device=self.device,
                    default_sample_rate=self.default_sample_rate,
                )
                return self._pipeline_instance(  # type: ignore[misc]
                    text,
                    **kwargs,
                )

            return _call_instance

        message = "kokoro pipeline is not available; install the kokoro package"
        if _pipeline_import_error is not None:
            message = f"{message} ({_pipeline_import_error})"
        raise RuntimeError(message)


def _segment_from_chunk(chunk, sample_rate: Optional[int]) -> AudioSegment:
    """Convert a Kokoro chunk to an AudioSegment."""

    if isinstance(chunk, (bytes, bytearray)):
        data = bytes(chunk)
        sample_width = 2
    else:
        np = _require_numpy()
        if hasattr(chunk, "detach") and hasattr(chunk, "cpu"):
            array = chunk.detach().cpu().numpy()
        elif isinstance(chunk, np.ndarray):
            array = chunk
        else:
            array = np.asarray(chunk)
        if array.ndim != 1:
            array = array.reshape(-1)
        if array.dtype.kind == "f":
            max_int16 = float(np.iinfo(np.int16).max)
            array = (array.clip(-1.0, 1.0) * max_int16).astype(np.int16)
        elif array.dtype != np.int16:
            array = array.astype(np.int16)
        data = array.tobytes()
        sample_width = array.dtype.itemsize
    return AudioSegment(
        data=data,
        frame_rate=sample_rate or 22050,
        sample_width=sample_width,
        channels=1,
    )


def _extract_audio_chunk(chunk, default_sample_rate: int) -> Tuple[object, int]:
    """Extract the raw audio payload and sample rate from a Kokoro output chunk."""

    audio = None
    sample_rate: Optional[int] = None

    def _maybe_set_sample_rate(value: object) -> None:
        nonlocal sample_rate
        if sample_rate is not None:
            return
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        if isinstance(value, (int, float)) and 1000 <= int(value) <= 384000:
            sample_rate = int(value)

    if isinstance(chunk, dict):
        for key in ("audio", "audio_wav", "wav", "samples"):
            if key in chunk:
                audio = chunk[key]
                break
        for key in ("sample_rate", "sampleRate", "rate"):
            if key in chunk:
                _maybe_set_sample_rate(chunk[key])
    elif hasattr(chunk, "_asdict"):
        mapping = chunk._asdict()
        audio, sample_rate = _extract_audio_chunk(mapping, default_sample_rate)
    elif isinstance(chunk, (list, tuple)):
        for item in chunk:
            _maybe_set_sample_rate(item)
        for item in reversed(chunk):
            if _looks_like_audio(item):
                audio = item
                break
    else:
        audio = chunk

    if audio is None:
        raise RuntimeError("Kokoro pipeline yielded data without audio payload")

    return audio, sample_rate or default_sample_rate


def _looks_like_audio(value: object) -> bool:
    if isinstance(value, (bytes, bytearray)):
        return True
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy optional
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        return True
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return True
    if isinstance(value, (list, tuple)) and value:
        return isinstance(value[0], (int, float))
    return False


def _require_numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency missing
        raise RuntimeError(
            "numpy is required to process Kokoro audio output; install numpy to continue"
        ) from exc
    return np


def _signature_parameters(callable_obj: Callable[..., object]) -> Dict[str, inspect.Parameter]:
    target = callable_obj
    if inspect.isclass(callable_obj):
        target = callable_obj.__init__  # type: ignore[assignment]
    try:
        params = dict(inspect.signature(target).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins without signature
        return {}
    if inspect.isclass(callable_obj):
        params.pop("self", None)
    return params


def _build_pipeline_kwargs(
    parameters: Dict[str, inspect.Parameter],
    *,
    voice: str,
    speed: float,
    pitch: float,
    lang_code: Optional[str],
    device: Optional[str],
    default_sample_rate: int,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if "voice" in parameters:
        kwargs["voice"] = voice
    elif "speaker" in parameters:
        kwargs["speaker"] = voice
    if "speed" in parameters:
        kwargs["speed"] = speed
    elif "rate" in parameters:
        kwargs["rate"] = speed
    if "pitch" in parameters:
        kwargs["pitch"] = pitch
    if lang_code is not None:
        if "lang_code" in parameters:
            kwargs["lang_code"] = lang_code
        elif "language" in parameters:
            kwargs["language"] = lang_code
    if device is not None:
        if "device" in parameters:
            kwargs["device"] = device
        elif "torch_device" in parameters:
            kwargs["torch_device"] = device
    if "sample_rate" in parameters:
        kwargs.setdefault("sample_rate", default_sample_rate)
    if "sampling_rate" in parameters:
        kwargs.setdefault("sampling_rate", default_sample_rate)
    return kwargs


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
