"""Tests for Kokoro pipeline resolution helper."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - numpy may be unavailable in test environment
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - lightweight numpy stub for tests
    fake_numpy = types.ModuleType("numpy")

    class _FakeScalarType:
        def __init__(self, kind: str, itemsize: int, *, min_value=None, max_value=None):
            self.kind = kind
            self.itemsize = itemsize
            self.min = min_value
            self.max = max_value

        def __call__(self, value):
            if self.kind == "i":
                return int(value)
            return float(value)

    _float32 = _FakeScalarType("f", 4)
    _int16 = _FakeScalarType("i", 2, min_value=-32768, max_value=32767)

    class _FakeNDArray(list):
        def __init__(self, data, dtype):
            super().__init__(data)
            self._dtype = dtype

        @property
        def dtype(self):
            return self._dtype

        @property
        def ndim(self):
            return 1

        def reshape(self, *_):
            return self

        def astype(self, dtype):
            target = dtype if isinstance(dtype, _FakeScalarType) else _float32
            return _FakeNDArray([target(x) for x in self], target)

        def clip(self, min_value, max_value):
            return _FakeNDArray(
                [max(min(x, max_value), min_value) for x in self],
                self._dtype,
            )

        def tobytes(self):
            if self._dtype is not _int16:
                raise TypeError("only int16 conversion supported in fake numpy")
            import struct

            return struct.pack("<" + "h" * len(self), *[int(x) for x in self])

        def __mul__(self, other):
            if isinstance(other, (int, float)):
                return _FakeNDArray([x * other for x in self], self._dtype)
            return NotImplemented

        __rmul__ = __mul__

    def _asarray(value):
        if isinstance(value, _FakeNDArray):
            return value
        if isinstance(value, (bytes, bytearray)):
            return fake_numpy.frombuffer(bytes(value), fake_numpy.int16)
        if isinstance(value, (list, tuple)):
            if not value:
                return _FakeNDArray([], _float32)
            data = []
            dtype = _float32
            for item in value:
                if isinstance(item, _FakeNDArray):
                    data.extend(item)
                    dtype = item.dtype
                else:
                    data.append(float(item))
            return _FakeNDArray(data, dtype)
        return _FakeNDArray([float(value)], _float32)

    def _concatenate(arrays):
        arrays = [_asarray(item) for item in arrays]
        if not arrays:
            return _FakeNDArray([], _float32)
        dtype = arrays[0].dtype
        data = []
        for array in arrays:
            data.extend(array)
        return _FakeNDArray(data, dtype)

    def _frombuffer(buffer, dtype):
        if dtype is not _int16:
            raise TypeError("fake numpy only supports int16 buffers")
        import struct

        count = len(buffer) // dtype.itemsize
        if count == 0:
            return _FakeNDArray([], dtype)
        values = struct.unpack("<" + "h" * count, buffer[: count * dtype.itemsize])
        return _FakeNDArray(list(values), dtype)

    def _zeros(length, dtype=None):
        target = dtype if isinstance(dtype, _FakeScalarType) else _float32
        return _FakeNDArray([target(0) for _ in range(length)], target)

    def _linspace(start, stop, num, dtype=None):
        if num <= 1:
            values = [start]
        else:
            step = (stop - start) / (num - 1)
            values = [start + step * i for i in range(num)]
        target = dtype if isinstance(dtype, _FakeScalarType) else _float32
        return _FakeNDArray([target(v) for v in values], target)

    class _FakeIInfo:
        def __init__(self, dtype):
            self.max = dtype.max

    def _iinfo(dtype):
        return _FakeIInfo(dtype)

    fake_numpy.ndarray = _FakeNDArray
    fake_numpy.asarray = _asarray
    fake_numpy.concatenate = _concatenate
    fake_numpy.frombuffer = _frombuffer
    fake_numpy.zeros = _zeros
    fake_numpy.linspace = _linspace
    fake_numpy.iinfo = _iinfo
    fake_numpy.int16 = _int16
    fake_numpy.float32 = _float32

    np = fake_numpy  # type: ignore
    sys.modules.setdefault("numpy", fake_numpy)

import pytest

_dummy_pyside6 = types.ModuleType("PySide6")
_dummy_qtwidgets = types.ModuleType("PySide6.QtWidgets")


class _FakeQApplication:  # pragma: no cover - helper used for import isolation
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple stub
        pass

    def exec(self) -> int:
        return 0


_dummy_qtwidgets.QApplication = _FakeQApplication
sys.modules.setdefault("PySide6", _dummy_pyside6)
sys.modules.setdefault("PySide6.QtWidgets", _dummy_qtwidgets)
setattr(_dummy_pyside6, "QtWidgets", _dummy_qtwidgets)


class _FakeEasyID3(dict):  # pragma: no cover - metadata stub
    def save(self, *args, **kwargs) -> None:
        return None


class _FakeID3(dict):  # pragma: no cover - metadata stub
    def add(self, frame) -> None:
        return None

    def save(self, *args, **kwargs) -> None:
        return None


class _FakeID3Frame:  # pragma: no cover - metadata stub
    def __init__(self, *args, **kwargs) -> None:
        pass


class _FakeMP4(dict):  # pragma: no cover - metadata stub
    def save(self) -> None:
        return None


class _FakeMP4Cover:  # pragma: no cover - metadata stub
    FORMAT_JPEG = object()

    def __init__(self, data, imageformat=None) -> None:
        self.data = data
        self.imageformat = imageformat


mutagen_module = types.ModuleType("mutagen")
mutagen_module.__path__ = []  # type: ignore[attr-defined]
mutagen_easyid3 = types.ModuleType("mutagen.easyid3")
mutagen_easyid3.EasyID3 = _FakeEasyID3
mutagen_id3 = types.ModuleType("mutagen.id3")
mutagen_id3.ID3 = _FakeID3
mutagen_id3.ID3NoHeaderError = type("ID3NoHeaderError", (Exception,), {})
mutagen_id3.TIT2 = _FakeID3Frame
mutagen_id3.TRCK = _FakeID3Frame
mutagen_mp4 = types.ModuleType("mutagen.mp4")
mutagen_mp4.MP4 = _FakeMP4
mutagen_mp4.MP4Cover = _FakeMP4Cover

sys.modules.setdefault("mutagen", mutagen_module)
sys.modules.setdefault("mutagen.easyid3", mutagen_easyid3)
sys.modules.setdefault("mutagen.id3", mutagen_id3)
sys.modules.setdefault("mutagen.mp4", mutagen_mp4)

class _FakeAudioSegment:
    def __init__(self, data: bytes | bytearray | None = None, *, frame_rate: int = 22050, sample_width: int = 2, channels: int = 1):
        payload = bytes(data or b"")
        self.data = payload
        self.frame_rate = frame_rate
        self.sample_width = sample_width or 2
        self.channels = channels or 1
        if payload:
            bytes_per_frame = max(self.sample_width, 1) * self.channels
            frames = max(len(payload) // bytes_per_frame, 1)
            self.duration_ms = max(int(frames / max(self.frame_rate, 1) * 1000), 1)
        else:
            self.duration_ms = 0

    @classmethod
    def silent(cls, duration: int = 0):
        segment = cls()
        segment.duration_ms = duration
        return segment

    def __len__(self) -> int:
        return int(self.duration_ms)

    def __add__(self, other):
        combined = _FakeAudioSegment()
        combined.duration_ms = self.duration_ms + getattr(other, "duration_ms", 0)
        return combined

    def __iadd__(self, other):
        self.duration_ms += getattr(other, "duration_ms", 0)
        return self

    def export(self, out_f, format: str = "wav", **kwargs):  # pragma: no cover - helper stub
        from pathlib import Path as _Path
        if hasattr(out_f, "write"):
            out_f.write(b"stub-audio")
            return out_f
        path = _Path(out_f)
        path.write_bytes(b"stub-audio")
        return path

    def set_frame_rate(self, frame_rate: int):
        clone = _FakeAudioSegment()
        clone.duration_ms = self.duration_ms
        return clone

    @classmethod
    def from_file(cls, path):  # pragma: no cover - helper stub
        segment = cls()
        segment.duration_ms = 1000
        return segment

pydub_module = types.ModuleType("pydub")
pydub_module.AudioSegment = _FakeAudioSegment
sys.modules.setdefault("pydub", pydub_module)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

package = types.ModuleType("kokoro_audiobook_maker")
package.__path__ = [str(PROJECT_ROOT / "kokoro_audiobook_maker")]
sys.modules.setdefault("kokoro_audiobook_maker", package)

ingest_spec = importlib.util.spec_from_file_location(
    "kokoro_audiobook_maker.ingest",
    PROJECT_ROOT / "kokoro_audiobook_maker" / "ingest" / "__init__.py",
    submodule_search_locations=[str(PROJECT_ROOT / "kokoro_audiobook_maker" / "ingest")],
)
ingest_module = importlib.util.module_from_spec(ingest_spec)
sys.modules["kokoro_audiobook_maker.ingest"] = ingest_module
ingest_spec.loader.exec_module(ingest_module)

tts_spec = importlib.util.spec_from_file_location(
    "kokoro_audiobook_maker.tts",
    PROJECT_ROOT / "kokoro_audiobook_maker" / "tts.py",
    submodule_search_locations=[str(PROJECT_ROOT / "kokoro_audiobook_maker")],
)
tts_module = importlib.util.module_from_spec(tts_spec)
sys.modules["kokoro_audiobook_maker.tts"] = tts_module
tts_spec.loader.exec_module(tts_module)

setattr(package, "ingest", ingest_module)
setattr(package, "tts", tts_module)

tts = tts_module
Chapter = ingest_module.Chapter


def _pcm_silence() -> bytes:
    """Return 16-bit PCM bytes for a tiny silent audio chunk."""

    return (b"\x00\x00" * 32)


def _stub_export(self, out_f, format="wav", **kwargs):  # pragma: no cover - helper stub
    if hasattr(out_f, "write"):
        out_f.write(b"stub-wav")
        return out_f
    path = Path(out_f)
    path.write_bytes(b"stub-wav")
    return path


@pytest.fixture(autouse=True)
def stub_audio_export(monkeypatch):
    monkeypatch.setattr(tts.AudioSegment, "export", _stub_export)


def test_pipeline_function_is_used_for_synthesis(monkeypatch, tmp_path):
    calls: List[Dict[str, object]] = []

    def fake_pipeline(text, *, voice, speed=1.0, pitch=0.0, sample_rate=None):
        calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "pitch": pitch,
                "sample_rate": sample_rate,
            }
        )
        yield {"audio": _pcm_silence(), "sample_rate": sample_rate or 22050}

    monkeypatch.setattr(tts, "_kokoro_pipeline", fake_pipeline)
    monkeypatch.setattr(tts, "_kokoro_class", None)

    synthesizer = tts.KokoroSynthesizer(
        cache=tts.ChapterCache(base_dir=tmp_path),
        default_sample_rate=16000,
    )
    chapter = Chapter(index=1, title="Intro", text="Hello world")

    result = synthesizer.synthesize_chapter(
        chapter,
        voice="af_heart",
        use_cache=False,
    )

    assert calls, "pipeline callable should be invoked"
    assert calls[0]["text"] == chapter.text
    assert calls[0]["voice"] == "af_heart"
    assert calls[0]["sample_rate"] == 16000
    assert result.duration_ms > 0
    assert result.path.exists()


def test_pipeline_handles_nested_audio_sequences(monkeypatch, tmp_path):
    def fake_pipeline(text, *, voice, speed=1.0, pitch=0.0, sample_rate=None):
        nested_audio = [
            np.linspace(-0.5, 0.5, 32, dtype=np.float32),
            [
                np.linspace(0.5, -0.5, 16, dtype=np.float32),
                np.zeros(16, dtype=np.float32),
            ],
        ]
        yield {
            "audio": nested_audio,
            "sample_rate": sample_rate or 16000,
        }

    monkeypatch.setattr(tts, "_kokoro_pipeline", fake_pipeline)
    monkeypatch.setattr(tts, "_kokoro_class", None)

    synthesizer = tts.KokoroSynthesizer(
        cache=tts.ChapterCache(base_dir=tmp_path),
        default_sample_rate=22050,
    )
    chapter = Chapter(index=3, title="Nested", text="Nested audio test")

    result = synthesizer.synthesize_chapter(
        chapter,
        voice="af_heart",
        use_cache=False,
    )

    assert result.duration_ms > 0
    assert result.path.exists()


def test_pipeline_class_alias_is_initialised(monkeypatch, tmp_path):
    class DummyPipeline:
        init_calls: List[Dict[str, object]] = []

        def __init__(self, *, lang_code=None, device=None):
            self.calls: List[Dict[str, object]] = []
            DummyPipeline.init_calls.append({"lang_code": lang_code, "device": device})

        def __call__(self, text, *, voice, speed=1.0, pitch=0.0, sample_rate=None):
            self.calls.append(
                {
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "pitch": pitch,
                    "sample_rate": sample_rate,
                }
            )
            yield {"audio": _pcm_silence(), "sample_rate": sample_rate or 24000}

    monkeypatch.setattr(tts, "_kokoro_pipeline", DummyPipeline)
    monkeypatch.setattr(tts, "_kokoro_class", None)

    synthesizer = tts.KokoroSynthesizer(
        cache=tts.ChapterCache(base_dir=tmp_path),
        lang_code="en",
        device="cpu",
        default_sample_rate=12000,
    )
    chapter = Chapter(index=2, title="Chapter", text="Another")

    result = synthesizer.synthesize_chapter(
        chapter,
        voice="bf_friendly",
        use_cache=False,
    )

    assert DummyPipeline.init_calls
    assert DummyPipeline.init_calls[0] == {"lang_code": "en", "device": "cpu"}
    assert synthesizer._pipeline_instance is not None
    assert synthesizer._pipeline_instance.calls[0]["text"] == chapter.text
    assert synthesizer._pipeline_instance.calls[0]["voice"] == "bf_friendly"
    assert synthesizer._pipeline_instance.calls[0]["sample_rate"] == 12000
    assert result.duration_ms > 0
    assert result.path.exists()
