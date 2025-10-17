from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_dummy_pyside6 = types.ModuleType("PySide6")
_dummy_qtwidgets = types.ModuleType("PySide6.QtWidgets")


class _FakeQApplication:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def exec(self) -> int:
        return 0


_dummy_qtwidgets.QApplication = _FakeQApplication
sys.modules.setdefault("PySide6", _dummy_pyside6)
sys.modules.setdefault("PySide6.QtWidgets", _dummy_qtwidgets)
setattr(_dummy_pyside6, "QtWidgets", _dummy_qtwidgets)


package_module = types.ModuleType("kokoro_audiobook_maker")
package_module.__path__ = [str(PROJECT_ROOT / "kokoro_audiobook_maker")]
sys.modules.setdefault("kokoro_audiobook_maker", package_module)


_CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "kokoro_audiobook_maker.converter",
    PROJECT_ROOT / "kokoro_audiobook_maker" / "converter.py",
)
converter = importlib.util.module_from_spec(_CONVERTER_SPEC)
assert _CONVERTER_SPEC and _CONVERTER_SPEC.loader
sys.modules.setdefault("kokoro_audiobook_maker.converter", converter)
_CONVERTER_SPEC.loader.exec_module(converter)

AudiobookConverter = converter.AudiobookConverter
Chapter = converter.Chapter
ConversionOptions = converter.ConversionOptions


def _make_options(tmp_path: Path) -> ConversionOptions:
    input_path = tmp_path / "input.txt"
    input_path.write_text("dummy", encoding="utf-8")
    output_path = tmp_path / "output.m4b"
    return ConversionOptions(input_path=input_path, output_path=output_path)


def test_chapter_cache_key_changes_when_text_changes(tmp_path):
    options = _make_options(tmp_path)
    converter = AudiobookConverter()

    chapter_a = Chapter(index=0, title="Test", text="Hello world")
    chapter_b = Chapter(index=0, title="Test", text="Hxllo worle")

    key_a = converter._chapter_cache_key(chapter_a, options)
    key_b = converter._chapter_cache_key(chapter_b, options)

    assert key_a != key_b


def test_chapter_cache_key_stable_for_same_text(tmp_path):
    options = _make_options(tmp_path)
    converter = AudiobookConverter()

    chapter = Chapter(index=1, title="Another", text="Identical text")

    key_first = converter._chapter_cache_key(chapter, options)
    key_second = converter._chapter_cache_key(chapter, options)

    assert key_first == key_second
