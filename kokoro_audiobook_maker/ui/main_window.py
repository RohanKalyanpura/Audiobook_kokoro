"""Qt GUI for Kokoro Audiobook Maker."""

from __future__ import annotations

import importlib.resources as resources
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QThread, Signal, Slot, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtUiTools import loadUi
from PySide6.QtWidgets import (
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
)

from ebooklib import epub
from pypdf import PdfReader

from ..ingest import Chapter
from ..ingest.epub_loader import EpubLoader
from ..ingest.pdf_loader import PdfLoader
from ..text.normalize import NormalizationOptions, Normalizer
from ..tts import ChapterAudio, KokoroSynthesizer, assemble_audiobook, list_available_voices
from .resources import app_icon


@dataclass
class BookJob:
    path: Path
    chapters: List[Chapter]
    metadata: Dict[str, str]


class RenderWorker(QThread):
    progress = Signal(str, float)
    finished = Signal(Path)
    error = Signal(str)

    def __init__(
        self,
        *,
        books: List[BookJob],
        voice: str,
        speed: float,
        pitch: float,
        output_dir: Path,
        output_format: str,
        template: str,
        silence_ms: int,
        metadata_overrides: Dict[str, str],
        normalizer: Normalizer,
    ) -> None:
        super().__init__()
        self.books = books
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.output_dir = output_dir
        self.output_format = output_format
        self.template = template
        self.silence_ms = silence_ms
        self.metadata_overrides = metadata_overrides
        self.normalizer = normalizer
        self.synthesizer = KokoroSynthesizer()

    def run(self) -> None:  # pragma: no cover - executed in thread
        try:
            total_books = len(self.books) or 1
            for idx, book in enumerate(self.books, start=1):
                normalized = [
                    Chapter(
                        index=chapter.index,
                        title=chapter.title,
                        text=self.normalizer.normalize(chapter.text),
                        start_page=chapter.start_page,
                        end_page=chapter.end_page,
                    )
                    for chapter in book.chapters
                ]
                chapter_audio: List[ChapterAudio] = self.synthesizer.synthesize_book(
                    normalized,
                    voice=self.voice,
                    speed=self.speed,
                    pitch=self.pitch,
                    progress=lambda title, value, base=idx - 1: self.progress.emit(
                        f"{book.metadata.get('title', book.path.stem)} – {title}",
                        (base + value) / total_books,
                    ),
                )
                output_path = self._build_output_path(book)
                metadata = {**book.metadata, **self.metadata_overrides}
                assemble_audiobook(
                    chapter_audio,
                    output_path=output_path,
                    output_format=self.output_format,
                    metadata=metadata,
                    silence_ms=self.silence_ms,
                )
            self.finished.emit(self.output_dir)
        except Exception as exc:  # pragma: no cover - error path
            self.error.emit(str(exc))

    def _build_output_path(self, book: BookJob) -> Path:
        template = self.template or "{title}"
        safe_template = template.format(
            title=book.metadata.get("title", book.path.stem),
            author=book.metadata.get("author", "Unknown"),
            year=book.metadata.get("year", ""),
        )
        safe_name = "".join(c for c in safe_template if c not in '<>:"/\\|?*')
        suffix = ".mp3" if self.output_format.lower() == "mp3" else ".m4b"
        return self.output_dir / f"{safe_name}{suffix}"


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self._load_ui()
        self.setWindowIcon(app_icon())
        self.books: List[BookJob] = []
        self._configure_widgets()
        self._refresh_voices()
        self._worker: Optional[RenderWorker] = None

    # ----- UI setup -----
    def _load_ui(self) -> None:
        ui_path = resources.files("kokoro_audiobook_maker.ui").joinpath("main_window.ui")
        with resources.as_file(ui_path) as path:
            loadUi(path, self)
        status = self.findChild(QStatusBar, "statusbar")
        self.statusbar = status if status else self.statusBar()

    def _configure_widgets(self) -> None:
        self.bookList: QListWidget = self.findChild(QListWidget, "bookList")
        self.addBookButton: QPushButton = self.findChild(QPushButton, "addBookButton")
        self.removeBookButton: QPushButton = self.findChild(QPushButton, "removeBookButton")
        self.previewButton: QPushButton = self.findChild(QPushButton, "previewButton")
        self.convertButton: QPushButton = self.findChild(QPushButton, "convertButton")
        self.formatCombo: QComboBox = self.findChild(QComboBox, "formatCombo")
        self.voiceCombo: QComboBox = self.findChild(QComboBox, "voiceCombo")
        self.speedSpin: QDoubleSpinBox = self.findChild(QDoubleSpinBox, "speedSpin")
        self.pitchSpin: QDoubleSpinBox = self.findChild(QDoubleSpinBox, "pitchSpin")
        self.outputTemplateEdit: QLineEdit = self.findChild(QLineEdit, "outputTemplateEdit")
        self.hyphenCheck: QCheckBox = self.findChild(QCheckBox, "hyphenCheck")
        self.quotesCheck: QCheckBox = self.findChild(QCheckBox, "quotesCheck")
        self.ligatureCheck: QCheckBox = self.findChild(QCheckBox, "ligatureCheck")
        self.pageNumberCheck: QCheckBox = self.findChild(QCheckBox, "pageNumberCheck")
        self.progressBar: QProgressBar = self.findChild(QProgressBar, "progressBar")

        self.addBookButton.clicked.connect(self._add_book_dialog)
        self.removeBookButton.clicked.connect(self._remove_selected)
        self.previewButton.clicked.connect(self._preview_selected)
        self.convertButton.clicked.connect(self._convert_books)
        self.formatCombo.addItems(["MP3", "M4B"])
        self.outputTemplateEdit.setPlaceholderText("{author} - {title}")
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)

    def _refresh_voices(self) -> None:
        voices = list_available_voices()
        self.voiceCombo.clear()  # type: ignore[attr-defined]
        self.voiceCombo.addItems(voices)  # type: ignore[attr-defined]

    # ----- Book management -----
    def _add_book_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select EPUB or PDF",
            str(Path.home()),
            "Books (*.epub *.pdf)",
        )
        for path in paths:
            self._add_book(Path(path))

    def _add_book(self, path: Path) -> None:
        try:
            if path.suffix.lower() == ".epub":
                loader = EpubLoader()
                chapters = loader.load(str(path))
                book = epub.read_epub(str(path))
                metadata = self._extract_epub_metadata(book)
            elif path.suffix.lower() == ".pdf":
                loader = PdfLoader()
                chapters = loader.load(str(path))
                metadata = self._extract_pdf_metadata(path)
            else:
                QMessageBox.warning(self, "Unsupported", f"Unsupported file type: {path.suffix}")
                return
            job = BookJob(path=path, chapters=chapters, metadata=metadata)
            self.books.append(job)
            item = QListWidgetItem(f"{metadata.get('title', path.stem)} ({len(chapters)} chapters)")
            item.setData(Qt.UserRole, job)
            self.bookList.addItem(item)  # type: ignore[attr-defined]
            self.statusbar.showMessage(f"Loaded {path.name}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", str(exc))

    def _extract_epub_metadata(self, book: epub.EpubBook) -> Dict[str, str]:
        title = self._first_metadata(book, "title") or "Untitled EPUB"
        author = self._first_metadata(book, "creator") or "Unknown"
        year = self._first_metadata(book, "date") or ""
        return {"title": title, "author": author, "year": year}

    def _first_metadata(self, book: epub.EpubBook, key: str) -> Optional[str]:
        values = book.get_metadata("DC", key)
        if values:
            value = values[0][0]
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore")
            return str(value)
        return None

    def _extract_pdf_metadata(self, path: Path) -> Dict[str, str]:
        reader = PdfReader(str(path))
        info = reader.metadata or {}
        title = str(info.get("/Title", path.stem))
        author = str(info.get("/Author", "Unknown"))
        year = str(info.get("/CreationDate", ""))[2:6] if info.get("/CreationDate") else ""
        return {"title": title, "author": author, "year": year}

    def _remove_selected(self) -> None:
        for item in self.bookList.selectedItems():  # type: ignore[attr-defined]
            job: BookJob = item.data(Qt.UserRole)
            self.books.remove(job)
            row = self.bookList.row(item)  # type: ignore[attr-defined]
            self.bookList.takeItem(row)  # type: ignore[attr-defined]
        self.statusbar.showMessage("Removed selected books", 2000)

    # ----- Preview & conversion -----
    def _preview_selected(self) -> None:
        item = self.bookList.currentItem()  # type: ignore[attr-defined]
        if not item:
            QMessageBox.information(self, "Preview", "Select a book first")
            return
        job: BookJob = item.data(Qt.UserRole)
        if not job.chapters:
            QMessageBox.warning(self, "Preview", "No chapters to preview")
            return
        chapter = job.chapters[0]
        normalizer = Normalizer(self._normalization_options())
        normalized_text = normalizer.normalize(chapter.text[:1500])
        synth = KokoroSynthesizer()
        audio = synth.synthesize_chapter(
            Chapter(index=chapter.index, title=chapter.title, text=normalized_text),
            voice=self.voiceCombo.currentText(),  # type: ignore[attr-defined]
            speed=self.speedSpin.value(),  # type: ignore[attr-defined]
            pitch=self.pitchSpin.value(),  # type: ignore[attr-defined]
        )
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(audio.path)))
        self.statusbar.showMessage("Preview rendered", 3000)

    def _convert_books(self) -> None:
        if not self.books:
            QMessageBox.information(self, "Convert", "Add at least one book")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory", str(Path.home()))
        if not output_dir:
            return
        options = self._normalization_options()
        worker = RenderWorker(
            books=list(self.books),
            voice=self.voiceCombo.currentText(),  # type: ignore[attr-defined]
            speed=self.speedSpin.value(),  # type: ignore[attr-defined]
            pitch=self.pitchSpin.value(),  # type: ignore[attr-defined]
            output_dir=Path(output_dir),
            output_format=self.formatCombo.currentText().lower(),  # type: ignore[attr-defined]
            template=self.outputTemplateEdit.text(),  # type: ignore[attr-defined]
            silence_ms=500,
            metadata_overrides={},
            normalizer=Normalizer(options),
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._worker = worker
        self.convertButton.setEnabled(False)  # type: ignore[attr-defined]
        worker.start()

    def _normalization_options(self) -> NormalizationOptions:
        return NormalizationOptions(
            fix_hyphenation=self.hyphenCheck.isChecked(),  # type: ignore[attr-defined]
            normalize_quotes=self.quotesCheck.isChecked(),  # type: ignore[attr-defined]
            replace_ligatures=self.ligatureCheck.isChecked(),  # type: ignore[attr-defined]
            strip_page_numbers=self.pageNumberCheck.isChecked(),  # type: ignore[attr-defined]
        )

    @Slot(str, float)
    def _on_progress(self, message: str, value: float) -> None:
        self.statusbar.showMessage(message)
        self.progressBar.setValue(int(value * 100))  # type: ignore[attr-defined]

    @Slot(Path)
    def _on_finished(self, output_dir: Path) -> None:
        self.statusbar.showMessage(f"Finished! Files saved to {output_dir}", 5000)
        self.progressBar.setValue(0)  # type: ignore[attr-defined]
        self.convertButton.setEnabled(True)  # type: ignore[attr-defined]
        self._worker = None

    @Slot(str)
    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Conversion failed", message)
        self.progressBar.setValue(0)  # type: ignore[attr-defined]
        self.convertButton.setEnabled(True)  # type: ignore[attr-defined]
        self._worker = None


__all__ = ["MainWindow"]
