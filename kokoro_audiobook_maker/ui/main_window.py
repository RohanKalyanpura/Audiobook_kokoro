"""Qt GUI for Kokoro Audiobook Maker."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QModelIndex, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)

from ebooklib import ITEM_IMAGE, epub
from pypdf import PdfReader

from ..ingest import Chapter
from ..ingest.epub_loader import EpubLoader
from ..ingest.pdf_loader import PdfLoader
from ..text.normalize import NormalizationOptions, Normalizer
from ..tts import (
    ChapterAudio,
    ChapterCache,
    KokoroSynthesizer,
    assemble_audiobook,
    list_available_voices,
)
from .resources import app_icon


@dataclass
class BookJob:
    path: Path
    chapters: List[Chapter]
    metadata: Dict[str, str]
    cover_art: Optional[bytes] = None


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
        lang_code: str,
        output_dir: Path,
        output_format: str,
        template: str,
        bitrate: str,
        sample_rate: int,
        silence_ms: int,
        metadata_overrides: Dict[str, str],
        normalizer: Normalizer,
        cache_dir: Path,
        reuse_cache: bool,
    ) -> None:
        super().__init__()
        self.books = books
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.lang_code = lang_code
        self.output_dir = output_dir
        self.output_format = output_format
        self.template = template
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
        self.metadata_overrides = metadata_overrides
        self.normalizer = normalizer
        self.cache_dir = cache_dir
        self.reuse_cache = reuse_cache
        self.synthesizer = KokoroSynthesizer(
            cache=ChapterCache(self.cache_dir), silence_padding_ms=self.silence_ms
        )

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
                    use_cache=self.reuse_cache,
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
                    bitrate=self.bitrate,
                    sample_rate=self.sample_rate,
                    metadata=metadata,
                    silence_ms=self.silence_ms,
                    cover_art=book.cover_art,
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
        self.setAcceptDrops(True)
        self._cache_dir = ChapterCache().base_dir
        self._pronunciation_dict: Dict[str, str] = {}
        self._dictionary_path: Optional[Path] = None
        self._all_voices: List[str] = []
        self._build_ui()
        self.setWindowIcon(app_icon())
        self.books: List[BookJob] = []
        self._configure_widgets()
        self._refresh_voices()
        self._worker: Optional[RenderWorker] = None

    # ----- UI setup -----
    def _build_ui(self) -> None:
        self.setWindowTitle("Kokoro Audiobook Maker")
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)

        self.bookList = QListWidget(central)
        self.bookList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.bookList.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.bookList.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.bookList.setAlternatingRowColors(True)
        root_layout.addWidget(self.bookList)

        book_buttons = QHBoxLayout()
        self.addBookButton = QPushButton("Add Book…", central)
        self.removeBookButton = QPushButton("Remove Selected", central)
        self.clearBooksButton = QPushButton("Clear List", central)
        book_buttons.addWidget(self.addBookButton)
        book_buttons.addWidget(self.removeBookButton)
        book_buttons.addWidget(self.clearBooksButton)
        book_buttons.addStretch(1)
        root_layout.addLayout(book_buttons)

        options_layout = QFormLayout()
        self.voiceFilterEdit = QLineEdit(central)
        self.voiceFilterEdit.setPlaceholderText("Filter voices (e.g. en, female)")
        options_layout.addRow("Voice filter", self.voiceFilterEdit)

        self.voiceCombo = QComboBox(central)
        options_layout.addRow("Voice", self.voiceCombo)

        self.speedSpin = QDoubleSpinBox(central)
        self.speedSpin.setRange(0.5, 2.0)
        self.speedSpin.setSingleStep(0.05)
        self.speedSpin.setValue(1.0)
        options_layout.addRow("Speed", self.speedSpin)

        self.pitchSpin = QDoubleSpinBox(central)
        self.pitchSpin.setRange(-6.0, 6.0)
        self.pitchSpin.setSingleStep(0.5)
        self.pitchSpin.setValue(0.0)
        options_layout.addRow("Pitch", self.pitchSpin)

        self.formatCombo = QComboBox(central)
        self.formatCombo.addItems(["MP3", "M4B"])
        options_layout.addRow("Format", self.formatCombo)

        self.bitrateCombo = QComboBox(central)
        self.bitrateCombo.addItems(["128k", "160k", "192k", "256k", "320k"])
        self.bitrateCombo.setCurrentText("192k")
        options_layout.addRow("Bitrate", self.bitrateCombo)

        self.sampleRateCombo = QComboBox(central)
        self.sampleRateCombo.addItems(["22050", "44100", "48000"])
        self.sampleRateCombo.setCurrentText("44100")
        options_layout.addRow("Sample rate", self.sampleRateCombo)

        self.silenceSpin = QSpinBox(central)
        self.silenceSpin.setRange(0, 5000)
        self.silenceSpin.setSingleStep(50)
        self.silenceSpin.setValue(500)
        options_layout.addRow("Silence padding (ms)", self.silenceSpin)

        self.outputTemplateEdit = QLineEdit(central)
        self.outputTemplateEdit.setPlaceholderText("{author} - {title}")
        options_layout.addRow("Output template", self.outputTemplateEdit)

        self.resumeCheck = QCheckBox("Reuse cached chapters", central)
        self.resumeCheck.setChecked(True)
        options_layout.addRow("Caching", self.resumeCheck)

        cache_layout = QHBoxLayout()
        self.cacheDirEdit = QLineEdit(central)
        self.cacheDirEdit.setReadOnly(True)
        self.cacheDirButton = QPushButton("Browse…", central)
        cache_layout.addWidget(self.cacheDirEdit)
        cache_layout.addWidget(self.cacheDirButton)
        options_layout.addRow("Cache directory", cache_layout)

        root_layout.addLayout(options_layout)

        metadata_group = QGroupBox("Metadata overrides (optional)", central)
        metadata_layout = QFormLayout(metadata_group)
        self.titleOverrideEdit = QLineEdit(metadata_group)
        self.authorOverrideEdit = QLineEdit(metadata_group)
        self.albumOverrideEdit = QLineEdit(metadata_group)
        self.genreOverrideEdit = QLineEdit(metadata_group)
        self.yearOverrideEdit = QLineEdit(metadata_group)
        self.commentOverrideEdit = QLineEdit(metadata_group)
        metadata_layout.addRow("Title", self.titleOverrideEdit)
        metadata_layout.addRow("Author", self.authorOverrideEdit)
        metadata_layout.addRow("Album", self.albumOverrideEdit)
        metadata_layout.addRow("Genre", self.genreOverrideEdit)
        metadata_layout.addRow("Year", self.yearOverrideEdit)
        metadata_layout.addRow("Comment", self.commentOverrideEdit)
        root_layout.addWidget(metadata_group)

        normalization_group = QGroupBox("Text cleanup & pronunciation", central)
        normalization_layout = QVBoxLayout(normalization_group)

        cleanup_row = QHBoxLayout()
        self.hyphenCheck = QCheckBox("Fix hyphenation", normalization_group)
        self.hyphenCheck.setChecked(True)
        self.quotesCheck = QCheckBox("Normalize quotes", normalization_group)
        self.quotesCheck.setChecked(True)
        self.ligatureCheck = QCheckBox("Replace ligatures", normalization_group)
        self.ligatureCheck.setChecked(True)
        self.pageNumberCheck = QCheckBox("Strip page numbers", normalization_group)
        self.pageNumberCheck.setChecked(True)
        self.whitespaceCheck = QCheckBox("Collapse whitespace", normalization_group)
        self.whitespaceCheck.setChecked(True)
        cleanup_row.addWidget(self.hyphenCheck)
        cleanup_row.addWidget(self.quotesCheck)
        cleanup_row.addWidget(self.ligatureCheck)
        cleanup_row.addWidget(self.pageNumberCheck)
        cleanup_row.addWidget(self.whitespaceCheck)
        cleanup_row.addStretch(1)
        normalization_layout.addLayout(cleanup_row)

        self.customReplacementsEdit = QPlainTextEdit(normalization_group)
        self.customReplacementsEdit.setPlaceholderText(
            "pattern=replacement (one per line). Lines starting with # are ignored."
        )
        self.customReplacementsEdit.setFixedHeight(80)
        normalization_layout.addWidget(self.customReplacementsEdit)

        dictionary_row = QHBoxLayout()
        self.dictionaryPathEdit = QLineEdit(normalization_group)
        self.dictionaryPathEdit.setReadOnly(True)
        self.dictionaryButton = QPushButton("Load dictionary…", normalization_group)
        self.dictionaryClearButton = QPushButton("Clear", normalization_group)
        dictionary_row.addWidget(self.dictionaryPathEdit)
        dictionary_row.addWidget(self.dictionaryButton)
        dictionary_row.addWidget(self.dictionaryClearButton)
        normalization_layout.addLayout(dictionary_row)

        root_layout.addWidget(normalization_group)

        control_row = QHBoxLayout()
        self.previewButton = QPushButton("Preview Chapter", central)
        self.convertButton = QPushButton("Convert", central)
        control_row.addWidget(self.previewButton)
        control_row.addWidget(self.convertButton)
        control_row.addStretch(1)
        root_layout.addLayout(control_row)

        self.progressBar = QProgressBar(central)
        self.progressBar.setRange(0, 100)
        root_layout.addWidget(self.progressBar)

        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.cacheDirEdit.setText(str(self._cache_dir))

    def _configure_widgets(self) -> None:
        self.addBookButton.clicked.connect(self._add_book_dialog)
        self.removeBookButton.clicked.connect(self._remove_selected)
        self.clearBooksButton.clicked.connect(self._clear_books)
        self.previewButton.clicked.connect(self._preview_selected)
        self.convertButton.clicked.connect(self._convert_books)
        self.voiceFilterEdit.textChanged.connect(self._apply_voice_filter)
        self.cacheDirButton.clicked.connect(self._choose_cache_dir)
        self.dictionaryButton.clicked.connect(self._choose_dictionary)
        self.dictionaryClearButton.clicked.connect(self._clear_dictionary)
        self.progressBar.setValue(0)
        self.bookList.model().rowsMoved.connect(self._on_rows_moved)

    def _refresh_voices(self) -> None:
        voices = list_available_voices()
        if not voices:
            voices = ["af_heart"]
        self._all_voices = voices
        self._apply_voice_filter()

    def _apply_voice_filter(self) -> None:
        filter_text = self.voiceFilterEdit.text().strip().lower()
        current_voice = self.voiceCombo.currentText()
        self.voiceCombo.blockSignals(True)
        self.voiceCombo.clear()
        if not filter_text:
            filtered = list(self._all_voices)
        else:
            filtered = [voice for voice in self._all_voices if filter_text in voice.lower()]
        if not filtered:
            filtered = list(self._all_voices)
            self.statusbar.showMessage("No voices matched filter; showing all voices", 4000)
        self.voiceCombo.addItems(filtered)
        if current_voice in filtered:
            self.voiceCombo.setCurrentText(current_voice)
        self.voiceCombo.blockSignals(False)

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
                metadata, cover = self._extract_epub_metadata(book)
            elif path.suffix.lower() == ".pdf":
                loader = PdfLoader()
                chapters = loader.load(str(path))
                metadata = self._extract_pdf_metadata(path)
                cover = None
            else:
                QMessageBox.warning(self, "Unsupported", f"Unsupported file type: {path.suffix}")
                return
            job = BookJob(path=path, chapters=chapters, metadata=metadata, cover_art=cover)
            self.books.append(job)
            item = QListWidgetItem(f"{metadata.get('title', path.stem)} ({len(chapters)} chapters)")
            item.setData(Qt.UserRole, job)
            self.bookList.addItem(item)
            self.statusbar.showMessage(f"Loaded {path.name}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load", str(exc))

    def _extract_epub_metadata(self, book: epub.EpubBook) -> tuple[Dict[str, str], Optional[bytes]]:
        title = self._first_metadata(book, "title") or "Untitled EPUB"
        author = self._first_metadata(book, "creator") or "Unknown"
        year = self._first_metadata(book, "date") or ""
        metadata = {"title": title, "author": author, "year": year}
        cover = self._extract_epub_cover(book)
        return metadata, cover

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

    def _extract_epub_cover(self, book: epub.EpubBook) -> Optional[bytes]:
        cover_item = None
        try:
            cover_meta = book.get_metadata("OPF", "cover")
        except KeyError:
            cover_meta = []
        if cover_meta:
            cover_id = cover_meta[0][0]
            cover_item = book.get_item_with_id(cover_id)
        if cover_item is None:
            for item in book.get_items():
                if item.get_type() == ITEM_IMAGE and "cover" in item.get_name().lower():
                    cover_item = item
                    break
        if cover_item is None:
            return None
        try:
            return cover_item.get_content()
        except Exception:  # pragma: no cover - best effort
            return None

    def _remove_selected(self) -> None:
        for item in self.bookList.selectedItems():
            job: BookJob = item.data(Qt.UserRole)
            self.books.remove(job)
            row = self.bookList.row(item)
            self.bookList.takeItem(row)
        self.statusbar.showMessage("Removed selected books", 2000)

    def _clear_books(self) -> None:
        self.bookList.clear()
        self.books.clear()
        self.statusbar.showMessage("Cleared book list", 2000)

    # ----- Preview & conversion -----
    def _preview_selected(self) -> None:
        item = self.bookList.currentItem()
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
        try:
            synth = KokoroSynthesizer(
                cache=ChapterCache(self._cache_dir),
                silence_padding_ms=self.silenceSpin.value(),
            )
            audio = synth.synthesize_chapter(
                Chapter(index=chapter.index, title=chapter.title, text=normalized_text),
                voice=self.voiceCombo.currentText(),
                speed=self.speedSpin.value(),
                pitch=self.pitchSpin.value(),
                use_cache=self.resumeCheck.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Preview failed", str(exc))
            return
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
            voice=self.voiceCombo.currentText(),
            speed=self.speedSpin.value(),
            pitch=self.pitchSpin.value(),
            output_dir=Path(output_dir),
            output_format=self.formatCombo.currentText().lower(),
            template=self.outputTemplateEdit.text(),
            bitrate=self.bitrateCombo.currentText(),
            sample_rate=int(self.sampleRateCombo.currentText()),
            silence_ms=self.silenceSpin.value(),
            metadata_overrides=self._metadata_overrides(),
            normalizer=Normalizer(options),
            cache_dir=self._cache_dir,
            reuse_cache=self.resumeCheck.isChecked(),
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._worker = worker
        self.convertButton.setEnabled(False)
        worker.start()

    def _current_lang_code(self) -> str:
        data = self.languageCombo.currentData()
        if isinstance(data, str) and data:
            return data
        return "a"

    def _normalization_options(self) -> NormalizationOptions:
        options = NormalizationOptions(
            fix_hyphenation=self.hyphenCheck.isChecked(),
            normalize_quotes=self.quotesCheck.isChecked(),
            replace_ligatures=self.ligatureCheck.isChecked(),
            strip_page_numbers=self.pageNumberCheck.isChecked(),
            collapse_whitespace=self.whitespaceCheck.isChecked(),
            custom_replacements=self._parse_custom_replacements(),
            pronunciation_dict=dict(self._pronunciation_dict),
        )
        return options

    def _parse_custom_replacements(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        invalid: List[int] = []
        for idx, raw_line in enumerate(self.customReplacementsEdit.toPlainText().splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                invalid.append(idx)
                continue
            key, value = line.split("=", 1)
            mapping[key.strip()] = value.strip()
        if invalid:
            joined = ", ".join(str(i) for i in invalid)
            self.statusbar.showMessage(
                f"Skipped invalid replacements on lines: {joined}",
                5000,
            )
        return mapping

    def _metadata_overrides(self) -> Dict[str, str]:
        overrides = {
            "title": self.titleOverrideEdit.text().strip(),
            "author": self.authorOverrideEdit.text().strip(),
            "album": self.albumOverrideEdit.text().strip(),
            "genre": self.genreOverrideEdit.text().strip(),
            "year": self.yearOverrideEdit.text().strip(),
            "comment": self.commentOverrideEdit.text().strip(),
        }
        return {key: value for key, value in overrides.items() if value}

    @Slot(str, float)
    def _on_progress(self, message: str, value: float) -> None:
        self.statusbar.showMessage(message)
        self.progressBar.setValue(int(value * 100))

    @Slot(Path)
    def _on_finished(self, output_dir: Path) -> None:
        self.statusbar.showMessage(f"Finished! Files saved to {output_dir}", 5000)
        self.progressBar.setValue(0)
        self.convertButton.setEnabled(True)
        self._worker = None

    @Slot(str)
    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Conversion failed", message)
        self.progressBar.setValue(0)
        self.convertButton.setEnabled(True)
        self._worker = None

    def _choose_cache_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select cache directory", str(self._cache_dir)
        )
        if not directory:
            return
        self._cache_dir = Path(directory)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.cacheDirEdit.setText(str(self._cache_dir))
        self.statusbar.showMessage("Cache directory updated", 3000)

    def _choose_dictionary(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Choose pronunciation dictionary",
            str(Path.home()),
            "CSV files (*.csv);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            mapping = self._load_dictionary_from_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load dictionary", str(exc))
            return
        self._dictionary_path = path
        self._pronunciation_dict = mapping
        self.dictionaryPathEdit.setText(str(path))
        self.statusbar.showMessage(
            f"Loaded pronunciation dictionary with {len(mapping)} entries", 4000
        )

    def _load_dictionary_from_csv(self, path: Path) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    continue
                mapping[row[0].strip()] = row[1].strip()
        return mapping

    def _clear_dictionary(self) -> None:
        self._pronunciation_dict.clear()
        self._dictionary_path = None
        self.dictionaryPathEdit.clear()
        self.statusbar.showMessage("Cleared pronunciation dictionary", 3000)

    def _on_rows_moved(
        self,
        _parent: QModelIndex,
        _start: int,
        _end: int,
        _destination: QModelIndex,
        _row: int,
    ) -> None:
        self._sync_book_order()

    def _sync_book_order(self) -> None:
        ordered: List[BookJob] = []
        for index in range(self.bookList.count()):
            item = self.bookList.item(index)
            job = item.data(Qt.UserRole)
            if job:
                ordered.append(job)
        self.books = ordered

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        handled = False
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if not local_path:
                    continue
                path = Path(local_path)
                if path.suffix.lower() not in {".epub", ".pdf"}:
                    continue
                self._add_book(path)
                handled = True
        if handled:
            event.acceptProposedAction()
        else:
            super().dropEvent(event)



__all__ = ["MainWindow"]
