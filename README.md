
---

# Kokoro Audiobook Maker (GUI)

Turn your ebooks (PDF & EPUB) into clean, chaptered audiobooks with a simple desktop GUI powered by the **Kokoro** open-weight TTS model.

* ✅ PDF & EPUB input
* ✅ MP3 or M4B (chaptered) output
* ✅ Chapter extraction from TOC / headings
* ✅ Voice & language picker (Kokoro voices)
* ✅ Batch convert multiple books
* ✅ Per-chapter caching & resumable runs
* ✅ Cross-platform (Windows, macOS, Linux)

> Kokoro is an Apache-licensed, 82M-parameter TTS model that’s fast, high-quality, and easy to run locally.

---

## Demo (what you’ll do)

1. Open the app → 2) Add your `.pdf` or `.epub` → 3) Pick a voice → 4) Click **Convert** → 5) Get an audiobook with chapters.

---

## Table of Contents

* [Why Kokoro](#why-kokoro)
* [Install](#install)
* [Quick Start](#quick-start)
* [GUI Usage](#gui-usage)
* [Voices & Languages](#voices--languages)
* [Output Formats](#output-formats)
* [How It Works](#how-it-works)
* [CLI (optional)](#cli-optional)
* [Advanced Settings](#advanced-settings)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [License](#license)
* [Acknowledgements](#acknowledgements)

---

## Why Kokoro

* **Open-weight & Apache-licensed** — use locally, in research, or in products.
* **Small & fast** (≈82M params) yet **surprisingly natural** speech.
* **Multiple languages & voices** without cloud calls or API keys.

---

## Install

> Requirements: Python 3.9–3.12, FFmpeg in PATH (for M4B/MP3 muxing), ~3–4 GB free disk for model/voices.

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Core dependencies
pip install --upgrade pip wheel

# Kokoro inference lib
pip install kokoro

# GUI & ebook parsing
pip install pyside6 ebooklib pypdf unidecode regex tqdm sentencepiece

# Audio, chapters, tags
pip install pydub mutagen

# Optional: faster sentence splitting & language detection
pip install spacy langdetect
python -m spacy download en_core_web_sm

# 3) FFmpeg (system)
# macOS: brew install ffmpeg
# Windows: winget install Gyan.FFmpeg or download static build
# Linux: sudo apt-get install ffmpeg
```

The first run will download Kokoro weights automatically to your local cache.

---

## Quick Start

```bash
# Run the GUI
python -m kokoro_audiobook_maker.app
```

* Click **Add Book**, choose a `.pdf` or `.epub`.
* Pick **Voice**, **Output: MP3 or M4B**, and a **Save To** folder.
* Press **Convert**. Progress shows per-chapter synthesis.

---

## GUI Usage

**Main pane**

* **Book List**: drag-and-drop EPUB/PDF; reorder for batch.
* **Voice Picker**: filter by language & gender; preview a sample.
* **Output Options**: MP3 or M4B, bitrate (e.g., 128k/192k), sample rate (22.05/44.1 kHz).
* **Chapters**: auto-detected from EPUB TOC or PDF headings; edit/merge/split before converting.
* **Pronunciations**: add custom dictionary entries (e.g., “SQL → sequel”, names).
* **Normalize**: optional text cleanup (hyphen fixes, smart quotes, ligatures, page headers/footers).
* **Convert**: batch safe; resumes where you left off.

**Settings (gear icon)**

* Output folder template: `"{author} - {title} ({year})"`
* Max chars per chunk, overlap seconds, sentence splitter mode
* Silence padding between clips (ms)
* Voice speed & pitch (where supported)
* Cache location & size cap

---

## Voices & Languages

Kokoro includes multiple voices across English (US/UK), Japanese, Mandarin, Spanish, French, and more. The app discovers installed Kokoro voices and lists them with short **Preview** buttons so you can audition before converting. You can change voices per-book or per-chapter.

> Tip: Pick a steady, neutral voice for long-form listening; bump the speed slightly (+5–10%) if you like podcast pacing.

---

## Output Formats

* **M4B** (AAC inside MP4 with chapters) — ideal for Apple Books, SmartAudiobookPlayer, etc.
* **MP3** (ID3 tags & chapter frames) — widely compatible; some players show chapters as cue points.
* Writes cover art (from EPUB cover or embedded PDF image), author, title, year, series, and per-chapter titles.

---

## How It Works

```mermaid
flowchart LR
  A[Ebook (PDF/EPUB)] --> B(Text Extract)
  B --> C(Clean & Segment\n(sentences/chapters))
  C --> D[Kokoro TTS\n(voice, speed)]
  D --> E(Cache & Join)
  E --> F(Tag & Chapters\n(M4B/MP3))
  F --> G(Output Audiobook)
```

**Parsing**

* **EPUB**: uses `ebooklib` + HTML stripping; TOC drives chapters; extracts cover & metadata.
* **PDF**: uses `pypdf`; a heuristic finds headings (font size deltas, page bookmarks) to propose chapters.
* **Cleanup**: remove headers/footers, de-hyphenate line endings, normalize ligatures/quotes, fix page numbers.

**Segmentation**

* Sentence splitting with spaCy or regex; configurable max chunk length and overlap to keep prosody smooth.

**Synthesis**

* Streams text through Kokoro, writing per-chunk WAV, then concatenates and encodes (AAC/MP3).
* Per-chapter caching so retries don’t redo completed work.

---

## CLI (optional)

The package also exposes a simple CLI for automation:

```bash
kokoro-book \
  --in "books/Moby-Dick.epub" \
  --out "output/Moby-Dick.m4b" \
  --voice "af_heart" \
  --format m4b \
  --bitrate 160k \
  --speed 1.05 \
  --chapters auto \
  --normalize strong
```

* `--chapters auto|none|file.json` (custom chapter JSON `{start_ms,title}`)
* `--dict pronunciations.csv` (CSV: `pattern,replace`)
* `--resume` (skip already rendered chapters)

---

## Advanced Settings

* **Pronunciation Dictionary**: project-level CSV; regex supported for tricky names/terms.
* **PDF tuning**: heading min font delta, keep figure/table captions, ignore page footers containing `^\d+$`.
* **Silence & spacing**: pre/post roll per chapter (ms); pause injection after `: ; —` if desired.
* **Batch**: run multiple books; per-book overrides (voice, speed).
* **GPU**: If your environment supports it, Kokoro can leverage acceleration where available.

---

## Troubleshooting

* **No audio / silent output**
  Check FFmpeg install and that the chosen bitrate/sample rate are supported by your player.

* **Weird hyphenation or page numbers in speech**
  Increase normalization level, enable de-hyphenation, and strip standalone numbers.

* **PDF chapters are messy**
  Toggle “Use bookmarks if present”, bump the heading size threshold, or manually edit the chapter list.

* **Voice list is empty**
  Ensure `kokoro` installed successfully; first run may download model & voice assets (stay online for that run).

* **Choppy prosody**
  Lower max chunk size or increase chunk overlap; reduce speed slightly.

---

## Roadmap

* [ ] Built-in pronunciation editor with search/replace previews
* [ ] Inline SSML-like tags for emphasis & pauses
* [ ] Save/load project presets per book
* [ ] Whisper fallback to auto-generate chapter titles from headings
* [ ] E2E installer builds (MSI/DMG/AppImage)

---

## License

* App code: MIT (this repo)
* Kokoro model & inference library: Apache-2.0 (see upstream LICENSE)
* You are responsible for your ebook content rights and local jurisdiction compliance.

---

## Acknowledgements

* **Kokoro** model & inference library by **hexgrad**
* Community tooling around Kokoro and ONNX variants
* Ebook parsing via `ebooklib` and `pypdf`, tagging via `mutagen`

---

### Minimal code stub (for contributors)

```python
# kokoro_audiobook_maker/tts.py
from kokoro import pipeline

def synthesize(text: str, voice: str = "af_heart"):
    gen = pipeline(text, voice=voice)
    for _, _, audio in gen:
        yield audio  # bytes or numpy array depending on backend
```

```python
# kokoro_audiobook_maker/app.py
import sys
from PySide6.QtWidgets import QApplication
from .ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

---

### Dev tips

* Use a consistent **chunk size** (e.g., 1200–1800 chars) and **overlap** (200–300 ms).
* For long books, enable **per-chapter caching** and consider **M4B** for best chapter support.
* Keep a **pronunciation CSV** in your project root for recurring names/terms.

---

**References:** Kokoro’s official GitHub & PyPI pages (install & usage), model card/voices on Hugging Face. ([GitHub][1])

---
## Sources
[1]: https://github.com/hexgrad/kokoro "GitHub - hexgrad/kokoro:

---