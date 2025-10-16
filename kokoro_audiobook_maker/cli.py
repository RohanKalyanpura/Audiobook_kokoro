"""Command line interface for the Kokoro Audiobook Maker."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from .converter import AudiobookConverter, ConversionOptions
from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kokoro-book",
        description=(
            "Render an ebook into a chaptered audiobook using the shared Kokoro "
            "Audiobook Maker backend."
        ),
    )
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="Input EPUB/PDF/text file")
    parser.add_argument("--out", dest="output_path", type=Path, required=True, help="Destination audio file")
    parser.add_argument("--voice", default="af_heart", help="Voice identifier exposed by Kokoro")
    parser.add_argument("--format", choices=["m4b", "mp3"], default="m4b", help="Output container format")
    parser.add_argument("--bitrate", default="160k", help="Target bitrate for encoded audio")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target audio sample rate")
    parser.add_argument(
        "--chapters",
        default="auto",
        help="Chapter mode: auto, none, or a JSON file exported from the GUI",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "light", "standard", "strong"],
        default="standard",
        help="Text normalization strength",
    )
    parser.add_argument("--dict", dest="dictionary", type=Path, help="Optional pronunciation dictionary CSV")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory for per-chapter renders")
    parser.add_argument("--resume", action="store_true", help="Reuse cached chapters when available")
    parser.add_argument(
        "--meta",
        dest="metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra metadata to embed in the output (repeatable)",
    )
    parser.add_argument(
        "--silence-padding",
        type=int,
        default=250,
        help="Silence padding between chapters in milliseconds",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print warnings and errors")
    parser.add_argument("--version", action="version", version=f"kokoro-book {__version__}")
    return parser


def parse_metadata(pairs: Iterable[str]) -> dict:
    metadata = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid metadata entry (expected key=value): {pair}")
        key, value = pair.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def configure_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def resolve_chapter_mode(chapters_arg: str) -> tuple[str, Optional[Path]]:
    lowered = chapters_arg.lower()
    if lowered == "auto":
        return "auto", None
    if lowered == "none":
        return "none", None
    path = Path(chapters_arg)
    return "from-file", path


def create_options(namespace: argparse.Namespace) -> ConversionOptions:
    chapter_mode, chapter_file = resolve_chapter_mode(namespace.chapters)
    metadata = parse_metadata(namespace.metadata)
    options = ConversionOptions(
        input_path=namespace.input_path,
        output_path=namespace.output_path,
        voice=namespace.voice,
        format=namespace.format,
        bitrate=namespace.bitrate,
        speed=namespace.speed,
        sample_rate=namespace.sample_rate,
        normalize=namespace.normalize,
        chapter_mode=chapter_mode,
        chapter_file=chapter_file,
        pronunciation_dictionary=namespace.dictionary,
        cache_dir=namespace.cache_dir,
        resume=namespace.resume,
        metadata=metadata,
        silence_padding_ms=namespace.silence_padding,
    )
    return options


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        configure_logging(args.verbose, args.quiet)
        options = create_options(args)
        converter = AudiobookConverter()
        result = converter.convert(options)
    except Exception as exc:  # pragma: no cover - CLI safety net
        logging.getLogger(__name__).error(str(exc))
        return 1

    print(f"Wrote {result.generated_chapters} new chapters to {result.output_path}")
    if result.reused_chapters:
        print(f"Reused {result.reused_chapters} chapters from cache")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
