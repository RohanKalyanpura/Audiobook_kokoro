"""Application entry point for Kokoro Audiobook Maker.

This module supports both ``python -m kokoro_audiobook_maker.app`` and direct
execution via ``python kokoro_audiobook_maker/app.py``. The latter path is
commonly used on Windows by double-clicking the file in Explorer, which means
``__package__`` is ``None`` and the relative imports would normally fail. To
accommodate that workflow we patch ``sys.path`` in that scenario before
importing the main window class.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

if __package__ in {None, ""}:  # pragma: no cover - executed when run as a script
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from kokoro_audiobook_maker.ui.main_window import MainWindow
else:  # pragma: no cover - exercised when run as a module
    from .ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
