"""Embedded resources for the Kokoro Audiobook Maker GUI."""

from __future__ import annotations

import base64
from functools import lru_cache

from PySide6.QtGui import QIcon, QPixmap

_ICON_DATA = (
    b"iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAALElEQVR42mNgGAXUBw4cOXLmHwY0"
    b"gBGIhAkGmCKBBYZAFUwGg1E6gFgAAAwB1JzUKGwAAAABJRU5ErkJggg=="
)


@lru_cache(maxsize=1)
def app_icon() -> QIcon:
    """Return the application icon."""

    pixmap = QPixmap()
    pixmap.loadFromData(base64.b64decode(_ICON_DATA), "PNG")
    return QIcon(pixmap)


__all__ = ["app_icon"]
