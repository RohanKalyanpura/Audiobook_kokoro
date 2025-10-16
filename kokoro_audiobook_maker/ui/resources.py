"""Embedded resources for the Kokoro Audiobook Maker GUI."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap


@lru_cache(maxsize=1)
def app_icon() -> QIcon:
    """Return a simple generated application icon.

    The previous implementation attempted to decode an embedded PNG which was
    accidentally corrupted and triggered ``libpng`` errors on some systems. To
    keep the GUI lightweight and resilient we procedurally create a small
    rounded-square icon at runtime instead of depending on binary assets.
    """

    size = 96
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor("#0c1d2a"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#3cc9a7"))
    painter.setPen(QColor("#3cc9a7"))
    padding = size * 0.18
    painter.drawRoundedRect(
        padding,
        padding,
        size - 2 * padding,
        size - 2 * padding,
        size * 0.2,
        size * 0.2,
    )
    painter.setPen(QColor("#0c1d2a"))
    painter.setBrush(QColor("#0c1d2a"))
    painter.drawEllipse(size * 0.38, size * 0.34, size * 0.24, size * 0.32)
    painter.end()

    return QIcon(pixmap)


__all__ = ["app_icon"]
