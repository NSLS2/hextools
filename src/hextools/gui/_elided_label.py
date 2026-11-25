"""A label that cuts its text short to fit a fixed number of lines."""

from __future__ import annotations

import html

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QTextLayout
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    """Label that wraps to at most ``max_lines``, ending with "…" if cut short.

    It is always ``max_lines`` tall and never asks for more width than it is
    given, so changing its text never resizes the layout around it. Cut-short
    text shows in full on hover, and :meth:`text` returns it.
    """

    def __init__(
        self, text: str = "", parent: QWidget | None = None, *, max_lines: int = 1
    ):
        super().__init__(parent)
        self._full_text = ""
        self._max_lines = max_lines
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setText(text)

    def text(self) -> str:
        return self._full_text

    def setText(self, text: str):
        self._full_text = text
        self._elide()

    def sizeHint(self) -> QSize:
        margins = self.contentsMargins()
        height = self.fontMetrics().lineSpacing() * self._max_lines
        return QSize(0, height + margins.top() + margins.bottom())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        text = self._full_text
        width = self.contentsRect().width()
        lines = self._wrap(text, width)
        cut = len(lines) > self._max_lines
        parts = [text[s : s + n].rstrip() for s, n in lines[: self._max_lines]]
        if cut:
            rest = text[lines[self._max_lines - 1][0] :]
            parts[-1] = self.fontMetrics().elidedText(
                rest, Qt.TextElideMode.ElideRight, width
            )
        super().setText("\n".join(parts))
        # Qt only wraps rich-text tooltips; plain ones are a single line.
        self.setToolTip(f"<p>{html.escape(text)}</p>" if cut else "")

    def _wrap(self, text: str, width: int) -> list[tuple[int, int]]:
        """``(start, length)`` of each line of ``text`` word-wrapped to ``width``."""
        layout = QTextLayout(text, self.font())
        layout.beginLayout()
        lines = []
        while (line := layout.createLine()).isValid():
            line.setLineWidth(max(width, 1))
            lines.append((line.textStart(), line.textLength()))
        layout.endLayout()
        return lines
