"""Opening the review window on its own, from ``speech2text review``."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtWidgets

from .review_window import ReviewWindow


def main(bundle: str | Path) -> int:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Speech2Text")
    directory = Path(bundle)
    if directory.is_file():
        directory = directory.parent
    window = ReviewWindow(directory)
    window.show()
    return application.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1]))
