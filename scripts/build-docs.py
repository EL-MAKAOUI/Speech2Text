#!/usr/bin/env python3
"""Regenerate ``docs/guide.md`` from the guide built into the window.

The window's guide is the original; this file is the copy for reading on the
web. ``tests/test_guide.py`` fails when they disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech2text.ui.guide_window import as_markdown  # noqa: E402


def main() -> int:
    target = ROOT / "docs" / "guide.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(as_markdown(), encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
