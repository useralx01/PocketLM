"""Development import shim for running pcketlm modules from the repo root."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "pcketlm"
__path__ = [str(_SRC_PACKAGE)]
