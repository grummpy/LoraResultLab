"""Repository paths and safe resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "sample_data"
ART_DIR = REPO_ROOT / "art"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def resolve_user_path(value: str, root: Path | None = None) -> Path:
    """Resolve a user-supplied path. Relative paths start at the repo root."""
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = (root or REPO_ROOT) / raw
    return raw.resolve()


def sandbox_child(root: Path, relative: str) -> Path:
    """Resolve `relative` and require it to stay inside `root`."""
    base = root.resolve()
    candidate = (base / relative).resolve()
    if candidate != base and os.path.commonpath([str(base), str(candidate)]) != str(base):
        raise ValueError("path escapes the allowed directory")
    return candidate
