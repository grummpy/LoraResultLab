"""SHA256 + AutoV2 index of local .safetensors files, stored in SQLite.

Scanning reads files and updates the catalog. It does not delete model files.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict, dict, int]:
    """Return (header, metadata, tensor_offset)."""
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path} has a truncated safetensors header")
    header = json.loads(header_bytes)
    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return header, metadata, 8 + header_len


def autov2_hash(path: Path) -> str:
    """First 10 hex chars of SHA256 over the tensor bytes, excluding the header.

    This matches the common Civitai AutoV2 safetensors hash: metadata edits do
    not change it, weight edits do.
    """
    _header, _meta, offset = read_safetensors_header(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:10]


def trigger_words(metadata: dict) -> list[str]:
    found: list[str] = []

    def add(value: str):
        text = value.strip()
        if text and text not in found:
            found.append(text)

    phrase = metadata.get("modelspec.trigger_phrase")
    if isinstance(phrase, str):
        add(phrase)
    raw_triggers = metadata.get("ss_trigger_words")
    if isinstance(raw_triggers, str):
        for part in raw_triggers.split(","):
            add(part)
    frequency = metadata.get("ss_tag_frequency")
    if isinstance(frequency, str) and frequency.strip():
        try:
            parsed = json.loads(frequency)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if isinstance(value, dict):
                    for tag in value.keys():
                        add(str(tag))
                else:
                    add(str(key))
    return found


def base_model(metadata: dict) -> str:
    for key in ("ss_base_model_version", "ss_sd_model_name", "modelspec.architecture"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def training_keys(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items() if str(key).startswith("ss_")}


def inspect_file(path: Path) -> dict:
    _header, metadata, _offset = read_safetensors_header(path)
    triggers = trigger_words(metadata)
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": file_sha256(path),
        "autov2": autov2_hash(path),
        "size_bytes": path.stat().st_size,
        "base_model": base_model(metadata),
        "trigger_words": triggers,
        "metadata": {str(key): str(value) for key, value in metadata.items()},
        "training_keys": {str(key): str(value) for key, value in training_keys(metadata).items()},
        "missing_metadata": not metadata or not base_model(metadata) or not triggers,
    }


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS loras (
            path TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            autov2 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            base_model TEXT,
            trigger_words TEXT,
            metadata_json TEXT,
            missing_metadata INTEGER NOT NULL,
            scanned_at TEXT NOT NULL
        )
        """
    )
    return connection


def scan_folder(folder: Path, db_path: Path) -> list[dict]:
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    files = sorted(folder.rglob("*.safetensors"))
    scanned_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = []
    prefix = str(folder.resolve())
    with connect(db_path) as connection:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        connection.execute(
            "DELETE FROM loras WHERE path = ? OR path LIKE ? ESCAPE '\\'",
            (prefix, escaped + "/%"),
        )
        for file_path in files:
            info = inspect_file(file_path)
            info["scanned_at"] = scanned_at
            connection.execute(
                """
                INSERT INTO loras (
                    path, filename, sha256, autov2, size_bytes, base_model,
                    trigger_words, metadata_json, missing_metadata, scanned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    info["path"],
                    info["filename"],
                    info["sha256"],
                    info["autov2"],
                    info["size_bytes"],
                    info["base_model"],
                    json.dumps(info["trigger_words"]),
                    json.dumps(info["metadata"]),
                    1 if info["missing_metadata"] else 0,
                    scanned_at,
                ),
            )
            rows.append(info)
        connection.commit()
    return rows


def list_indexed(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    with connect(db_path) as connection:
        found = connection.execute("SELECT * FROM loras ORDER BY filename, path").fetchall()
    rows = []
    for row in found:
        rows.append(
            {
                "path": row["path"],
                "filename": row["filename"],
                "sha256": row["sha256"],
                "autov2": row["autov2"],
                "size_bytes": row["size_bytes"],
                "base_model": row["base_model"] or "",
                "trigger_words": json.loads(row["trigger_words"] or "[]"),
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "missing_metadata": bool(row["missing_metadata"]),
                "scanned_at": row["scanned_at"],
            }
        )
    return rows
