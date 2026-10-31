"""Suggest archives for stale, duplicate, or under-documented LoRAs.

Suggestions never delete files. `deleted` is always an empty list.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _parse_day(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def analyze(loras: list[dict], usage: dict[str, str], *, unused_days: int, today: str) -> dict:
    if unused_days < 0:
        raise ValueError("unused_days must be >= 0")
    as_of = _parse_day(today)
    by_sha: dict[str, list[str]] = {}
    by_autov2: dict[str, list[str]] = {}
    for lora in loras:
        by_sha.setdefault(lora["sha256"], []).append(lora["path"])
        by_autov2.setdefault(lora["autov2"], []).append(lora["path"])

    duplicate_groups = []
    seen_groups = set()
    for hash_type, groups in (("sha256", by_sha), ("autov2", by_autov2)):
        for digest, paths in groups.items():
            unique_paths = sorted(set(paths))
            if len(unique_paths) < 2:
                continue
            key = (hash_type, digest, tuple(unique_paths))
            if key in seen_groups:
                continue
            seen_groups.add(key)
            duplicate_groups.append({"hash_type": hash_type, "hash": digest, "paths": unique_paths})

    sha_dups = {path for group in duplicate_groups if group["hash_type"] == "sha256" for path in group["paths"]}
    auto_dups = {path for group in duplicate_groups if group["hash_type"] == "autov2" for path in group["paths"]}

    suggestions = []
    for lora in sorted(loras, key=lambda item: item["filename"]):
        reasons = []
        filename = lora["filename"]
        last = usage.get(filename)
        if last is None:
            reasons.append("unused")
            age = None
        else:
            age = (as_of - _parse_day(last)).days
            if age >= unused_days:
                reasons.append("unused")
        if lora["path"] in sha_dups or lora["path"] in auto_dups:
            reasons.append("duplicate")
        if lora.get("missing_metadata"):
            reasons.append("missing_metadata")
        if not reasons:
            continue
        detail_parts = []
        if "unused" in reasons:
            if last is None:
                detail_parts.append("No dated provenance usage.")
            else:
                detail_parts.append(f"Last provenance use {last} ({age} days before {today}).")
        if "duplicate" in reasons:
            detail_parts.append("Shares a SHA256 or AutoV2 hash with another file.")
        if "missing_metadata" in reasons:
            detail_parts.append("Missing base model, trigger words, or the safetensors metadata block.")
        detail_parts.append("Suggested action: move it to an archive folder yourself. No file was deleted.")
        suggestions.append(
            {
                "path": lora["path"],
                "filename": filename,
                "sha256": lora["sha256"],
                "autov2": lora["autov2"],
                "reasons": reasons,
                "last_used": last,
                "action": "suggest_archive",
                "detail": " ".join(detail_parts),
            }
        )
    return {
        "unused_days": unused_days,
        "today": today,
        "deleted": [],
        "suggestions": suggestions,
        "duplicates": duplicate_groups,
        "counts": {
            "indexed": len(loras),
            "suggestions": len(suggestions),
            "duplicate_groups": len(duplicate_groups),
        },
    }


def files_still_present(loras: list[dict]) -> bool:
    return all(Path(lora["path"]).is_file() for lora in loras)
