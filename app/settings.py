"""Local settings. ComfyUI is off and the host is blank unless a person changes that."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Settings:
    comfyui_enabled: bool = False
    comfyui_host: str = ""

    def to_dict(self) -> dict:
        return {
            "comfyui_enabled": bool(self.comfyui_enabled),
            "comfyui_host": self.comfyui_host or "",
        }


def settings_path(data_dir: Path) -> Path:
    return data_dir / "settings.json"


def load_settings(data_dir: Path) -> Settings:
    path = settings_path(data_dir)
    if not path.exists():
        settings = Settings()
        save_settings(data_dir, settings)
        return settings
    data = json.loads(path.read_text(encoding="utf-8"))
    return Settings(
        comfyui_enabled=bool(data.get("comfyui_enabled", False)),
        comfyui_host=str(data.get("comfyui_host") or ""),
    )


def save_settings(data_dir: Path, settings: Settings) -> Settings:
    data_dir.mkdir(parents=True, exist_ok=True)
    # Only these two fields are persisted. Confirm is never stored.
    payload = {
        "comfyui_enabled": bool(settings.comfyui_enabled),
        "comfyui_host": str(settings.comfyui_host or ""),
    }
    settings_path(data_dir).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return Settings(**payload)


def update_settings(data_dir: Path, patch: dict) -> Settings:
    current = load_settings(data_dir)
    enabled = patch.get("comfyui_enabled", current.comfyui_enabled)
    host = patch.get("comfyui_host", current.comfyui_host)
    return save_settings(
        data_dir,
        Settings(comfyui_enabled=bool(enabled), comfyui_host=str(host or "").strip()),
    )
