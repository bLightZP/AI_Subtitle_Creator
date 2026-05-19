"""Persistent settings for the desktop GUI."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(slots=True)
class GuiSettings:
    """Values persisted between GUI launches."""

    window_geometry: str | None = None
    default_model: str | None = None
    selected_model: str | None = None
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = ""
    task: str = "transcribe"
    priority: str = "Below normal"
    cpu_threads: str = "0"
    model_cache: str | None = None


def default_settings_path() -> Path:
    """Return the per-user GUI settings file path."""

    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "AI Subtitle Creator" / "settings.json"


def load_gui_settings(path: Path | None = None) -> GuiSettings:
    """Load settings from disk, returning defaults on errors."""

    settings_path = path or default_settings_path()
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return GuiSettings()
    except (OSError, ValueError, TypeError):
        return GuiSettings()
    if not isinstance(payload, dict):
        return GuiSettings()

    allowed_keys = {field.name for field in fields(GuiSettings)}
    values = {
        key: value
        for key, value in payload.items()
        if key in allowed_keys and (value is None or isinstance(value, str))
    }
    return GuiSettings(**values)


def save_gui_settings(settings: GuiSettings, path: Path | None = None) -> Path:
    """Write settings to disk and return the saved path."""

    settings_path = path or default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(settings), indent=2, sort_keys=True)
    temp_path = settings_path.with_suffix(".tmp")
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(settings_path)
    return settings_path
