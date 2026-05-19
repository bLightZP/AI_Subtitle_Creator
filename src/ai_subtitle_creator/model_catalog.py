"""Model catalog and download helpers for faster-whisper."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

MODEL_DESCRIPTIONS = {
    "tiny": "Fastest, lowest accuracy, good for smoke tests.",
    "tiny.en": "Fastest English-only model.",
    "base": "Fast, light, better than tiny.",
    "base.en": "Fast English-only base model.",
    "small": "Good default for CPU quality and speed.",
    "small.en": "English-only small model.",
    "medium": "Higher quality, slower and larger.",
    "medium.en": "English-only medium model.",
    "large-v3": "High quality, large download and slower CPU runs.",
    "large": "Alias for large-v3.",
    "large-v3-turbo": "Fast large-v3 turbo variant.",
    "turbo": "Alias for large-v3-turbo.",
    "distil-small.en": "Distilled English-only small model.",
    "distil-medium.en": "Distilled English-only medium model.",
    "distil-large-v2": "Distilled large-v2 model.",
    "distil-large-v3": "Distilled large-v3 model.",
    "distil-large-v3.5": "Distilled large-v3.5 model.",
}

DEFAULT_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
    "distil-large-v3.5",
]
IMPORTED_MODELS_DIR = "imported"
MODEL_FOLDER_MARKERS = {"config.json", "model.bin", "tokenizer.json", "vocabulary.json"}


def default_model_cache() -> Path:
    """Return the default model cache next to the app or current project."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "models"
    return Path.cwd() / "models"


def available_model_names() -> list[str]:
    """Return faster-whisper model names, falling back to a curated list."""
    try:
        from faster_whisper import available_models
    except ImportError:
        return DEFAULT_MODELS

    names = available_models()
    preferred = [name for name in DEFAULT_MODELS if name in names]
    remaining = [name for name in names if name not in preferred]
    return [*preferred, *remaining]


def describe_model(model_name: str) -> str:
    """Return short display text for a model."""
    if model_name.startswith("local:"):
        return "Imported local model in the configured model cache."
    return MODEL_DESCRIPTIONS.get(model_name, "CTranslate2 faster-whisper model.")


def imported_model_names(cache_dir: Path) -> list[str]:
    """Return imported local model labels from the configured cache."""

    imported_root = cache_dir / IMPORTED_MODELS_DIR
    if not imported_root.exists():
        return []
    return [f"local:{path.name}" for path in sorted(imported_root.iterdir()) if path.is_dir()]


def resolve_model_reference(model_name: str, cache_dir: Path | None) -> str:
    """Resolve a model selector value to a backend model name or path."""

    if not model_name.startswith("local:"):
        return model_name
    if cache_dir is None:
        raise ValueError("Imported local models require a model cache directory.")
    local_name = model_name.removeprefix("local:").strip()
    if not local_name or Path(local_name).name != local_name:
        raise ValueError(f"Invalid imported model name: {model_name}")
    model_path = cache_dir / IMPORTED_MODELS_DIR / local_name
    if not model_path.exists():
        raise FileNotFoundError(f"Imported model not found: {model_path}")
    return str(model_path)


def is_model_downloaded(model_name: str, cache_dir: Path) -> bool:
    """Return whether a model is available in the local cache."""
    try:
        from faster_whisper import download_model

        download_model(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except Exception:
        return False
    return True


def download_model_to_cache(model_name: str, cache_dir: Path) -> Path:
    """Download a faster-whisper model into the cache and return its local path."""
    from faster_whisper import download_model

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(download_model(model_name, cache_dir=str(cache_dir)))


def import_model_to_cache(source_file: Path, cache_dir: Path) -> Path:
    """Copy a selected local model file or model folder into the app cache."""

    if not source_file.exists():
        raise FileNotFoundError(f"Model file not found: {source_file}")
    source_root = _model_root_from_selection(source_file)
    imported_root = cache_dir / IMPORTED_MODELS_DIR
    imported_root.mkdir(parents=True, exist_ok=True)

    if source_root.resolve().is_relative_to(imported_root.resolve()):
        return source_root

    destination = _unique_destination(imported_root, source_root.stem if source_root.is_file() else source_root.name)
    if source_root.is_dir():
        shutil.copytree(source_root, destination)
    else:
        destination.mkdir(parents=True)
        shutil.copy2(source_root, destination / source_root.name)
    return destination


def _model_root_from_selection(source_file: Path) -> Path:
    if source_file.is_dir():
        return source_file
    parent_names = {path.name for path in source_file.parent.iterdir() if path.is_file()}
    if source_file.name in MODEL_FOLDER_MARKERS or MODEL_FOLDER_MARKERS.intersection(parent_names):
        return source_file.parent
    return source_file


def _unique_destination(parent: Path, name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in name).strip("._")
    safe_name = safe_name or "imported-model"
    destination = parent / safe_name
    counter = 2
    while destination.exists():
        destination = parent / f"{safe_name}-{counter}"
        counter += 1
    return destination


def cuda_device_count() -> int:
    """Return the number of CUDA devices CTranslate2 can see."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0

