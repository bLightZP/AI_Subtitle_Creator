"""Model catalog and download helpers for faster-whisper."""

from __future__ import annotations

import sys
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
    return MODEL_DESCRIPTIONS.get(model_name, "CTranslate2 faster-whisper model.")


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


def cuda_device_count() -> int:
    """Return the number of CUDA devices CTranslate2 can see."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0

