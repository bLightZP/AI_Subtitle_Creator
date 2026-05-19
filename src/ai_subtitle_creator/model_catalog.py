"""Model catalog and download helpers for faster-whisper."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Sequence
from fnmatch import fnmatch
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
MODEL_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

DownloadProgressCallback = Callable[[int, int], None]


class _NullTqdmFile:
    """Headless text sink for tqdm in windowed GUI builds."""

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


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


def local_model_names(cache_dir: Path, downloadable_names: Sequence[str] | None = None) -> list[str]:
    """Return locally usable model selectors for the configured cache."""

    names = list(downloadable_names or available_model_names())
    downloaded = [name for name in names if is_model_downloaded(name, cache_dir)]
    return [*downloaded, *imported_model_names(cache_dir)]


def download_model_size_bytes(model_name: str) -> int | None:
    """Return the remote download size for a faster-whisper model when known."""

    from faster_whisper.utils import _MODELS
    from huggingface_hub import model_info

    repo_id = model_name if "/" in model_name else _MODELS.get(model_name)
    if repo_id is None:
        return None

    info = model_info(repo_id, files_metadata=True, timeout=10)
    total_size = 0
    for sibling in info.siblings or []:
        filename = getattr(sibling, "rfilename", "")
        size = getattr(sibling, "size", None)
        if size is None or not _is_model_download_file(filename):
            continue
        total_size += int(size)
    return total_size or None


def format_model_size(size_bytes: int | None) -> str:
    """Format a model size using mb below 1 gb and gb above it."""

    if size_bytes is None or size_bytes <= 0:
        return ""
    if size_bytes < 1_000_000_000:
        return f"{size_bytes / 1_000_000:.0f} mb"
    return f"{size_bytes / 1_000_000_000:.1f} gb"


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


def download_model_to_cache(
    model_name: str,
    cache_dir: Path,
    progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    """Download a faster-whisper model into the cache and return its local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if progress_callback is None:
        from faster_whisper import download_model

        return Path(download_model(model_name, cache_dir=str(cache_dir)))

    from faster_whisper.utils import _MODELS
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    if "/" in model_name:
        repo_id = model_name
    else:
        repo_id = _MODELS.get(model_name)
        if repo_id is None:
            raise ValueError(f"Invalid model size '{model_name}'")

    class DownloadProgressBar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("file", _NullTqdmFile())
            self._report_progress = kwargs.get("unit") == "B"
            self._last_report = (-1, -1)
            super().__init__(*args, **kwargs)
            self._notify()

        def update(self, n: int | float = 1):  # type: ignore[override]
            result = super().update(n)
            self._notify()
            return result

        def refresh(self, *args, **kwargs):  # type: ignore[override]
            result = super().refresh(*args, **kwargs)
            self._notify()
            return result

        def close(self) -> None:  # type: ignore[override]
            self._notify(force=True)
            super().close()

        def _notify(self, *, force: bool = False) -> None:
            if not self._report_progress:
                return
            completed = int(self.n or 0)
            total = int(self.total or 0)
            snapshot = (completed, total)
            if force or snapshot != self._last_report:
                self._last_report = snapshot
                progress_callback(completed, total)

    return Path(
        snapshot_download(
            repo_id,
            cache_dir=str(cache_dir),
            allow_patterns=MODEL_ALLOW_PATTERNS,
            tqdm_class=DownloadProgressBar,
        )
    )


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


def _is_model_download_file(filename: str) -> bool:
    return any(fnmatch(filename, pattern) for pattern in MODEL_ALLOW_PATTERNS)


def cuda_device_count() -> int:
    """Return the number of CUDA devices CTranslate2 can see."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0

