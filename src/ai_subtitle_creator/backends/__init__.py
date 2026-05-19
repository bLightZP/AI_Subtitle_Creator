"""Transcription backend registry."""

from ai_subtitle_creator.backends.base import BackendName, TranscriptionBackend
from ai_subtitle_creator.backends.faster_whisper_backend import FasterWhisperBackend


def create_backend(name: BackendName) -> TranscriptionBackend:
    """Create a transcription backend by name."""
    if name is BackendName.FASTER_WHISPER:
        return FasterWhisperBackend()

    supported = ", ".join(backend.value for backend in BackendName)
    raise ValueError(f"Unsupported backend '{name}'. Supported backends: {supported}.")


__all__ = ["BackendName", "TranscriptionBackend", "create_backend"]

