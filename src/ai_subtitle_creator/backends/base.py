"""Shared backend types."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol

from ai_subtitle_creator.models import TranscriptionOptions, TranscriptionResult


class BackendName(str, Enum):
    """Supported transcription backend names."""

    FASTER_WHISPER = "faster-whisper"


class TranscriptionBackend(Protocol):
    """Backend contract for local speech-to-text engines."""

    def transcribe(
        self,
        input_path: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        """Transcribe a media file into normalized subtitle segments."""

