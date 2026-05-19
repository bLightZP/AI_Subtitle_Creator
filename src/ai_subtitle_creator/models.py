"""Shared transcription models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ProgressCallback = Callable[[float | None, float | None], None]


class TaskName(str, Enum):
    """Whisper task modes."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


@dataclass(frozen=True)
class Segment:
    """A normalized transcription segment."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionInfo:
    """Metadata returned by a transcription backend."""

    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None


@dataclass(frozen=True)
class TranscriptionOptions:
    """User-selected transcription options."""

    model: str
    language: str | None
    task: TaskName
    device: str
    compute_type: str
    beam_size: int
    cpu_threads: int
    model_cache: Path | None
    vad_filter: bool
    progress_callback: ProgressCallback | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Transcription output returned by a backend."""

    segments: list[Segment]
    info: TranscriptionInfo

