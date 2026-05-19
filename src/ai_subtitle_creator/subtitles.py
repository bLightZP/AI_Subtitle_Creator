"""SRT subtitle formatting."""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from ai_subtitle_creator.models import Segment


@dataclass(frozen=True)
class SrtOptions:
    """Options for SRT cue formatting."""

    max_line_length: int = 42
    max_lines: int = 2
    max_duration: float = 7.0


@dataclass(frozen=True)
class Cue:
    """A single subtitle cue."""

    start: float
    end: float
    text: str


def write_srt(segments: list[Segment], output_path: Path, options: SrtOptions) -> int:
    """Write segments as SRT and return the number of cues written."""
    cues = list(to_cues(segments, options))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = [
        f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}"
        for index, cue in enumerate(cues, start=1)
    ]
    content = "\n\n".join(blocks)
    if content:
        content += "\n"

    output_path.write_text(content, encoding="utf-8", newline="\r\n")
    return len(cues)


def to_cues(segments: list[Segment], options: SrtOptions) -> list[Cue]:
    """Convert transcription segments into display-friendly subtitle cues."""
    cues: list[Cue] = []

    for segment in segments:
        text = _normalize_text(segment.text)
        if not text:
            continue

        chunks = _chunk_text(text, options)
        if len(chunks) == 1:
            cues.append(Cue(segment.start, segment.end, _wrap_text(chunks[0], options)))
            continue

        duration = max(0.0, segment.end - segment.start)
        weights = [max(1, len(chunk)) for chunk in chunks]
        total_weight = sum(weights)
        cursor = segment.start

        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                chunk_end = segment.end
            else:
                chunk_duration = duration * (weights[index] / total_weight)
                chunk_end = min(segment.end, cursor + chunk_duration)

            cues.append(Cue(cursor, chunk_end, _wrap_text(chunk, options)))
            cursor = chunk_end

    return _fix_cue_timings(cues, options)


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _chunk_text(text: str, options: SrtOptions) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not _fits_in_cue(sentence, options):
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_words(sentence, options))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and not _fits_in_cue(candidate, options):
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks or [text]


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _split_words(text: str, options: SrtOptions) -> list[str]:
    chunks: list[str] = []
    current_words: list[str] = []

    for word in text.split():
        candidate = " ".join([*current_words, word])
        if current_words and not _fits_in_cue(candidate, options):
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _wrap_text(text: str, options: SrtOptions) -> str:
    return "\n".join(_wrapped_lines(text, options.max_line_length)) or text


def _fits_in_cue(text: str, options: SrtOptions) -> bool:
    return len(_wrapped_lines(text, options.max_line_length)) <= options.max_lines


def _wrapped_lines(text: str, max_line_length: int) -> list[str]:
    wrapped = textwrap.wrap(
        text,
        width=max_line_length,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return wrapped


def _fix_cue_timings(cues: list[Cue], options: SrtOptions) -> list[Cue]:
    fixed: list[Cue] = []
    previous_end = 0.0

    for cue in cues:
        start = max(cue.start, previous_end)
        end = max(cue.end, start)
        if end - start > options.max_duration:
            end = start + options.max_duration
        if math.isclose(start, end):
            end = start + 0.001
        fixed.append(Cue(start, end, cue.text))
        previous_end = end

    return fixed


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

