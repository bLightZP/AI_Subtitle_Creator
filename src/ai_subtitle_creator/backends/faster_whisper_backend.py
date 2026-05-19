"""faster-whisper backend implementation."""

from __future__ import annotations

import re
from pathlib import Path

from ai_subtitle_creator.models import Segment, TranscriptionInfo, TranscriptionOptions, TranscriptionResult


class FasterWhisperBackend:
    """Transcription backend powered by SYSTRAN faster-whisper."""

    def transcribe(
        self,
        input_path: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "The faster-whisper backend is not installed. "
                'Install it with: python -m pip install -e ".[dev]"'
            ) from exc

        model = WhisperModel(
            options.model,
            device=options.device,
            compute_type=options.compute_type,
            cpu_threads=options.cpu_threads,
            download_root=str(options.model_cache) if options.model_cache else None,
        )

        segments_iter, info = model.transcribe(
            str(input_path),
            language=options.language,
            task=options.task.value,
            beam_size=options.beam_size,
            vad_filter=options.vad_filter,
        )
        duration = getattr(info, "duration", None)
        if options.progress_callback:
            options.progress_callback(0.0, duration)

        segments: list[Segment] = []
        for backend_segment in segments_iter:
            text = _normalize_text(backend_segment.text)
            start = max(0.0, float(backend_segment.start))
            end = max(0.0, float(backend_segment.end))
            if text:
                segments.append(
                    Segment(
                        start=start,
                        end=end,
                        text=text,
                    )
                )
            if options.progress_callback:
                options.progress_callback(end, duration)

        if options.progress_callback:
            options.progress_callback(duration, duration)

        return TranscriptionResult(
            segments=segments,
            info=TranscriptionInfo(
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
                duration=duration,
            )
        )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

