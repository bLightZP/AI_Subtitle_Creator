"""faster-whisper backend implementation."""

from __future__ import annotations

import logging
import re
import gc
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ai_subtitle_creator.models import (
    ProgressCallback,
    Segment,
    TranscriptionInfo,
    TranscriptionOptions,
    TranscriptionResult,
)


class _NullTqdmFile:
    """Headless text sink for tqdm in windowed GUI builds."""

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


class FasterWhisperBackend:
    """Transcription backend powered by SYSTRAN faster-whisper."""

    def transcribe(
        self,
        input_path: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        _report_progress(options.progress_callback, 0.0)
        try:
            import av
            import faster_whisper.transcribe as faster_whisper_transcribe
            import faster_whisper.vad as faster_whisper_vad
            import numpy as np
            from faster_whisper import WhisperModel
            from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames
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
        _report_progress(options.progress_callback, 5.0)

        audio = _decode_audio_with_progress(
            av_module=av,
            np_module=np,
            input_path=input_path,
            sampling_rate=model.feature_extractor.sampling_rate,
            progress_callback=options.progress_callback,
            phase_start=5.0,
            phase_end=35.0,
            group_frames=_group_frames,
            ignore_invalid_frames=_ignore_invalid_frames,
            resample_frames=_resample_frames,
        )
        _report_progress(options.progress_callback, 35.0)

        segments: list[Segment] = []
        transcription_progress = _scale_progress_callback(options.progress_callback, 45.0, 99.0)
        with (
            _patch_vad_progress(faster_whisper_vad, options.progress_callback, 35.0, 45.0),
            _patch_faster_whisper_progress(faster_whisper_transcribe, transcription_progress),
        ):
            segments_iter, info = model.transcribe(
                audio,
                language=options.language,
                task=options.task.value,
                beam_size=options.beam_size,
                vad_filter=options.vad_filter,
                log_progress=transcription_progress is not None,
            )
            _report_progress(options.progress_callback, 45.0)
            duration = getattr(info, "duration", None)
            work_duration = getattr(info, "duration_after_vad", None) or duration
            if transcription_progress:
                transcription_progress(0.0, work_duration)

            with _capture_faster_whisper_chunk_progress(transcription_progress, work_duration):
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
                    if transcription_progress:
                        transcription_progress(end, duration)

        _report_progress(options.progress_callback, 100.0)

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


def _report_progress(callback: ProgressCallback | None, percent: float) -> None:
    if callback is None:
        return
    callback(max(0.0, min(100.0, percent)), 100.0)


def _scale_progress_callback(
    callback: ProgressCallback | None,
    phase_start: float,
    phase_end: float,
) -> ProgressCallback | None:
    if callback is None:
        return None

    span = phase_end - phase_start

    def report(completed: float | None, total: float | None) -> None:
        if total is None or total <= 0:
            return
        fraction = max(0.0, min(1.0, float(completed or 0.0) / float(total)))
        _report_progress(callback, phase_start + fraction * span)

    return report


def _decode_audio_with_progress(
    *,
    av_module,
    np_module,
    input_path: Path,
    sampling_rate: int,
    progress_callback: ProgressCallback | None,
    phase_start: float,
    phase_end: float,
    group_frames,
    ignore_invalid_frames,
    resample_frames,
):
    resampler = av_module.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    raw_buffer = io.BytesIO()
    dtype = None
    decoded_samples = 0
    duration = None

    with av_module.open(str(input_path), mode="r", metadata_errors="ignore") as container:
        duration = _media_duration_seconds(container)
        frames = container.decode(audio=0)
        frames = ignore_invalid_frames(frames)
        frames = group_frames(frames, 500000)
        frames = resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)
            decoded_samples += int(array.size)
            if duration and duration > 0:
                decoded_seconds = decoded_samples / sampling_rate
                progress = phase_start + min(decoded_seconds / duration, 1.0) * (phase_end - phase_start)
                _report_progress(progress_callback, progress)

    del resampler
    gc.collect()
    _report_progress(progress_callback, phase_end)

    if dtype is None:
        return np_module.array([], dtype=np_module.float32)

    audio = np_module.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    return audio.astype(np_module.float32) / 32768.0


def _media_duration_seconds(container) -> float | None:
    if getattr(container, "duration", None):
        return float(container.duration) / 1_000_000

    streams = getattr(container, "streams", None)
    if not streams:
        return None

    audio_streams = getattr(streams, "audio", None)
    if not audio_streams:
        return None

    stream = audio_streams[0]
    if getattr(stream, "duration", None) is None or getattr(stream, "time_base", None) is None:
        return None

    return float(stream.duration * stream.time_base)


@contextmanager
def _patch_faster_whisper_progress(module, callback: ProgressCallback | None) -> Iterator[None]:
    if callback is None:
        yield
        return

    original_tqdm = module.tqdm

    class TranscriptionProgressBar(original_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("file", _NullTqdmFile())
            self._last_report = (-1.0, -1.0)
            super().__init__(*args, **kwargs)
            self._notify()

        def update(self, n: int | float = 1):  # type: ignore[override]
            result = super().update(n)
            self._notify()
            return result

        def close(self) -> None:  # type: ignore[override]
            self._notify(force=True)
            super().close()

        def _notify(self, *, force: bool = False) -> None:
            total = self.total
            if total is None or total <= 0:
                return

            completed = min(max(float(self.n or 0.0), 0.0), float(total))
            snapshot = (completed, float(total))
            if force or snapshot != self._last_report:
                self._last_report = snapshot
                callback(completed, float(total))

    module.tqdm = TranscriptionProgressBar
    try:
        yield
    finally:
        module.tqdm = original_tqdm


@contextmanager
def _patch_vad_progress(
    vad_module,
    callback: ProgressCallback | None,
    phase_start: float,
    phase_end: float,
) -> Iterator[None]:
    if callback is None:
        yield
        return

    model_class = getattr(vad_module, "SileroVADModel", None) or getattr(vad_module, "SileroVadModel")
    original_call = model_class.__call__

    def call_with_progress(self, audio, num_samples: int = 512, context_size_samples: int = 64):
        np = vad_module.np
        assert audio.ndim == 1, "Input should be a 1D array"
        assert audio.shape[0] % num_samples == 0, "Input size should be a multiple of num_samples"

        h = np.zeros((1, 1, 128), dtype="float32")
        c = np.zeros((1, 1, 128), dtype="float32")
        context = np.zeros((1, context_size_samples), dtype="float32")

        batched_audio = audio.reshape(-1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[-1] = 0
        context = np.roll(context, 1, 0)
        batched_audio = np.concatenate([context, batched_audio], 1)
        batched_audio = batched_audio.reshape(-1, num_samples + context_size_samples)

        encoder_batch_size = 2000
        num_segments = batched_audio.shape[0]
        outputs = []
        for i in range(0, num_segments, encoder_batch_size):
            output, h, c = self.session.run(
                None,
                {"input": batched_audio[i : i + encoder_batch_size], "h": h, "c": c},
            )
            outputs.append(output)
            fraction = min(i + encoder_batch_size, num_segments) / max(num_segments, 1)
            _report_progress(callback, phase_start + fraction * (phase_end - phase_start))

        return np.concatenate(outputs, axis=0)

    model_class.__call__ = call_with_progress
    try:
        yield
    finally:
        model_class.__call__ = original_call


@contextmanager
def _capture_faster_whisper_chunk_progress(
    callback: ProgressCallback | None,
    total: float | None,
) -> Iterator[None]:
    if callback is None or total is None or total <= 0:
        yield
        return

    logger = logging.getLogger("faster_whisper")
    original_level = logger.level
    original_propagate = logger.propagate
    original_disabled = logger.disabled
    handler = _ChunkProgressHandler(callback, float(total))

    logger.disabled = False
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        logger.disabled = original_disabled


class _ChunkProgressHandler(logging.Handler):
    def __init__(self, callback: ProgressCallback, total: float) -> None:
        super().__init__(logging.DEBUG)
        self._callback = callback
        self._total = total
        self._last_completed = -1.0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        prefix = "Processing segment at "
        if not message.startswith(prefix):
            return

        completed = _parse_faster_whisper_timestamp(message.removeprefix(prefix))
        if completed is None:
            return

        completed = min(max(completed, 0.0), self._total)
        if completed < self._last_completed:
            return

        self._last_completed = completed
        self._callback(completed, self._total)


def _parse_faster_whisper_timestamp(value: str) -> float | None:
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        return None

    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[-3]) if len(parts) == 3 else 0
    except ValueError:
        return None

    return hours * 3600 + minutes * 60 + seconds

