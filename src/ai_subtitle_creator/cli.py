"""Command-line interface for AI Subtitle Creator."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from ai_subtitle_creator.backends import BackendName, create_backend
from ai_subtitle_creator.models import TaskName, TranscriptionOptions
from ai_subtitle_creator.process_priority import ProcessPriority, apply_process_priority
from ai_subtitle_creator.subtitles import SrtOptions, write_srt

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Create local AI-generated .srt subtitles from media files.",
)
console = Console()
PROGRESS_COLUMNS = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
)


@app.command()
def transcribe(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Local media file to transcribe, such as .mp4 or .mkv.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output .srt path. Defaults to input filename with .srt suffix."),
    ] = None,
    backend: Annotated[
        BackendName,
        typer.Option("--backend", help="Local transcription backend to use."),
    ] = BackendName.FASTER_WHISPER,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper model size or local model path."),
    ] = "small",
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Spoken language code, for example en, es, fr. Omit for auto-detect."),
    ] = None,
    task: Annotated[
        TaskName,
        typer.Option("--task", help="Transcribe source language or translate speech to English."),
    ] = TaskName.TRANSCRIBE,
    device: Annotated[
        str,
        typer.Option("--device", help="Inference device: auto, cpu, cuda, or another faster-whisper-supported device."),
    ] = "auto",
    compute_type: Annotated[
        str,
        typer.Option("--compute-type", help="Inference precision, such as default, int8, float16, or int8_float16."),
    ] = "default",
    beam_size: Annotated[
        int,
        typer.Option("--beam-size", min=1, help="Beam size used by the transcription backend."),
    ] = 5,
    cpu_threads: Annotated[
        int,
        typer.Option("--cpu-threads", min=0, help="CPU worker threads. 0 lets the backend choose."),
    ] = 0,
    process_priority: Annotated[
        ProcessPriority,
        typer.Option("--process-priority", help="Current process CPU priority."),
    ] = ProcessPriority.NORMAL,
    model_cache: Annotated[
        Path | None,
        typer.Option("--model-cache", help="Directory for downloaded or cached models."),
    ] = None,
    vad_filter: Annotated[
        bool,
        typer.Option("--vad-filter/--no-vad-filter", help="Use voice activity detection to reduce silence/hallucination."),
    ] = True,
    max_line_length: Annotated[
        int,
        typer.Option("--max-line-length", min=16, max=80, help="Maximum characters per subtitle line."),
    ] = 42,
    max_lines: Annotated[
        int,
        typer.Option("--max-lines", min=1, max=4, help="Maximum lines per subtitle cue."),
    ] = 2,
    max_cue_duration: Annotated[
        float,
        typer.Option("--max-cue-duration", min=1.0, help="Maximum display duration for one subtitle cue in seconds."),
    ] = 7.0,
) -> None:
    """Transcribe a media file and write an SRT subtitle file."""
    output_path = output or input_file.with_suffix(".srt")
    if output_path.suffix.lower() != ".srt":
        raise typer.BadParameter("Output path must end with .srt.", param_hint="--output")

    srt_options = SrtOptions(
        max_line_length=max_line_length,
        max_lines=max_lines,
        max_duration=max_cue_duration,
    )

    console.print(f"Transcribing [bold]{input_file}[/bold] with [bold]{backend.value}[/bold] ({model})")
    try:
        apply_process_priority(process_priority)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Could not set process priority: {exc}")
        raise typer.Exit(1) from exc
    transcription_backend = create_backend(backend)

    try:
        with Progress(*PROGRESS_COLUMNS, console=console) as progress:
            task_id = progress.add_task("Processing", total=None)

            def update_progress(completed: float | None, total: float | None) -> None:
                if total and total > 0:
                    progress.update(
                        task_id,
                        total=total,
                        completed=min(completed or 0.0, total),
                    )
                else:
                    progress.update(task_id, total=None, completed=0)

            options = TranscriptionOptions(
                model=model,
                language=language,
                task=task,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                cpu_threads=cpu_threads,
                model_cache=model_cache,
                vad_filter=vad_filter,
                progress_callback=update_progress,
            )
            result = transcription_backend.transcribe(input_file, options)
        cue_count = write_srt(result.segments, output_path, srt_options)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    language_label = result.info.language or "unknown"
    console.print(f"Wrote [bold]{cue_count}[/bold] cues to [bold]{output_path}[/bold]")
    console.print(f"Detected language: [bold]{language_label}[/bold]")


if __name__ == "__main__":
    app()

