# AI Subtitle Creator

Local AI subtitle creator for media files. The first CLI implementation accepts local media files such as `.mp4` and `.mkv`, transcribes them with a local Whisper-compatible backend, and writes an `.srt` subtitle file.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

`faster-whisper` decodes media through PyAV, which bundles FFmpeg libraries in its package. GPU use requires a compatible NVIDIA CUDA/cuDNN setup.

## Usage

Create subtitles next to the input file:

```powershell
aisub .\movie.mp4
```

The CLI shows a percent progress bar while the backend processes the media.

Launch the GUI queue manager from a development install:

```powershell
aisub-gui
```

The GUI lets you queue multiple media files, choose a model per file, download models into the configured cache, and track both current-file and full-queue progress.

Choose an output file and model:

```powershell
aisub .\movie.mkv --output .\movie.en.srt --model small --language en
```

Use CPU int8 inference:

```powershell
aisub .\movie.mp4 --device cpu --compute-type int8
```

Reduce CPU impact while transcribing:

```powershell
aisub .\movie.mp4 --process-priority below-normal --cpu-threads 2
```

The GUI exposes the same controls as `CPU priority` and `CPU threads`. `CPU priority` adjusts the current app process priority, which affects its native inference worker threads. `CPU threads` limits faster-whisper CPU worker threads; use `0` to let the backend choose.

Limit how long a single cue stays on screen:

```powershell
aisub .\movie.mp4 --max-cue-duration 6
```

Translate supported non-English audio to English:

```powershell
aisub .\movie.mp4 --task translate --model medium
```

## Current Backends

- `faster-whisper`: default backend, local inference, CPU/GPU capable.

The CLI is structured around a backend interface so additional engines, such as `whisper.cpp` or WhisperX, can be added without changing subtitle formatting or command-line behavior.

## Build a Standalone EXE

Build a one-file Windows executable:

```powershell
.\build_exe.bat
```

Build the GUI executable:

```powershell
.\build_gui_exe.bat
```

Build a faster-starting folder-mode GUI package:

```powershell
.\build_gui_folder.bat
```

The output is written to:

```text
dist\aisub.exe
dist\aisub-gui.exe
dist\aisub-gui-folder\aisub-gui.exe
```

The EXE bundles the Python app and native runtime libraries. Keep model files outside the EXE and point the CLI at them with `--model-cache`, for example:

```powershell
.\dist\aisub.exe .\movie.mp4 --model tiny --device cpu --compute-type int8 --model-cache .\models
```

CPU inference is the most portable build target. CUDA inference can still require NVIDIA CUDA/cuDNN runtime files on the target machine.

The one-file GUI is the simplest file to share, but it starts slower because PyInstaller extracts bundled native libraries at launch. The folder-mode GUI is larger as a folder, but usually starts faster because the files are already unpacked.

The folder-mode package stores support files under `dist\aisub-gui-folder\data\`. Keep that folder next to `aisub-gui.exe` when distributing it.

For GPU mode with current faster-whisper/CTranslate2 releases, install an up-to-date NVIDIA driver plus CUDA 12.x and cuDNN 9.x runtime libraries. The GUI includes buttons that open the official NVIDIA driver, CUDA Toolkit, and cuDNN download pages.

