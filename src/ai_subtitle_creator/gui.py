"""Tkinter GUI for AI Subtitle Creator."""

from __future__ import annotations

import queue
import threading
import traceback
import webbrowser
import sys
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, W, X, filedialog, messagebox, ttk
import tkinter as tk

from ai_subtitle_creator.backends import BackendName, create_backend
from ai_subtitle_creator.model_catalog import (
    available_model_names,
    cuda_device_count,
    default_model_cache,
    describe_model,
    download_model_to_cache,
    is_model_downloaded,
)
from ai_subtitle_creator.models import TaskName, TranscriptionOptions
from ai_subtitle_creator.subtitles import SrtOptions, write_srt

MEDIA_PATTERNS = (
    ("Media files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mp3 *.wav *.flac *.aac *.ogg"),
    ("MP4 files", "*.mp4"),
    ("MKV files", "*.mkv"),
    ("All files", "*.*"),
)
COLOR_BG = "#101216"
COLOR_PANEL = "#151922"
COLOR_PANEL_ALT = "#1c2230"
COLOR_CONTROL = "#171b24"
COLOR_CONTROL_ACTIVE = "#242b3a"
COLOR_BORDER = "#2b3342"
COLOR_BORDER_SOFT = "#202632"
COLOR_TEXT = "#e7eaf0"
COLOR_MUTED = "#a8b0bf"
COLOR_ACCENT = "#5d9cec"
COLOR_ACCENT_ACTIVE = "#77b2ff"
COLOR_ENTRY = "#0c0f14"
COLOR_SELECT = "#263d63"
CUDA_INSTALL_TEXT = """GPU mode uses faster-whisper through CTranslate2.

Current faster-whisper/CTranslate2 GPU builds require NVIDIA CUDA 12.x and cuDNN 9.x runtime libraries. Install an up-to-date NVIDIA driver first, then install CUDA Toolkit 12.x and cuDNN 9.x for Windows. Restart the machine after changing PATH or installing NVIDIA runtime components.

Use CPU + int8 for the most portable setup. Use GPU + float16 or int8_float16 only after the GPU check succeeds.

Automatic NVIDIA runtime installation is not built into this app. The official installers may require an NVIDIA account, administrator approval, GPU-specific driver choices, and license acceptance, so the GUI opens the official download pages instead."""


@dataclass
class QueueItem:
    """One queued transcription job."""

    item_id: str
    input_path: Path
    model: str
    status: str = "Queued"

    @property
    def output_path(self) -> Path:
        return self.input_path.with_suffix(".srt")


class SubtitleCreatorGui:
    """Desktop queue UI for local subtitle generation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Subtitle Creator")
        self.root.geometry("1120x760")
        self.root.minsize(980, 650)

        self.model_names = available_model_names()
        self.items: dict[str, QueueItem] = {}
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_requested = False
        self.active_item_id: str | None = None
        self.completed_count = 0
        self.current_fraction = 0.0

        self.default_model_var = tk.StringVar(value="small" if "small" in self.model_names else self.model_names[0])
        self.selected_model_var = tk.StringVar(value=self.default_model_var.get())
        self.device_var = tk.StringVar(value="cpu")
        self.compute_type_var = tk.StringVar(value="int8")
        self.language_var = tk.StringVar(value="")
        self.task_var = tk.StringVar(value=TaskName.TRANSCRIBE.value)
        self.model_cache_var = tk.StringVar(value=str(default_model_cache()))
        self.status_var = tk.StringVar(value="Ready")
        self.current_label_var = tk.StringVar(value="Current file: idle")
        self.queue_label_var = tk.StringVar(value="Queue: 0 files")
        self.download_status_var = tk.StringVar(value="Select a model to download.")
        self.task_help_var = tk.StringVar()

        self._configure_styles()
        self._build_layout()
        self._update_task_help()
        self._refresh_model_list()
        self.root.after(100, self._drain_events)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.root.configure(bg=COLOR_BG)
        self._enable_dark_title_bar()
        self.root.option_add("*Background", COLOR_BG)
        self.root.option_add("*Foreground", COLOR_TEXT)
        self.root.option_add("*activeBackground", COLOR_CONTROL_ACTIVE)
        self.root.option_add("*activeForeground", COLOR_TEXT)
        self.root.option_add("*insertBackground", COLOR_TEXT)
        self.root.option_add("*selectBackground", COLOR_SELECT)
        self.root.option_add("*selectForeground", COLOR_TEXT)
        self.root.option_add("*Entry.Background", COLOR_ENTRY)
        self.root.option_add("*Entry.Foreground", COLOR_TEXT)
        self.root.option_add("*Listbox.Background", COLOR_ENTRY)
        self.root.option_add("*Listbox.Foreground", COLOR_TEXT)
        self.root.option_add("*Listbox.selectBackground", COLOR_SELECT)
        self.root.option_add("*Listbox.selectForeground", COLOR_TEXT)
        self.root.option_add("*TCombobox*Listbox.background", COLOR_ENTRY)
        self.root.option_add("*TCombobox*Listbox.foreground", COLOR_TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", COLOR_SELECT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", COLOR_TEXT)

        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 9))
        style.configure("TFrame", background=COLOR_BG, borderwidth=0)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, padding=(0, 2))
        style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUTED)
        style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 14, "bold"))
        style.configure("Subheader.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10, "bold"))
        style.configure(
            "TLabelframe",
            background=COLOR_BG,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            darkcolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            relief="solid",
        )
        style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_TEXT)

        style.configure(
            "TButton",
            background=COLOR_CONTROL,
            foreground=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            darkcolor=COLOR_BORDER,
            focuscolor=COLOR_BORDER_SOFT,
            lightcolor=COLOR_BORDER,
            padding=(10, 6),
            relief="flat",
        )
        style.map(
            "TButton",
            background=[("active", COLOR_CONTROL_ACTIVE), ("pressed", COLOR_SELECT), ("disabled", COLOR_PANEL)],
            bordercolor=[("active", COLOR_ACCENT), ("pressed", COLOR_ACCENT), ("disabled", COLOR_BORDER_SOFT)],
            foreground=[("disabled", COLOR_MUTED)],
        )
        style.configure(
            "Accent.TButton",
            background=COLOR_ACCENT,
            foreground="#08111f",
            bordercolor=COLOR_ACCENT,
            darkcolor=COLOR_ACCENT,
            lightcolor=COLOR_ACCENT,
            focuscolor=COLOR_ACCENT,
            padding=(14, 7),
            relief="flat",
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLOR_ACCENT_ACTIVE), ("pressed", COLOR_ACCENT), ("disabled", COLOR_PANEL)],
            bordercolor=[("active", COLOR_ACCENT_ACTIVE), ("pressed", COLOR_ACCENT), ("disabled", COLOR_BORDER_SOFT)],
            foreground=[("disabled", COLOR_MUTED)],
        )

        style.configure(
            "TEntry",
            fieldbackground=COLOR_ENTRY,
            foreground=COLOR_TEXT,
            insertcolor=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            darkcolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            relief="flat",
            selectbackground=COLOR_SELECT,
            selectforeground=COLOR_TEXT,
        )
        style.map("TEntry", bordercolor=[("focus", COLOR_ACCENT), ("disabled", COLOR_BORDER_SOFT)])
        style.configure(
            "TCombobox",
            arrowcolor=COLOR_TEXT,
            arrowsize=12,
            background=COLOR_CONTROL,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            darkcolor=COLOR_BORDER,
            fieldbackground=COLOR_ENTRY,
            foreground=COLOR_TEXT,
            lightcolor=COLOR_BORDER,
            relief="flat",
            selectbackground=COLOR_ENTRY,
            selectforeground=COLOR_TEXT,
        )
        style.map(
            "TCombobox",
            background=[("active", COLOR_CONTROL_ACTIVE), ("pressed", COLOR_CONTROL_ACTIVE), ("disabled", COLOR_PANEL)],
            bordercolor=[("focus", COLOR_ACCENT), ("active", COLOR_ACCENT), ("disabled", COLOR_BORDER_SOFT)],
            fieldbackground=[("readonly", COLOR_ENTRY)],
            foreground=[("readonly", COLOR_TEXT)],
            selectbackground=[("readonly", COLOR_ENTRY)],
            selectforeground=[("readonly", COLOR_TEXT)],
        )
        style.configure(
            "Treeview",
            rowheight=28,
            background=COLOR_ENTRY,
            fieldbackground=COLOR_ENTRY,
            foreground=COLOR_TEXT,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            darkcolor=COLOR_BORDER,
            lightcolor=COLOR_BORDER,
            relief="flat",
        )
        style.configure(
            "Treeview.Heading",
            background=COLOR_PANEL_ALT,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            darkcolor=COLOR_BORDER,
            foreground=COLOR_TEXT,
            lightcolor=COLOR_BORDER,
            relief="flat",
        )
        style.map("Treeview.Heading", background=[("active", COLOR_CONTROL_ACTIVE)])
        style.map("Treeview", background=[("selected", COLOR_SELECT)], foreground=[("selected", COLOR_TEXT)])
        style.configure(
            "TProgressbar",
            background=COLOR_ACCENT,
            bordercolor=COLOR_BORDER_SOFT,
            darkcolor=COLOR_ACCENT,
            lightcolor=COLOR_ACCENT,
            troughcolor=COLOR_ENTRY,
            thickness=14,
        )
        style.configure("TPanedwindow", background=COLOR_BG)
        style.configure(
            "Vertical.TScrollbar",
            arrowcolor=COLOR_MUTED,
            background=COLOR_CONTROL,
            bordercolor=COLOR_BG,
            borderwidth=0,
            darkcolor=COLOR_CONTROL,
            lightcolor=COLOR_CONTROL,
            relief="flat",
            troughcolor=COLOR_ENTRY,
        )
        style.map(
            "Vertical.TScrollbar",
            arrowcolor=[("active", COLOR_TEXT)],
            background=[("active", COLOR_CONTROL_ACTIVE), ("pressed", COLOR_SELECT)],
        )

    def _enable_dark_title_bar(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            value = ctypes.c_int(1)
            for attribute in (20, 19):
                result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attribute,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
                if result == 0:
                    break
        except Exception:
            return

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=16)
        root_frame.pack(fill=BOTH, expand=True)

        header = ttk.Frame(root_frame)
        header.pack(fill=X)
        ttk.Label(header, text="AI Subtitle Creator", style="Header.TLabel").pack(side=LEFT)
        ttk.Label(header, textvariable=self.status_var).pack(side=RIGHT)

        body = ttk.PanedWindow(root_frame, orient=tk.HORIZONTAL)
        body.pack(fill=BOTH, expand=True, pady=(14, 0))

        left = ttk.Frame(body, padding=(0, 0, 12, 0))
        right = ttk.Frame(body, padding=(12, 0, 0, 0))
        body.add(left, weight=3)
        body.add(right, weight=2)

        self._build_queue_panel(left)
        self._build_settings_panel(right)
        self._build_models_panel(right)
        self._build_gpu_panel(right)

    def _build_queue_panel(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill=X)
        ttk.Button(controls, text="Add media files", command=self._add_files).pack(side=LEFT)
        ttk.Button(controls, text="Remove selected", command=self._remove_selected).pack(side=LEFT, padx=(8, 0))
        ttk.Button(controls, text="Clear queue", command=self._clear_queue).pack(side=LEFT, padx=(8, 0))

        columns = ("file", "model", "status", "output")
        table = ttk.Frame(parent)
        table.pack(fill=BOTH, expand=True, pady=(10, 8))

        self.queue_tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="extended")
        self.queue_tree.heading("file", text="Media file")
        self.queue_tree.heading("model", text="Model")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.heading("output", text="SRT output")
        self.queue_tree.column("file", width=330, anchor=W)
        self.queue_tree.column("model", width=100, anchor=W)
        self.queue_tree.column("status", width=110, anchor=W)
        self.queue_tree.column("output", width=300, anchor=W)
        self.queue_tree.pack(fill=BOTH, expand=True, side=LEFT)
        self.queue_tree.bind("<<TreeviewSelect>>", self._on_queue_selection)

        scrollbar = ttk.Scrollbar(table, orient=VERTICAL, command=self.queue_tree.yview)
        scrollbar.pack(fill=tk.Y, side=RIGHT)
        self.queue_tree.configure(yscrollcommand=scrollbar.set)

        model_row = ttk.Frame(parent)
        model_row.pack(fill=X, pady=(0, 8))
        ttk.Label(model_row, text="Model for selected:").pack(side=LEFT)
        self.selected_model_combo = ttk.Combobox(
            model_row,
            textvariable=self.selected_model_var,
            values=self.model_names,
            state="readonly",
            width=22,
        )
        self.selected_model_combo.pack(side=LEFT, padx=(8, 8))
        ttk.Button(model_row, text="Apply", command=self._apply_model_to_selected).pack(side=LEFT)

        progress = ttk.Frame(parent)
        progress.pack(fill=X)
        ttk.Label(progress, textvariable=self.current_label_var).pack(anchor=W)
        self.current_progress = ttk.Progressbar(progress, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.current_progress.pack(fill=X, pady=(2, 8))
        ttk.Label(progress, textvariable=self.queue_label_var).pack(anchor=W)
        self.queue_progress = ttk.Progressbar(progress, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.queue_progress.pack(fill=X, pady=(2, 10))

        run_row = ttk.Frame(parent)
        run_row.pack(fill=X)
        self.start_button = ttk.Button(run_row, text="Start queue", style="Accent.TButton", command=self._start_queue)
        self.start_button.pack(side=LEFT)
        self.stop_button = ttk.Button(run_row, text="Stop after current", command=self._request_stop, state=tk.DISABLED)
        self.stop_button.pack(side=LEFT, padx=(8, 0))

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Transcription Settings", padding=12)
        frame.pack(fill=X)

        ttk.Label(frame, text="Default model").grid(row=0, column=0, sticky=W)
        ttk.Combobox(frame, textvariable=self.default_model_var, values=self.model_names, state="readonly", width=22).grid(
            row=0,
            column=1,
            sticky=W,
            padx=(8, 0),
        )

        ttk.Label(frame, text="Device").grid(row=1, column=0, sticky=W, pady=(8, 0))
        device_box = ttk.Combobox(
            frame,
            textvariable=self.device_var,
            values=("cpu", "cuda", "auto"),
            state="readonly",
            width=22,
        )
        device_box.grid(row=1, column=1, sticky=W, padx=(8, 0), pady=(8, 0))
        device_box.bind("<<ComboboxSelected>>", self._on_device_changed)

        ttk.Label(frame, text="Compute type").grid(row=2, column=0, sticky=W, pady=(8, 0))
        ttk.Combobox(
            frame,
            textvariable=self.compute_type_var,
            values=("int8", "float16", "int8_float16", "float32", "default"),
            state="readonly",
            width=22,
        ).grid(row=2, column=1, sticky=W, padx=(8, 0), pady=(8, 0))

        ttk.Label(frame, text="Task").grid(row=3, column=0, sticky=W, pady=(8, 0))
        task_box = ttk.Combobox(
            frame,
            textvariable=self.task_var,
            values=(TaskName.TRANSCRIBE.value, TaskName.TRANSLATE.value),
            state="readonly",
            width=22,
        )
        task_box.grid(row=3, column=1, sticky=W, padx=(8, 0), pady=(8, 0))
        task_box.bind("<<ComboboxSelected>>", self._on_task_changed)

        ttk.Label(frame, text="Language (source)").grid(row=4, column=0, sticky=W, pady=(8, 0))
        ttk.Entry(frame, textvariable=self.language_var, width=25).grid(row=4, column=1, sticky=W, padx=(8, 0), pady=(8, 0))

        ttk.Label(frame, textvariable=self.task_help_var, wraplength=410).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky=W,
            pady=(8, 0),
        )

        ttk.Label(frame, text="Model cache").grid(row=6, column=0, sticky=W, pady=(8, 0))
        ttk.Entry(frame, textvariable=self.model_cache_var, width=34).grid(row=6, column=1, sticky=W, padx=(8, 0), pady=(8, 0))
        ttk.Button(frame, text="Browse", command=self._browse_model_cache).grid(row=6, column=2, sticky=W, padx=(8, 0), pady=(8, 0))

    def _build_models_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Model Downloads", padding=12)
        frame.pack(fill=BOTH, expand=True, pady=(12, 0))

        model_area = ttk.Frame(frame)
        model_area.pack(fill=BOTH, expand=True)
        self.model_list = tk.Listbox(model_area, height=8, exportselection=False)
        self.model_list.configure(
            background=COLOR_ENTRY,
            borderwidth=1,
            foreground=COLOR_TEXT,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_ACCENT,
            relief=tk.FLAT,
            selectbackground=COLOR_SELECT,
            selectforeground=COLOR_TEXT,
        )
        self.model_list.pack(fill=BOTH, expand=True, side=LEFT)
        self.model_list.bind("<<ListboxSelect>>", self._on_model_list_selected)
        model_scroll = ttk.Scrollbar(model_area, orient=VERTICAL, command=self.model_list.yview)
        model_scroll.pack(side=RIGHT, fill=tk.Y)
        self.model_list.configure(yscrollcommand=model_scroll.set)

        ttk.Label(frame, textvariable=self.download_status_var, wraplength=380).pack(fill=X, pady=(8, 6))
        buttons = ttk.Frame(frame)
        buttons.pack(fill=X)
        self.download_button = ttk.Button(buttons, text="Download selected model", command=self._download_selected_model)
        self.download_button.pack(side=LEFT)
        ttk.Button(buttons, text="Refresh", command=self._refresh_model_list).pack(side=LEFT, padx=(8, 0))

    def _build_gpu_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="GPU Setup", padding=12)
        frame.pack(fill=BOTH, expand=True, pady=(12, 0))

        instructions_area = ttk.Frame(frame)
        instructions_area.pack(fill=BOTH, expand=True)

        instructions = tk.Text(instructions_area, height=9, wrap=tk.WORD)
        instructions.insert("1.0", CUDA_INSTALL_TEXT)
        instructions.configure(
            state=tk.DISABLED,
            background=COLOR_ENTRY,
            borderwidth=1,
            foreground=COLOR_TEXT,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_ACCENT,
            insertbackground=COLOR_TEXT,
            relief=tk.FLAT,
            selectbackground=COLOR_SELECT,
            selectforeground=COLOR_TEXT,
            yscrollcommand=lambda *args: instructions_scroll.set(*args),
        )
        instructions_scroll = ttk.Scrollbar(instructions_area, orient=VERTICAL, command=instructions.yview)
        instructions_scroll.pack(fill=tk.Y, side=RIGHT)
        instructions.pack(fill=BOTH, expand=True, side=LEFT)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=X, pady=(8, 0))
        ttk.Button(buttons, text="Check GPU runtime", command=self._check_gpu).pack(side=LEFT)
        ttk.Button(buttons, text="NVIDIA driver", command=lambda: self._open_url("https://www.nvidia.com/Download/index.aspx")).pack(
            side=LEFT,
            padx=(8, 0),
        )
        ttk.Button(buttons, text="CUDA Toolkit", command=lambda: self._open_url("https://developer.nvidia.com/cuda-downloads")).pack(
            side=LEFT,
            padx=(8, 0),
        )
        ttk.Button(buttons, text="cuDNN", command=lambda: self._open_url("https://developer.nvidia.com/cudnn")).pack(
            side=LEFT,
            padx=(8, 0),
        )

    def _add_files(self) -> None:
        filenames = filedialog.askopenfilenames(title="Select media files", filetypes=MEDIA_PATTERNS)
        for filename in filenames:
            path = Path(filename)
            if any(item.input_path == path for item in self.items.values()):
                continue
            item_id = self.queue_tree.insert(
                "",
                END,
                values=(str(path), self.default_model_var.get(), "Queued", str(path.with_suffix(".srt"))),
            )
            self.items[item_id] = QueueItem(item_id=item_id, input_path=path, model=self.default_model_var.get())
        self._update_queue_progress()

    def _remove_selected(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Queue running", "Stop the queue before removing files.")
            return
        for item_id in self.queue_tree.selection():
            self.queue_tree.delete(item_id)
            self.items.pop(item_id, None)
        self._update_queue_progress()

    def _clear_queue(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Queue running", "Stop the queue before clearing files.")
            return
        for item_id in list(self.items):
            self.queue_tree.delete(item_id)
        self.items.clear()
        self.completed_count = 0
        self.current_fraction = 0
        self.current_progress["value"] = 0
        self._update_queue_progress()

    def _on_queue_selection(self, _event: object | None = None) -> None:
        selection = self.queue_tree.selection()
        if len(selection) == 1 and selection[0] in self.items:
            self.selected_model_var.set(self.items[selection[0]].model)

    def _apply_model_to_selected(self) -> None:
        model = self.selected_model_var.get()
        for item_id in self.queue_tree.selection():
            item = self.items.get(item_id)
            if not item:
                continue
            item.model = model
            self.queue_tree.set(item_id, "model", model)

    def _browse_model_cache(self) -> None:
        directory = filedialog.askdirectory(title="Select model cache directory", initialdir=self.model_cache_var.get())
        if directory:
            self.model_cache_var.set(directory)
            self._refresh_model_list()

    def _on_device_changed(self, _event: object | None = None) -> None:
        if self.device_var.get() == "cuda" and self.compute_type_var.get() == "int8":
            self.compute_type_var.set("float16")
        elif self.device_var.get() == "cpu" and self.compute_type_var.get() == "float16":
            self.compute_type_var.set("int8")

    def _on_task_changed(self, _event: object | None = None) -> None:
        self._update_task_help()

    def _update_task_help(self) -> None:
        if self.task_var.get() == TaskName.TRANSLATE.value:
            self.task_help_var.set(
                "Translate converts the detected source speech into English subtitles. "
                "Language is the source audio language hint, such as es or fr; leave it blank to auto-detect. "
                "It is not the target language."
            )
        else:
            self.task_help_var.set(
                "Transcribe writes subtitles in the spoken language. "
                "Language is an optional source audio hint, such as en or ja; leave it blank to auto-detect."
            )

    def _start_queue(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        queued = [item for item in self.items.values() if item.status in {"Queued", "Failed", "Done"}]
        if not queued:
            messagebox.showinfo("Empty queue", "Add at least one media file.")
            return
        self.stop_requested = False
        self.completed_count = 0
        self.current_fraction = 0
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Running queue")
        for item in self.items.values():
            item.status = "Queued"
            self.queue_tree.set(item.item_id, "status", item.status)
        self.worker = threading.Thread(target=self._run_queue_worker, daemon=True)
        self.worker.start()

    def _request_stop(self) -> None:
        self.stop_requested = True
        self.status_var.set("Stop requested; current file will finish")

    def _run_queue_worker(self) -> None:
        items = list(self.items.values())
        backend = create_backend(BackendName.FASTER_WHISPER)
        cache_dir = Path(self.model_cache_var.get())

        for index, item in enumerate(items):
            if self.stop_requested:
                break
            self.events.put(("start_item", (item.item_id, index, len(items))))
            try:
                result = backend.transcribe(
                    item.input_path,
                    TranscriptionOptions(
                        model=item.model,
                        language=self.language_var.get().strip() or None,
                        task=TaskName(self.task_var.get()),
                        device=self.device_var.get(),
                        compute_type=self.compute_type_var.get(),
                        beam_size=5,
                        cpu_threads=0,
                        model_cache=cache_dir,
                        vad_filter=True,
                        progress_callback=lambda completed, total, item_id=item.item_id: self.events.put(
                            ("item_progress", (item_id, completed, total))
                        ),
                    ),
                )
                cue_count = write_srt(
                    result.segments,
                    item.output_path,
                    SrtOptions(max_line_length=42, max_lines=2, max_duration=7.0),
                )
                self.events.put(("item_done", (item.item_id, cue_count, result.info.language)))
            except Exception as exc:
                self.events.put(("item_failed", (item.item_id, str(exc), traceback.format_exc())))

        self.events.put(("queue_done", None))

    def _drain_events(self) -> None:
        try:
            while True:
                event_name, payload = self.events.get_nowait()
                self._handle_event(event_name, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _handle_event(self, event_name: str, payload: object) -> None:
        if event_name == "start_item":
            item_id, index, total = payload  # type: ignore[misc]
            self.active_item_id = item_id
            self.completed_count = index
            self.current_fraction = 0
            item = self.items[item_id]
            item.status = "Running"
            self.queue_tree.set(item_id, "status", "Running")
            self.current_label_var.set(f"Current file: {item.input_path.name}")
            self.queue_label_var.set(f"Queue: {index + 1} of {total}")
            self.current_progress["value"] = 0
            self._update_queue_progress()
        elif event_name == "item_progress":
            item_id, completed, total = payload  # type: ignore[misc]
            if item_id != self.active_item_id:
                return
            if total and total > 0:
                self.current_fraction = max(0.0, min(1.0, float(completed or 0.0) / float(total)))
                self.current_progress["value"] = self.current_fraction * 100
                self._update_queue_progress()
        elif event_name == "item_done":
            item_id, cue_count, language = payload  # type: ignore[misc]
            item = self.items[item_id]
            item.status = f"Done ({cue_count} cues)"
            self.queue_tree.set(item_id, "status", item.status)
            self.current_progress["value"] = 100
            self.current_label_var.set(f"Current file: finished {item.input_path.name} ({language or 'unknown'})")
        elif event_name == "item_failed":
            item_id, message, details = payload  # type: ignore[misc]
            item = self.items[item_id]
            item.status = "Failed"
            self.queue_tree.set(item_id, "status", "Failed")
            self.status_var.set(f"Failed: {item.input_path.name}")
            messagebox.showerror("Transcription failed", f"{message}\n\n{details}")
        elif event_name == "queue_done":
            self.active_item_id = None
            self.current_fraction = 0
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.status_var.set("Queue finished" if not self.stop_requested else "Queue stopped")
            self._update_queue_progress(final=True)
        elif event_name == "download_done":
            model_name, path = payload  # type: ignore[misc]
            self.download_button.configure(state=tk.NORMAL)
            self.download_status_var.set(f"Downloaded {model_name} to {path}")
            self._refresh_model_list()
        elif event_name == "download_failed":
            model_name, message, details = payload  # type: ignore[misc]
            self.download_button.configure(state=tk.NORMAL)
            self.download_status_var.set(f"Failed to download {model_name}: {message}")
            messagebox.showerror("Download failed", f"{message}\n\n{details}")

    def _update_queue_progress(self, final: bool = False) -> None:
        total = len(self.items)
        if not total:
            self.queue_progress["value"] = 0
            self.queue_label_var.set("Queue: 0 files")
            return
        if final:
            done = len([item for item in self.items.values() if item.status.startswith("Done")])
            self.queue_progress["value"] = (done / total) * 100
            self.queue_label_var.set(f"Queue: {done} of {total} finished")
            return
        self.queue_progress["value"] = ((self.completed_count + self.current_fraction) / total) * 100

    def _refresh_model_list(self) -> None:
        selected = self._selected_download_model()
        self.model_list.delete(0, END)
        cache_dir = Path(self.model_cache_var.get())
        for model_name in self.model_names:
            status = "cached" if is_model_downloaded(model_name, cache_dir) else "not downloaded"
            self.model_list.insert(END, f"{model_name} [{status}]")
        if selected and selected in self.model_names:
            index = self.model_names.index(selected)
            self.model_list.selection_set(index)
            self.model_list.see(index)
        self._on_model_list_selected()

    def _selected_download_model(self) -> str | None:
        selection = self.model_list.curselection() if hasattr(self, "model_list") else ()
        if not selection:
            return None
        return self.model_names[selection[0]]

    def _on_model_list_selected(self, _event: object | None = None) -> None:
        model_name = self._selected_download_model()
        if not model_name:
            self.download_status_var.set("Select a model to download.")
            return
        cached = is_model_downloaded(model_name, Path(self.model_cache_var.get()))
        status = "Downloaded" if cached else "Not downloaded"
        self.download_status_var.set(f"{model_name}: {status}. {describe_model(model_name)}")

    def _download_selected_model(self) -> None:
        model_name = self._selected_download_model()
        if not model_name:
            messagebox.showinfo("No model selected", "Select a model from the list first.")
            return
        self.download_button.configure(state=tk.DISABLED)
        self.download_status_var.set(f"Downloading {model_name}...")
        thread = threading.Thread(target=self._download_model_worker, args=(model_name,), daemon=True)
        thread.start()

    def _download_model_worker(self, model_name: str) -> None:
        try:
            path = download_model_to_cache(model_name, Path(self.model_cache_var.get()))
            self.events.put(("download_done", (model_name, str(path))))
        except Exception as exc:
            self.events.put(("download_failed", (model_name, str(exc), traceback.format_exc())))

    def _check_gpu(self) -> None:
        count = cuda_device_count()
        if count > 0:
            messagebox.showinfo("GPU runtime", f"CTranslate2 can see {count} CUDA device(s).")
        else:
            messagebox.showwarning(
                "GPU runtime",
                "CTranslate2 cannot see a CUDA device. Use CPU mode or install/update NVIDIA driver, CUDA 12.x, and cuDNN 9.x.",
            )

    def _open_url(self, url: str) -> None:
        webbrowser.open(url)


def app() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    SubtitleCreatorGui(root)
    root.mainloop()


if __name__ == "__main__":
    app()
