"""Process priority helpers."""

from __future__ import annotations

import ctypes
import os
import sys
from enum import Enum


class ProcessPriority(str, Enum):
    """Supported process priority levels."""

    LOW = "low"
    BELOW_NORMAL = "below-normal"
    NORMAL = "normal"
    ABOVE_NORMAL = "above-normal"
    HIGH = "high"


PRIORITY_LABELS: dict[ProcessPriority, str] = {
    ProcessPriority.LOW: "Low",
    ProcessPriority.BELOW_NORMAL: "Below normal",
    ProcessPriority.NORMAL: "Normal",
    ProcessPriority.ABOVE_NORMAL: "Above normal",
    ProcessPriority.HIGH: "High",
}
PRIORITY_BY_LABEL = {label: priority for priority, label in PRIORITY_LABELS.items()}


def priority_labels() -> tuple[str, ...]:
    return tuple(PRIORITY_LABELS.values())


def priority_from_label(label: str) -> ProcessPriority:
    try:
        return PRIORITY_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"Unknown priority label: {label}") from exc


def apply_process_priority(priority: ProcessPriority) -> None:
    """Apply a process priority class to the current process."""

    if sys.platform == "win32":
        _apply_windows_priority(priority)
        return
    _apply_posix_priority(priority)


def _apply_windows_priority(priority: ProcessPriority) -> None:
    priority_classes = {
        ProcessPriority.LOW: 0x00000040,
        ProcessPriority.BELOW_NORMAL: 0x00004000,
        ProcessPriority.NORMAL: 0x00000020,
        ProcessPriority.ABOVE_NORMAL: 0x00008000,
        ProcessPriority.HIGH: 0x00000080,
    }
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetCurrentProcess()
    if not kernel32.SetPriorityClass(handle, priority_classes[priority]):
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "SetPriorityClass failed")


def _apply_posix_priority(priority: ProcessPriority) -> None:
    if priority == ProcessPriority.NORMAL:
        return
    if priority in {ProcessPriority.LOW, ProcessPriority.BELOW_NORMAL}:
        os.nice(10 if priority == ProcessPriority.LOW else 5)
        return
    raise RuntimeError("Raising process priority is only supported on Windows.")
