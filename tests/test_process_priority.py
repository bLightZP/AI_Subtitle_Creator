"""Tests for process priority helpers."""

from __future__ import annotations

import pytest

from ai_subtitle_creator.process_priority import ProcessPriority, priority_from_label, priority_labels


def test_priority_labels_round_trip() -> None:
    labels = priority_labels()

    assert labels == ("Low", "Below normal", "Normal", "Above normal", "High")
    assert priority_from_label("Below normal") == ProcessPriority.BELOW_NORMAL
    assert priority_from_label("High") == ProcessPriority.HIGH


def test_priority_from_label_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        priority_from_label("Realtime")
