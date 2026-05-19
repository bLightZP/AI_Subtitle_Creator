from pathlib import Path

from ai_subtitle_creator.models import Segment
from ai_subtitle_creator.subtitles import SrtOptions, format_timestamp, to_cues, write_srt


def test_format_timestamp_rounds_to_milliseconds() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(61.2345) == "00:01:01,234"
    assert format_timestamp(3661.9996) == "01:01:02,000"


def test_to_cues_splits_long_segments() -> None:
    segments = [
        Segment(
            start=0,
            end=10,
            text="This is a long subtitle sentence that should be split into multiple readable cues.",
        )
    ]

    cues = to_cues(segments, SrtOptions(max_line_length=20, max_lines=2))

    assert len(cues) > 1
    assert cues[0].start == 0
    assert cues[-1].end == 10
    assert all(cue.start <= cue.end for cue in cues)
    assert all(len(line) <= 20 for cue in cues for line in cue.text.splitlines())


def test_to_cues_does_not_truncate_wrapped_text() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    segments = [Segment(start=0, end=6, text=text)]

    cues = to_cues(segments, SrtOptions(max_line_length=18, max_lines=2))

    rendered = " ".join(cue.text.replace("\n", " ") for cue in cues)
    assert rendered == text


def test_to_cues_caps_long_display_duration() -> None:
    segments = [Segment(start=5, end=125, text="Short text")]

    cues = to_cues(segments, SrtOptions(max_duration=6))

    assert len(cues) == 1
    assert cues[0].start == 5
    assert cues[0].end == 11


def test_write_srt_uses_crlf(tmp_path: Path) -> None:
    output = tmp_path / "sample.srt"
    count = write_srt(
        [Segment(start=0, end=1.5, text="Hello world")],
        output,
        SrtOptions(),
    )

    assert count == 1
    data = output.read_bytes()
    assert b"1\r\n00:00:00,000 --> 00:00:01,500\r\nHello world\r\n" in data

