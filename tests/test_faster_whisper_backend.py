import logging
from types import SimpleNamespace

import numpy as np

from ai_subtitle_creator.backends.faster_whisper_backend import (
    _capture_faster_whisper_chunk_progress,
    _parse_faster_whisper_timestamp,
    _patch_faster_whisper_progress,
    _patch_vad_progress,
    _scale_progress_callback,
)


class FakeTqdm:
    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.file = kwargs.get("file")
        self.n = 0

    def update(self, n=1):
        self.n += n
        return True

    def close(self) -> None:
        return None


def test_faster_whisper_progress_patch_reports_chunk_updates() -> None:
    updates = []
    module = SimpleNamespace(tqdm=FakeTqdm)

    with _patch_faster_whisper_progress(
        module,
        lambda completed, total: updates.append((completed, total)),
    ):
        progress_bar = module.tqdm(total=10, unit="seconds")
        progress_bar.update(2)
        progress_bar.update(3)
        progress_bar.close()

    assert module.tqdm is FakeTqdm
    assert updates[0] == (0.0, 10.0)
    assert (2.0, 10.0) in updates
    assert updates[-1] == (5.0, 10.0)
    assert progress_bar.file.write("hidden") == 0


def test_faster_whisper_chunk_log_progress_reports_segment_start() -> None:
    updates = []
    logger = logging.getLogger("faster_whisper")

    with _capture_faster_whisper_chunk_progress(
        lambda completed, total: updates.append((completed, total)),
        120.0,
    ):
        logger.debug("Processing segment at 01:02.500")

    assert updates == [(62.5, 120.0)]


def test_parse_faster_whisper_timestamp() -> None:
    assert _parse_faster_whisper_timestamp("02:03.250") == 123.25
    assert _parse_faster_whisper_timestamp("01:02:03.500") == 3723.5
    assert _parse_faster_whisper_timestamp("not a timestamp") is None


def test_scaled_progress_callback_reports_percent_range() -> None:
    updates = []
    callback = _scale_progress_callback(lambda completed, total: updates.append((completed, total)), 45.0, 99.0)

    assert callback is not None
    callback(50.0, 100.0)

    assert updates == [(72.0, 100.0)]


def test_vad_progress_patch_reports_batch_updates() -> None:
    updates = []

    class FakeSession:
        def run(self, _outputs, inputs):
            batch = inputs["input"]
            return np.zeros((len(batch), 1), dtype="float32"), inputs["h"], inputs["c"]

    class FakeSileroVADModel:
        def __init__(self) -> None:
            self.session = FakeSession()

        def __call__(self, _audio, num_samples: int = 512, context_size_samples: int = 64):
            return np.zeros((1, 1), dtype="float32")

    module = SimpleNamespace(np=np, SileroVADModel=FakeSileroVADModel)
    original_call = module.SileroVADModel.__call__

    with _patch_vad_progress(
        module,
        lambda completed, total: updates.append((completed, total)),
        35.0,
        45.0,
    ):
        result = module.SileroVADModel()(np.zeros(512 * 4001, dtype=np.float32))

    assert module.SileroVADModel.__call__ is original_call
    assert result.shape == (4001, 1)
    assert len(updates) == 3
    assert updates[-1] == (45.0, 100.0)
