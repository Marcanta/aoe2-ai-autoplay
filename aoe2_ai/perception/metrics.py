"""Per-stage timing for the perception pipeline.

Stages (seconds, summed over every call in one observation):

* ``capture``  - grabbing or loading the frame
* ``hud``      - HUD-presence template match
* ``regions``  - scaling the region set and cropping regions
* ``ocr``      - OCR calls (Tesseract preprocessing + process launches)
* ``parse``    - parsing / validating OCR text into values
* ``cache``    - region fingerprinting and cache lookups
* ``panels``   - Town Center / villager panel classifiers, OCR excluded
* ``track``    - merging the observation into the tracked state

Detectors run in parallel threads, so the summed stage times of one observation
can exceed its wall time; ``total`` is the wall time.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import numpy as np


class StageTimer:
    """Thread-safe accumulator of time per named stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seconds: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    def add(self, stage: str, seconds: float, calls: int = 1) -> None:
        with self._lock:
            self._seconds[stage] += seconds
            self._calls[stage] += calls

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def seconds(self) -> dict[str, float]:
        with self._lock:
            return dict(self._seconds)

    def calls(self) -> dict[str, int]:
        with self._lock:
            return dict(self._calls)

    def merge(self, other: "StageTimer") -> None:
        secs, calls = other.seconds(), other.calls()
        for name, s in secs.items():
            self.add(name, s, calls.get(name, 0))


class TimedOcr:
    """OCR engine proxy that records every ``read`` under the ``ocr`` stage."""

    def __init__(self, engine, timer: StageTimer) -> None:
        self.engine = engine
        self.timer = timer

    def read(self, crop: np.ndarray, **kwargs):
        with self.timer.stage("ocr"):
            return self.engine.read(crop, **kwargs)

    def __getattr__(self, name: str):  # config_key, launches, ... of the wrapped engine
        return getattr(self.engine, name)


def ocr_launches(engine) -> int | None:
    """Tesseract process launches so far, when the engine counts them."""
    launches = getattr(engine, "launches", None)
    return launches if isinstance(launches, int) else None


def ocr_memo_hits(engine) -> int | None:
    """Tesseract passes answered from the memo so far (launches avoided), when the engine has one."""
    for e in (engine, getattr(engine, "fallback", None), getattr(engine, "engine", None)):
        hits = getattr(e, "memo_hits", None) if e is not None else None
        if isinstance(hits, int):
            return hits
    return None
