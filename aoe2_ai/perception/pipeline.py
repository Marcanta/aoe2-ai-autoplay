"""Perception pipeline: Frame -> detectors -> GameState (with staleness tracking)."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from .capture import Frame
from .detectors import (
    DEFAULT_MIN_CONFIDENCE,
    Detector,
    DetectorContext,
    HudDetector,
    RegionReading,
    default_detectors,
)
from .game_state import FIELDS, GameState, StateTracker
from .metrics import StageTimer, TimedOcr, ocr_launches
from .ocr import OcrEngine, TesseractOcr
from .regions import RegionSet


@dataclass(frozen=True)
class PerceptionResult:
    state: GameState  # tracked state (STALE values filled in from earlier frames)
    observed: GameState  # what this frame alone showed
    readings: dict[str, list[RegionReading]] = field(default_factory=dict)  # per field, for debugging
    hud_score: float = 0.0
    regions: RegionSet | None = None  # scaled to the frame
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage (see metrics.py), "total" = wall time
    ocr_launches: int | None = None  # Tesseract process launches for this observation, when the engine counts them


class Perception:
    """Reads the game state from frames. Keep one instance per game session.

    ``hud_check`` skips OCR (all fields UNKNOWN/STALE) when the in-game top bar
    is not on screen, e.g. in menus or while alt-tabbed.
    """

    def __init__(
        self,
        regions: RegionSet | None = None,
        ocr: OcrEngine | None = None,
        *,
        detectors: list[Detector] | None = None,
        stale_after: float = 5.0,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        hud_check: bool = True,
    ) -> None:
        self.regions = regions or RegionSet.default()
        self.ocr = ocr or TesseractOcr()
        self.detectors = detectors or default_detectors()
        self.tracker = StateTracker(stale_after=stale_after)
        self.min_confidence = min_confidence
        self.hud = HudDetector() if hud_check else None
        self._pool = ThreadPoolExecutor(max_workers=min(len(self.detectors), os.cpu_count() or 4))

    def reset(self) -> None:
        self.tracker.reset()

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def observe(self, frame: Frame) -> PerceptionResult:
        """Run all detectors on one frame, without updating the tracker."""
        started = time.perf_counter()
        launches_before = ocr_launches(self.ocr)
        timer = StageTimer()
        result = self._observe(frame, timer)
        return self._finish(result, timer, started, launches_before)

    def _observe(self, frame: Frame, timer: StageTimer) -> PerceptionResult:
        with timer.stage("regions"):
            regions = self.regions.for_frame(frame.image)
        with timer.stage("hud"):
            hud_visible, hud_score = (True, 1.0) if self.hud is None else self.hud.visible(frame.image, regions)
        if not hud_visible:
            observed = GameState.empty(frame.timestamp, frame.source, f"HUD not visible (score {hud_score:.2f})")
            return PerceptionResult(observed, observed, {}, hud_score, regions)

        ctx = DetectorContext(frame.image, regions, TimedOcr(self.ocr, timer), frame.timestamp, self.min_confidence, timer)
        # Detectors are independent and Tesseract runs as a subprocess, so threads run them in parallel.
        values, readings = {}, {}
        for detector, (det, reads) in zip(self.detectors, self._pool.map(lambda d: d.detect(ctx), self.detectors)):
            values[detector.field], readings[detector.field] = det, reads
        missing = {name: GameState.empty(frame.timestamp, reason="no detector").detections()[name] for name in FIELDS if name not in values}
        observed = GameState(frame.timestamp, frame.source, True, **values, **missing)
        return PerceptionResult(observed, observed, readings, hud_score, regions)

    def _finish(self, result: PerceptionResult, timer: StageTimer, started: float, launches_before: int | None) -> PerceptionResult:
        timings = timer.seconds()
        timings["total"] = time.perf_counter() - started
        after = ocr_launches(self.ocr)
        launches = None if launches_before is None or after is None else after - launches_before
        return replace(result, timings=timings, ocr_launches=launches)

    def analyze(self, frame: Frame) -> PerceptionResult:
        """Observe ``frame`` and merge it with recent history."""
        started = time.perf_counter()
        launches_before = ocr_launches(self.ocr)
        timer = StageTimer()
        result = self._observe(frame, timer)
        with timer.stage("track"):
            state = self.tracker.update(result.observed)
        return self._finish(replace(result, state=state), timer, started, launches_before)
