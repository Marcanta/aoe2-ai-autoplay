"""Perception pipeline: Frame -> detectors -> GameState (with staleness tracking).

Two ways to make an observation cheaper, both off by default so the plain
``Perception().analyze(frame)`` call keeps reading every field from scratch:

* a :class:`~aoe2_ai.perception.profile.PerceptionProfile` limits which
  fields are computed (others are UNKNOWN, "not requested");
* ``cache=True`` reuses a field's detection while the pixels of its regions
  are bit-identical to when it was last recognised (see ``cache.py``).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Hashable

import numpy as np

from .cache import RegionCache
from .capture import Frame
from .detectors import (
    DEFAULT_MIN_CONFIDENCE,
    Detector,
    DetectorContext,
    HudDetector,
    RegionReading,
    default_detectors,
)
from .game_state import FIELDS, Detection, GameState, StateTracker
from .metrics import StageTimer, TimedOcr, ocr_launches
from .ocr import OcrEngine, TesseractOcr
from .profile import FULL, NOT_REQUESTED, PerceptionProfile
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
    profile: PerceptionProfile = FULL
    cache_hits: tuple[str, ...] = ()  # fields whose detection was reused from identical pixels


def _engine_key(ocr: OcrEngine) -> Hashable:
    key = getattr(ocr, "config_key", None)
    return key if key is not None else ("engine", id(ocr))


class Perception:
    """Reads the game state from frames. Keep one instance per game session.

    ``hud_check`` skips OCR (all fields UNKNOWN/STALE) when the in-game top bar
    is not on screen, e.g. in menus or while alt-tabbed. ``profile`` is the
    default field set (``analyze`` / ``observe`` can override it per call).
    ``cache`` turns on the region cache.
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
        profile: PerceptionProfile = FULL,
        cache: bool | RegionCache = False,
    ) -> None:
        self.regions = regions or RegionSet.default()
        self.ocr = ocr or TesseractOcr()
        self.detectors = detectors or default_detectors()
        self.tracker = StateTracker(stale_after=stale_after)
        self.min_confidence = min_confidence
        self.hud = HudDetector() if hud_check else None
        self.profile = profile
        self.cache: RegionCache | None = cache if isinstance(cache, RegionCache) else (RegionCache() if cache else None)
        self._pool = ThreadPoolExecutor(max_workers=min(len(self.detectors), os.cpu_count() or 4))

    def reset(self) -> None:
        """Forget history (new game session): tracked values and cached detections."""
        self.tracker.reset()
        if self.cache is not None:
            self.cache.clear()

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def config_key(self, frame: np.ndarray) -> Hashable:
        """Everything besides pixels that a cached detection depends on."""
        detectors = tuple((type(d).__qualname__, repr(sorted(vars(d).items()))) for d in self.detectors)
        return (frame.shape, repr(self.regions), _engine_key(self.ocr), self.min_confidence, detectors)

    def observe(self, frame: Frame, profile: PerceptionProfile | None = None) -> PerceptionResult:
        """Run the requested detectors on one frame, without updating the tracker."""
        started = time.perf_counter()
        launches_before = ocr_launches(self.ocr)
        timer = StageTimer()
        result = self._observe(frame, profile or self.profile, timer)
        return self._finish(result, timer, started, launches_before)

    def analyze(self, frame: Frame, profile: PerceptionProfile | None = None) -> PerceptionResult:
        """Observe ``frame`` and merge it with recent history."""
        started = time.perf_counter()
        launches_before = ocr_launches(self.ocr)
        timer = StageTimer()
        result = self._observe(frame, profile or self.profile, timer)
        with timer.stage("track"):
            state = self.tracker.update(result.observed)
        return self._finish(replace(result, state=state), timer, started, launches_before)

    # internals ------------------------------------------------------------------
    def _observe(self, frame: Frame, profile: PerceptionProfile, timer: StageTimer) -> PerceptionResult:
        with timer.stage("regions"):
            regions = self.regions.for_frame(frame.image)
        with timer.stage("hud"):
            hud_visible, hud_score = (True, 1.0) if self.hud is None else self.hud.visible(frame.image, regions)
        if not hud_visible:
            observed = GameState.empty(frame.timestamp, frame.source, f"HUD not visible (score {hud_score:.2f})")
            return PerceptionResult(observed, observed, {}, hud_score, regions, profile=profile)

        if self.cache is not None:
            with timer.stage("cache"):
                self.cache.configure(self.config_key(frame.image))

        ctx = DetectorContext(frame.image, regions, TimedOcr(self.ocr, timer), frame.timestamp, self.min_confidence, timer)
        values: dict[str, Detection] = {}
        readings: dict[str, list[RegionReading]] = {}
        hits: list[str] = []
        to_run: list[tuple[Detector, Hashable | None]] = []
        for detector in self.detectors:
            if detector.field not in profile:
                continue
            key = self._cache_key(detector, frame.image, regions, timer)
            cached = self.cache.get(detector.field, key) if key is not None else None
            if cached is not None:
                values[detector.field], readings[detector.field] = cached
                hits.append(detector.field)
            else:
                to_run.append((detector, key))

        # Detectors are independent and Tesseract runs as a subprocess, so threads run them in parallel.
        for (detector, key), (det, reads) in zip(to_run, self._pool.map(lambda dk: dk[0].detect(ctx), to_run)):
            values[detector.field], readings[detector.field] = det, reads
            if key is not None:
                self.cache.put(detector.field, key, (det, reads))

        for name in FIELDS:
            if name not in values:
                values[name] = Detection.unknown(frame.timestamp, NOT_REQUESTED if name not in profile else "no detector")
        observed = GameState(frame.timestamp, frame.source, True, **values)
        return PerceptionResult(observed, observed, readings, hud_score, regions, profile=profile, cache_hits=tuple(hits))

    def _cache_key(self, detector: Detector, image: np.ndarray, regions: RegionSet, timer: StageTimer) -> Hashable | None:
        names = detector.cache_regions
        if self.cache is None or names is None:
            return None
        with timer.stage("cache"):
            crops = tuple(regions[n].crop(image) if n in regions else np.empty((0, 0, 3), np.uint8) for n in names)
            return self.cache.key(crops, names)

    def _finish(self, result: PerceptionResult, timer: StageTimer, started: float, launches_before: int | None) -> PerceptionResult:
        timings = timer.seconds()
        timings["total"] = time.perf_counter() - started
        after = ocr_launches(self.ocr)
        launches = None if launches_before is None or after is None else after - launches_before
        return replace(result, timings=timings, ocr_launches=launches)
