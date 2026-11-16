"""Town Center panel perception: is a Town Center selected, is it producing, can it train villagers.

Read from the same frame and region set as the HUD, but mostly *without* OCR:

* **Queue status** comes from pixel structure, not text. When the building is
  producing, the panel shows an item icon (``tc_queue_slot``) above a progress
  bar (``tc_queue_bar``: red fill on a dark track). When it is idle, that area is
  plain parchment, which is matched against a template (``anchors/tc_queue_empty.png``).
  Both ways need *positive* evidence: a queue area that is neither a progress bar
  nor clean parchment (tooltip, hover, other panel, scaling problem) is UNKNOWN,
  never IDLE.
* **Villager available** is a template match of the villager button in the command grid.
* **Progress** and **current item** are OCR'd from the two text lines next to the
  queue icon (e.g. ``Création 40%`` / ``Villageois``). They are informational and
  only read while the queue is producing.

Why not OCR for the queue? The queue has no text at all when idle, so OCR can only
ever report "nothing read", which must not be confused with "nothing queued". The
progress bar and parchment are large, high-contrast features that classify in well
under a millisecond, while each OCR read costs a Tesseract launch (~100-200 ms).
See ``docs/town_center_controller.md`` for the measurements.

This module only describes what is on screen; deciding what to do is the controller's job.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detectors import ANCHOR_DIR, _normalize
from .game_state import Detection, Selection, Status
from .ocr import OcrEngine
from .regions import Region, RegionSet

# Selected-object names that mean "Town Center", per UI language. Matched fuzzily.
TOWN_CENTER_NAMES: tuple[str, ...] = ("Town Center", "Town Centre", "Forum", "Dorfzentrum", "Centro urbano", "Centro cittadino")
VILLAGER_NAMES: tuple[str, ...] = ("Villager", "Villageois", "Dorfbewohner", "Aldeano", "Abitante del villaggio", "Aldeão")


class QueueStatus(str, Enum):
    IDLE = "idle"  # positively seen empty
    PRODUCING = "producing"  # something (unit or research) is in production


@dataclass(frozen=True)
class TownCenterState:
    """What the selection panel says about the Town Center, for one frame.

    ``selected``: VALID True/False when the selected-object name was read;
    UNKNOWN when nothing could be read. The other fields are UNKNOWN unless
    ``selected`` is VALID True, since the panel only describes the TC then.
    """

    timestamp: float
    selected: Detection[bool]
    queue: Detection[QueueStatus]
    villager_available: Detection[bool]
    progress: Detection[int]  # percent of the item in production
    current_item: Detection[str]  # "villager" or the raw OCR name of something else
    scores: dict[str, float] = field(default_factory=dict)  # raw classifier scores, for debugging

    @classmethod
    def unknown(cls, timestamp: float, reason: str) -> "TownCenterState":
        u = Detection.unknown(timestamp, reason)
        return cls(timestamp, u, u, u, u, u)

    @property
    def is_selected(self) -> bool:
        return self.selected.is_valid and self.selected.value is True

    @property
    def is_idle(self) -> bool:
        return self.queue.is_valid and self.queue.value is QueueStatus.IDLE

    @property
    def is_producing(self) -> bool:
        return self.queue.is_valid and self.queue.value is QueueStatus.PRODUCING

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "selected": self.selected.to_dict(),
            "queue": self.queue.to_dict(),
            "villager_available": self.villager_available.to_dict(),
            "progress": self.progress.to_dict(),
            "current_item": self.current_item.to_dict(),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


# --- pure helpers ----------------------------------------------------------
def fuzzy_name_match(text: str, names: tuple[str, ...], min_ratio: float = 0.8) -> float:
    """Best similarity of ``text`` to any of ``names`` (accent/case-insensitive), or 0 below ``min_ratio``."""
    norm = _normalize(text)
    if len(norm) < 2:
        return 0.0
    best = 0.0
    for name in names:
        n = _normalize(name)
        best = max(best, 1.0 if norm == n else SequenceMatcher(None, norm, n).ratio())
    return best if best >= min_ratio else 0.0


_PERCENT = re.compile(r"(\d{1,3})\s*%")


def parse_percent(text: str) -> int | None:
    m = _PERCENT.search(text)
    if not m:
        return None
    value = int(m.group(1))
    return value if value <= 100 else None


def progress_bar_fill(crop: np.ndarray) -> float:
    """Share of pixels that belong to a progress bar: saturated red fill or the dark maroon track."""
    if crop.size == 0:
        return 0.0
    b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
    red = (r > 140) & (g < 70) & (b < 70)
    track = crop.max(axis=2) < 90
    return float((red | track).mean())


def parchment_fraction(crop: np.ndarray) -> float:
    """Share of light tan pixels: the selection panel background."""
    if crop.size == 0:
        return 0.0
    b, r = crop[..., 0].astype(np.int16), crop[..., 2].astype(np.int16)
    return float(((r > 150) & (r > b + 30)).mean())


def template_score(frame: np.ndarray, region: Region, template: np.ndarray, margin: int = 4) -> float:
    """Best normalised correlation (0..1) of ``template`` resized to ``region``, searched ``margin`` px around it."""
    if (region.w, region.h) != template.shape[1::-1]:
        template = cv2.resize(template, (region.w, region.h), interpolation=cv2.INTER_AREA)
    search = Region("search", region.x - margin, region.y - margin, region.w + 2 * margin, region.h + 2 * margin).crop(frame)
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return 0.0
    return float(max(0.0, cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).max()))


def _load_template(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"template missing: {path}")
    return image


# --- detector ----------------------------------------------------------------
class TownCenterDetector:
    """Builds a :class:`TownCenterState` from a frame and that frame's selection reading.

    Pass the selection observed in *this* frame (``PerceptionResult.observed.selection``),
    not the tracked one: a STALE selection says nothing about what is selected now.
    """

    def __init__(
        self,
        *,
        names: tuple[str, ...] = TOWN_CENTER_NAMES,
        villager_names: tuple[str, ...] = VILLAGER_NAMES,
        empty_template: Path = ANCHOR_DIR / "tc_queue_empty.png",
        villager_template: Path = ANCHOR_DIR / "villager_button.png",
        empty_threshold: float = 0.85,
        villager_threshold: float = 0.8,
        bar_producing: float = 0.6,
        bar_idle_max: float = 0.05,
        context_min: float = 0.8,
    ) -> None:
        self.names = names
        self.villager_names = villager_names
        self.empty_template = _load_template(empty_template)
        self.villager_template = _load_template(villager_template)
        self.empty_threshold = empty_threshold
        self.villager_threshold = villager_threshold
        self.bar_producing = bar_producing
        self.bar_idle_max = bar_idle_max
        self.context_min = context_min

    # selection --------------------------------------------------------------
    def selected(self, selection: Detection[Selection], timestamp: float) -> Detection[bool]:
        if selection.status is not Status.VALID or selection.value is None:
            # Unreadable name: we do not know what is selected. Never "not selected" by default.
            return Detection.unknown(timestamp, f"selection {selection.status.value}: {selection.raw_text}")
        ratio = fuzzy_name_match(selection.value.name, self.names)
        conf = selection.confidence * (ratio if ratio else 1.0)
        return Detection.valid(bool(ratio), conf, timestamp, selection.value.name)

    # queue --------------------------------------------------------------------
    def queue(self, frame: np.ndarray, regions: RegionSet, timestamp: float) -> tuple[Detection[QueueStatus], dict[str, float]]:
        needed = ("tc_queue_area", "tc_queue_bar", "tc_queue_context", "tc_queue_context_top")
        if any(n not in regions or not regions[n].calibrated for n in needed):
            return Detection.unknown(timestamp, "queue regions not configured"), {}
        bar = progress_bar_fill(regions["tc_queue_bar"].crop(frame))
        empty = template_score(frame, regions["tc_queue_area"], self.empty_template)
        # Parchment strips above the queue icon and under the bar: at least one must show the
        # selection panel. Without it (menu, loading screen, covered panel) the bar test would see
        # any dark area as a track. Two strips, so that one partly covered (e.g. by queued-item
        # icons under the bar) does not hide the queue.
        below = parchment_fraction(regions["tc_queue_context"].crop(frame))
        above = parchment_fraction(regions["tc_queue_context_top"].crop(frame))
        context = max(below, above)
        scores = {"bar_fill": bar, "empty_score": empty, "context": context, "context_below": below, "context_above": above}
        if context < self.context_min:
            return Detection.unknown(timestamp, f"selection panel not visible (context {context:.2f})", context), scores
        if bar >= self.bar_producing:
            return Detection.valid(QueueStatus.PRODUCING, bar, timestamp, f"progress bar {bar:.2f}"), scores
        if empty >= self.empty_threshold and bar <= self.bar_idle_max:
            return Detection.valid(QueueStatus.IDLE, empty, timestamp, f"empty queue {empty:.2f}"), scores
        # Neither clearly a bar nor clearly empty: a missing detection is not "idle".
        return Detection.unknown(timestamp, f"ambiguous queue: bar {bar:.2f}, empty {empty:.2f}", max(bar, empty)), scores

    def villager_available(self, frame: np.ndarray, regions: RegionSet, timestamp: float) -> tuple[Detection[bool], float]:
        if "tc_villager_button" not in regions or not regions["tc_villager_button"].calibrated:
            return Detection.unknown(timestamp, "villager button region not configured"), 0.0
        score = template_score(frame, regions["tc_villager_button"], self.villager_template)
        if score >= self.villager_threshold:
            return Detection.valid(True, score, timestamp, f"villager button {score:.2f}"), score
        # A missing match may be a hover highlight or tooltip, not proof the button is gone.
        return Detection.unknown(timestamp, f"villager button not matched ({score:.2f})", score), score

    def _ocr_line(self, frame: np.ndarray, regions: RegionSet, ocr: OcrEngine, name: str, whitelist: str | None = None):
        if name not in regions or not regions[name].calibrated:
            return None
        region = regions[name]
        crop = region.crop(frame)
        if crop.size == 0:
            return None
        return ocr.read(crop, whitelist=whitelist, dark=region.dark_text)

    # all ----------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        regions: RegionSet,
        selection: Detection[Selection],
        timestamp: float | None = None,
        ocr: OcrEngine | None = None,
        min_confidence: float = 0.6,
    ) -> TownCenterState:
        """``regions`` must already be scaled to ``frame``. ``ocr`` is optional (progress / item text)."""
        ts = time.time() if timestamp is None else timestamp
        selected = self.selected(selection, ts)
        if not (selected.is_valid and selected.value):
            reason = "town center not selected" if selected.is_valid else "selection unknown"
            u = Detection.unknown(ts, reason)
            return TownCenterState(ts, selected, u, u, u, u)

        queue, scores = self.queue(frame, regions, ts)
        villager, v_score = self.villager_available(frame, regions, ts)
        scores["villager_score"] = v_score

        progress: Detection[int] = Detection.unknown(ts, "not producing")
        item: Detection[str] = Detection.unknown(ts, "not producing")
        if queue.is_valid and queue.value is QueueStatus.PRODUCING and ocr is not None:
            p = self._ocr_line(frame, regions, ocr, "tc_queue_progress")
            if p is not None and not p.empty and p.confidence >= min_confidence and parse_percent(p.text) is not None:
                progress = Detection.valid(parse_percent(p.text), p.confidence, ts, p.text)
            elif p is not None:
                progress = Detection.unknown(ts, p.text, p.confidence)
            n = self._ocr_line(frame, regions, ocr, "tc_queue_item")
            if n is not None and not n.empty and n.confidence >= min_confidence:
                ratio = fuzzy_name_match(n.text, self.villager_names, min_ratio=0.75)
                item = Detection.valid("villager" if ratio else n.text, n.confidence * (ratio or 1.0), ts, n.text)
            elif n is not None:
                item = Detection.unknown(ts, n.text, n.confidence)
        return TownCenterState(ts, selected, queue, villager, progress, item, scores)
