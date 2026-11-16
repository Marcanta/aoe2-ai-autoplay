"""Idle-villager perception: are there idle villagers, how many, and is a villager selected.

How AoE2 DE shows it (measured on the fixture frames, 1680x1050, see
``docs/idle_villager_controller.md``):

* **Idle villager indicator.** The top bar has a round button right of the
  population (``idle_icon``) with the number of idle villagers in its corner
  (``idle_count``). With at least one idle villager the button is **yellow**;
  with none it is **grey** and shows ``0``. The colour is the primary signal:
  it is a large, saturated area that classifies in microseconds and separates
  the two cases by a wide margin (yellow share 0.37-0.38 vs 0.00; grey share
  0.47 vs <= 0.06). A button that is neither clearly yellow nor clearly grey
  (dimmed pause overlay, hover highlight, menu) is UNKNOWN, never "no idle".
* **Idle count.** OCR of the small digits, read only when the button is yellow.
  On the grey button the light background bleeds into the digit crop (``0`` reads
  as ``40``), so a grey button gives count 0 from the colour alone. A count that
  contradicts the colour (yellow but ``0``) is UNKNOWN.
* **Villager selected.** Two independent signals:
  the selected-object name (existing OCR) fuzzy-matches a villager role name
  ("Villageois", "Bûcheron", ... an idle villager keeps its last task's name), and
  the first command-grid slot shows the villager "build economic building" button
  (template ``anchors/villager_build_button.png``; 1.00 on every villager frame,
  <= 0.23 elsewhere). The name decides, because the grid can be on a build
  sub-page; the button alone (no name read) means "villagers selected, but not
  one we could name", e.g. a multi-selection: UNKNOWN, not "a villager".
* **Fingerprint.** To tell villagers apart without a unit id: the role name, the
  HP, and a small grey thumbnail of the middle of the screen. The idle-villager
  hotkey centres the camera on the villager it selects, so the thumbnail mostly
  shows where that villager stands.

This module only describes what is on screen; deciding what to do is the controller's job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cache import RegionCache, cached
from .detectors import ANCHOR_DIR, _normalize
from .game_state import Detection, Selection, Status
from .ocr import OcrEngine
from .regions import RegionSet
from .town_center import _load_template, fuzzy_name_match, template_score

# An idle villager keeps the name of its last task, so every role counts. Taken from the game's
# string tables (resources/<lang>/strings/key-value), ids 5121-5127, 5402, 5482, 5496, 5499, 5606-5607.
# Add other UI languages from the same ids.
VILLAGER_ROLE_NAMES: tuple[str, ...] = (
    "Villager", "Builder", "Farmer", "Hunter", "Lumberjack", "Stone Miner", "Repairer",
    "Forager", "Gold Miner", "Shepherd", "Fisherman",
    "Villageois", "Villageoise", "Constructeur", "Fermier", "Chasseur", "Bûcheron", "Mineur de pierre", "Réparateur",
    "Cueilleur", "Mineur d'or", "Berger", "Pêcheur",
)

MAX_IDLE = 500


# --- pure helpers ----------------------------------------------------------
def idle_icon_colors(crop: np.ndarray) -> tuple[float, float]:
    """(yellow share, light-grey share) of the idle-villager button."""
    if crop.size == 0:
        return 0.0, 0.0
    c = crop.astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    yellow = (r > 170) & (g > 110) & (b < 100) & (r - b > 100)
    hi, lo = c.max(axis=2), c.min(axis=2)
    grey = (hi - lo < 30) & (hi > 140)
    return float(yellow.mean()), float(grey.mean())


def white_only(crop: np.ndarray) -> np.ndarray:
    """Keep the white digits, drop the yellow button behind them.

    The count is white on the yellow icon. The OCR's colour pass otherwise also sees the
    yellow silhouette and votes "47" against "7", which drops the confidence below the
    threshold (seen live: 0.58 -> 0.73-0.96 with this).
    """
    if crop.size == 0 or crop.ndim != 3:
        return crop
    return np.repeat(crop.min(axis=2)[..., None], 3, axis=2)


def parse_idle_count(text: str) -> int | None:
    text = text.replace(" ", "")
    if not text.isdigit() or (len(text) > 1 and text[0] == "0"):
        return None
    value = int(text)
    return value if value <= MAX_IDLE else None


def best_role(text: str, names: tuple[str, ...] = VILLAGER_ROLE_NAMES, min_ratio: float = 0.8) -> tuple[str, float] | None:
    """Canonical (normalised) role name closest to ``text`` and its similarity, or None."""
    norm = _normalize(text)
    if len(norm) < 2:
        return None
    best, best_ratio = "", 0.0
    for name in names:
        ratio = fuzzy_name_match(text, (name,), min_ratio)
        if ratio > best_ratio:
            best, best_ratio = _normalize(name), ratio
    return (best, best_ratio) if best_ratio else None


def view_signature(crop: np.ndarray, size: tuple[int, int] = (32, 28)) -> np.ndarray | None:
    """Small zero-mean, unit-norm grey thumbnail; None for a flat or empty crop."""
    if crop.size == 0:
        return None
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    small = cv2.resize(grey, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    small -= small.mean()
    norm = float(np.linalg.norm(small))
    if norm < 1e-3 * small.size:
        return None  # flat (black screen): says nothing about where we are
    return small / norm


def signature_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Correlation in [-1, 1], or None when either side is missing."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.dot(a, b))


# --- states ------------------------------------------------------------------
@dataclass(frozen=True)
class IdleVillagerState:
    """The top-bar idle-villager button, for one frame.

    ``present``: VALID True (yellow button), VALID False (grey button), UNKNOWN otherwise.
    ``count``: VALID when read and consistent with ``present``; 0 when the button is grey.
    """

    timestamp: float
    present: Detection[bool]
    count: Detection[int]
    scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def unknown(cls, timestamp: float, reason: str) -> "IdleVillagerState":
        u = Detection.unknown(timestamp, reason)
        return cls(timestamp, u, u)

    @property
    def has_idle(self) -> bool:
        return self.present.is_valid and self.present.value is True

    @property
    def none_idle(self) -> bool:
        return self.present.is_valid and self.present.value is False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "present": self.present.to_dict(),
            "count": self.count.to_dict(),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


@dataclass(frozen=True)
class VillagerFingerprint:
    """Enough to tell two selected villagers apart, without a unit id."""

    role: str  # canonical role name, normalised ("villageois"), so OCR noise does not split one villager in two
    hp: int | None
    hp_max: int | None
    view: np.ndarray | None = field(default=None, compare=False, repr=False)

    def similarity(self, other: "VillagerFingerprint") -> float | None:
        return signature_similarity(self.view, other.view)

    def same_as(self, other: "VillagerFingerprint", threshold: float = 0.9) -> bool | None:
        """True / False when it can tell, None when it can't (no view to compare).

        Different role or HP: different villagers. Same role and HP (most idle villagers are
        "Villager 25/25"): decided by the view around the camera centre.
        """
        if self.role != other.role:
            return False
        if None not in (self.hp, other.hp) and (self.hp, self.hp_max) != (other.hp, other.hp_max):
            return False
        sim = self.similarity(other)
        if sim is None:
            return None
        return sim >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "hp": self.hp, "hp_max": self.hp_max, "has_view": self.view is not None}


@dataclass(frozen=True)
class VillagerSelectionState:
    """Is exactly one villager selected (this frame), and which one.

    ``selected``: VALID True when the name read this frame is a villager role; VALID False
    when a name was read and it is not a villager and the villager command button is absent;
    UNKNOWN otherwise (nothing read, multi-selection, contradicting signals).
    """

    timestamp: float
    selected: Detection[bool]
    fingerprint: VillagerFingerprint | None = None
    scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def unknown(cls, timestamp: float, reason: str) -> "VillagerSelectionState":
        return cls(timestamp, Detection.unknown(timestamp, reason))

    @property
    def is_villager(self) -> bool:
        return self.selected.is_valid and self.selected.value is True

    @property
    def is_other(self) -> bool:
        return self.selected.is_valid and self.selected.value is False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "selected": self.selected.to_dict(),
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


# --- detector ----------------------------------------------------------------
class VillagerDetector:
    """Reads the idle-villager button and the villager selection from one frame."""

    def __init__(
        self,
        *,
        role_names: tuple[str, ...] = VILLAGER_ROLE_NAMES,
        build_template: Path = ANCHOR_DIR / "villager_build_button.png",
        build_threshold: float = 0.8,
        yellow_present: float = 0.25,
        yellow_max_none: float = 0.02,
        grey_none: float = 0.35,
        grey_max_present: float = 0.15,
        name_ratio: float = 0.8,
    ) -> None:
        self.role_names = role_names
        self.build_template = _load_template(build_template)
        self.build_threshold = build_threshold
        self.yellow_present = yellow_present
        self.yellow_max_none = yellow_max_none
        self.grey_none = grey_none
        self.grey_max_present = grey_max_present
        self.name_ratio = name_ratio

    # idle button --------------------------------------------------------------
    def idle(self, frame: np.ndarray, regions: RegionSet, timestamp: float | None = None,
             ocr: OcrEngine | None = None, min_confidence: float = 0.6, cache: RegionCache | None = None) -> IdleVillagerState:
        """With ``cache``, the count is reused while the pixels of ``idle_count`` are unchanged
        (the colour check always runs)."""
        ts = time.time() if timestamp is None else timestamp
        if "idle_icon" not in regions or not regions["idle_icon"].calibrated:
            return IdleVillagerState.unknown(ts, "idle_icon region not configured")
        yellow, grey = idle_icon_colors(regions["idle_icon"].crop(frame))
        scores = {"yellow": yellow, "grey": grey}

        if yellow >= self.yellow_present and grey <= self.grey_max_present:
            present = Detection.valid(True, min(1.0, yellow / 0.35), ts, f"yellow button {yellow:.2f}")
        elif grey >= self.grey_none and yellow <= self.yellow_max_none:
            present = Detection.valid(False, min(1.0, grey / 0.45), ts, f"grey button {grey:.2f}")
            return IdleVillagerState(ts, present, Detection.valid(0, present.confidence, ts, "grey button"), scores)
        else:
            reason = f"idle button unclear (yellow {yellow:.2f}, grey {grey:.2f})"
            return IdleVillagerState(ts, Detection.unknown(ts, reason), Detection.unknown(ts, reason), scores)

        count: Detection[int] = Detection.unknown(ts, "count not read")
        if ocr is not None and "idle_count" in regions and regions["idle_count"].calibrated:
            raw = regions["idle_count"].crop(frame)
            count = cached(cache, "idle_count", (raw,), (min_confidence,),
                           lambda: self._count(raw, regions["idle_count"].dark_text, ocr, ts, min_confidence, count))
        return IdleVillagerState(ts, present, count, scores)

    @staticmethod
    def _count(raw: np.ndarray, dark: bool, ocr: OcrEngine, ts: float, min_confidence: float, default: Detection[int]) -> Detection[int]:
        crop = white_only(raw)
        read = ocr.read(crop, whitelist="0123456789", dark=dark) if crop.size else None
        if read is not None and not read.empty and read.confidence >= min_confidence:
            value = parse_idle_count(read.text)
            if value is not None and value >= 1:
                return Detection.valid(value, read.confidence, ts, read.text)
            # "0" on a yellow button, or garbage: contradicts the colour
            return Detection.unknown(ts, f"count {read.text!r} inconsistent with yellow button", read.confidence)
        if read is not None:
            return Detection.unknown(ts, read.text or "count unreadable", read.confidence)
        return default

    # selection ---------------------------------------------------------------
    def build_button_score(self, frame: np.ndarray, regions: RegionSet) -> float:
        if "villager_build_button" not in regions or not regions["villager_build_button"].calibrated:
            return 0.0
        return template_score(frame, regions["villager_build_button"], self.build_template)

    def selection(self, frame: np.ndarray, regions: RegionSet, selection: Detection[Selection],
                  timestamp: float | None = None) -> VillagerSelectionState:
        """``selection`` must be this frame's reading (``PerceptionResult.observed.selection``)."""
        ts = time.time() if timestamp is None else timestamp
        button = self.build_button_score(frame, regions)
        has_button = button >= self.build_threshold
        scores = {"build_button": button}

        if selection.status is not Status.VALID or selection.value is None:
            why = "villager command card but no single name (multi-selection?)" if has_button else "selection unreadable"
            return VillagerSelectionState(ts, Detection.unknown(ts, f"{why}: {selection.raw_text}", button), None, scores)

        name = selection.value.name
        match = best_role(name, self.role_names, self.name_ratio)
        ratio = match[1] if match else 0.0
        scores["name_ratio"] = ratio
        if match:
            # A build sub-page can hide the button, so the name is enough; both together is stronger.
            conf = selection.confidence * ratio * (1.0 if has_button else 0.9)
            view = None
            if "view_center" in regions:
                view = view_signature(regions["view_center"].crop(frame))
            fp = VillagerFingerprint(match[0], selection.value.hp, selection.value.hp_max, view)
            return VillagerSelectionState(ts, Detection.valid(True, conf, ts, name), fp, scores)
        if has_button:
            # A villager's command card under a name that is not a villager: misread name.
            return VillagerSelectionState(ts, Detection.unknown(ts, f"villager command card but name {name!r}", button), None, scores)
        return VillagerSelectionState(ts, Detection.valid(False, selection.confidence, ts, name), None, scores)
