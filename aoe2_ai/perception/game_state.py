"""Structured game state produced by perception.

Every observed field is a :class:`Detection` carrying its value, a confidence in
[0, 1], the time it was observed and a :class:`Status`:

* ``VALID``   - read from the current frame and passed validation.
* ``STALE``   - not readable now; ``value`` is the last valid reading and
                ``timestamp`` is when that reading was made.
* ``UNKNOWN`` - no trustworthy value; ``value`` is always ``None``.

An OCR failure therefore never turns into a fabricated value such as 0: a
consumer must check ``status`` (or ``is_valid``) before trusting ``value``.

This module only describes what was seen. It holds no gameplay logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Status(str, Enum):
    VALID = "valid"
    UNKNOWN = "unknown"
    STALE = "stale"


class Age(str, Enum):
    DARK = "dark"
    FEUDAL = "feudal"
    CASTLE = "castle"
    IMPERIAL = "imperial"


@dataclass(frozen=True)
class Population:
    current: int
    cap: int


@dataclass(frozen=True)
class Selection:
    """Text read from the selected unit/building panel."""

    name: str
    hp: int | None = None
    hp_max: int | None = None


@dataclass(frozen=True)
class Detection(Generic[T]):
    value: T | None
    confidence: float
    timestamp: float
    status: Status
    raw_text: str = ""

    @classmethod
    def valid(cls, value: T, confidence: float, timestamp: float, raw_text: str = "") -> "Detection[T]":
        if value is None:
            raise ValueError("a VALID detection needs a value")
        return cls(value, max(0.0, min(1.0, confidence)), timestamp, Status.VALID, raw_text)

    @classmethod
    def unknown(cls, timestamp: float, raw_text: str = "", confidence: float = 0.0) -> "Detection[T]":
        """No usable value. ``confidence`` may record how close a rejected read came."""
        return cls(None, max(0.0, min(1.0, confidence)), timestamp, Status.UNKNOWN, raw_text)

    @property
    def is_valid(self) -> bool:
        return self.status is Status.VALID

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _jsonable(self.value),
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "status": self.status.value,
            "raw_text": self.raw_text,
        }


RESOURCE_FIELDS = ("wood", "food", "gold", "stone")
FIELDS = (*RESOURCE_FIELDS, "population", "game_time", "age", "selection")


def _unknown_factory() -> Detection:
    return Detection.unknown(0.0)


@dataclass(frozen=True)
class GameState:
    """One perception snapshot. ``game_time`` is in seconds of game time."""

    timestamp: float
    source: str = ""
    hud_visible: bool = False
    wood: Detection[int] = field(default_factory=_unknown_factory)
    food: Detection[int] = field(default_factory=_unknown_factory)
    gold: Detection[int] = field(default_factory=_unknown_factory)
    stone: Detection[int] = field(default_factory=_unknown_factory)
    population: Detection[Population] = field(default_factory=_unknown_factory)
    game_time: Detection[int] = field(default_factory=_unknown_factory)
    age: Detection[Age] = field(default_factory=_unknown_factory)
    selection: Detection[Selection] = field(default_factory=_unknown_factory)

    @classmethod
    def empty(cls, timestamp: float, source: str = "", reason: str = "") -> "GameState":
        unknown = Detection.unknown(timestamp, reason)
        return cls(timestamp, source, False, **{name: unknown for name in FIELDS})

    def detections(self) -> dict[str, Detection]:
        return {name: getattr(self, name) for name in FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "hud_visible": self.hud_visible,
            **{name: det.to_dict() for name, det in self.detections().items()},
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    return value


class StateTracker:
    """Turns per-frame readings into a state that remembers recent values.

    A field that fails to read keeps its last valid value as ``STALE`` for
    ``stale_after`` seconds (measured from when that value was observed), then
    becomes ``UNKNOWN``. A game clock that runs backwards by more than
    ``clock_tolerance`` seconds against a recent valid reading is rejected.
    """

    def __init__(self, stale_after: float = 5.0, clock_tolerance: float = 2.0) -> None:
        self.stale_after = stale_after
        self.clock_tolerance = clock_tolerance
        self._last: dict[str, Detection] = {}

    def reset(self) -> None:
        self._last.clear()

    def update(self, observed: GameState) -> GameState:
        now = observed.timestamp
        merged: dict[str, Detection] = {}
        for name in FIELDS:
            det: Detection = getattr(observed, name)
            last = self._last.get(name)
            recent = last is not None and now - last.timestamp <= self.stale_after

            if det.is_valid and name == "game_time" and recent:
                if det.value < last.value - self.clock_tolerance:
                    det = Detection.unknown(det.timestamp, f"clock went backwards: {det.raw_text}", det.confidence)

            if det.is_valid:
                self._last[name] = det
                merged[name] = det
            elif recent:
                merged[name] = replace(last, status=Status.STALE, raw_text=det.raw_text)
            else:
                merged[name] = Detection.unknown(det.timestamp or now, det.raw_text, det.confidence)
        return replace(observed, **merged)
