"""Perception profiles: which fields an observation computes.

Most of an observation's cost is OCR (one Tesseract launch per pass, see
``docs/perception_performance.md``), so a controller asks only for the fields
it reads. A field that is not requested is reported as UNKNOWN with the reason
``"not requested"``: it is never filled with a guess, and the tracker treats it
like any other unread field (its last value goes STALE, then UNKNOWN).

Fields:

* the :class:`~aoe2_ai.perception.game_state.GameState` fields (``wood``,
  ``food``, ``gold``, ``stone``, ``population``, ``game_time``, ``age``, ``selection``);
* ``town_center``: Town Center selected / queue / villager button (no OCR of its own);
* ``town_center_text``: the progress and item lines of the Town Center queue (OCR, informational);
* ``idle``: the idle-villager button and its count;
* ``villager``: whether one villager is selected, and its fingerprint.

Panels read the selected-object name, so ``town_center``, ``town_center_text``
and ``villager`` pull in ``selection`` automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from .game_state import FIELDS

PANEL_FIELDS = ("town_center", "town_center_text", "idle", "villager")
ALL_FIELDS = (*FIELDS, *PANEL_FIELDS)
DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "town_center": ("selection",),
    "town_center_text": ("town_center", "selection"),
    "villager": ("selection",),
}
NOT_REQUESTED = "not requested"


@dataclass(frozen=True)
class PerceptionProfile:
    name: str
    fields: frozenset[str]

    @classmethod
    def of(cls, name: str, *fields: str) -> "PerceptionProfile":
        unknown = set(fields) - set(ALL_FIELDS)
        if unknown:
            raise ValueError(f"unknown perception field(s) {sorted(unknown)}; known: {', '.join(ALL_FIELDS)}")
        wanted = set(fields)
        pending = list(wanted)
        while pending:
            for dep in DEPENDS_ON.get(pending.pop(), ()):
                if dep not in wanted:
                    wanted.add(dep)
                    pending.append(dep)
        return cls(name, frozenset(wanted))

    def __contains__(self, field: str) -> bool:
        return field in self.fields

    @property
    def is_full(self) -> bool:
        return self.fields >= set(ALL_FIELDS)

    @classmethod
    def named(cls, name: str) -> "PerceptionProfile":
        try:
            return PROFILES[name.lower().replace("-", "_")]
        except KeyError:
            raise ValueError(f"unknown profile {name!r}; known: {', '.join(PROFILES)}") from None

    def __str__(self) -> str:
        return f"{self.name} ({', '.join(f for f in ALL_FIELDS if f in self.fields)})"


FULL = PerceptionProfile.of("full", *ALL_FIELDS)
TOWN_CENTER = PerceptionProfile.of("town_center", "food", "population", "selection", "town_center")
IDLE_VILLAGER = PerceptionProfile.of("idle_villager", "idle", "selection", "villager")
ECONOMY = PerceptionProfile.of("economy", "wood", "food", "gold", "stone", "population")

PROFILES: dict[str, PerceptionProfile] = {p.name: p for p in (FULL, TOWN_CENTER, IDLE_VILLAGER, ECONOMY)}
