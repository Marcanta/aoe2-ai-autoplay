"""Observations for controllers: one frame's GameState + Town Center panel + villager panels + game-detected flag."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from aoe2_ai.execution.guard import GameGuard
from aoe2_ai.perception.capture import Frame, LiveSource
from aoe2_ai.perception.game_state import GameState
from aoe2_ai.perception.pipeline import Perception
from aoe2_ai.perception.town_center import TownCenterDetector, TownCenterState
from aoe2_ai.perception.villager import IdleVillagerState, VillagerDetector, VillagerSelectionState


@dataclass(frozen=True)
class Observation:
    timestamp: float
    game_detected: bool  # game window in the foreground AND in-game HUD visible
    game_reason: str
    state: GameState  # tracked state (check each field's status)
    town_center: TownCenterState  # from this frame only
    frame: Frame | None = None  # kept for debug screenshots
    idle: IdleVillagerState = field(default_factory=lambda: IdleVillagerState.unknown(0.0, "not observed"))  # this frame only
    villager: VillagerSelectionState = field(default_factory=lambda: VillagerSelectionState.unknown(0.0, "not observed"))  # this frame only

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "game_detected": self.game_detected,
            "game_reason": self.game_reason,
            "state": self.state.to_dict(),
            "town_center": self.town_center.to_dict(),
            "idle": self.idle.to_dict(),
            "villager": self.villager.to_dict(),
            "source": self.frame.source if self.frame else "",
        }


class Observer(Protocol):
    def observe(self) -> Observation: ...


class LiveObserver:
    """Screen -> Perception -> TownCenterDetector + VillagerDetector, plus the foreground-window guard."""

    def __init__(
        self,
        guard: GameGuard,
        perception: Perception | None = None,
        detector: TownCenterDetector | None = None,
        source: LiveSource | None = None,
        villager_detector: VillagerDetector | None = None,
    ) -> None:
        self.guard = guard
        self.perception = perception or Perception()
        self.detector = detector or TownCenterDetector()
        self.source = source or LiveSource()
        self.villager_detector = villager_detector or VillagerDetector()

    def observe(self) -> Observation:
        frame = self.source.grab()
        result = self.perception.analyze(frame)
        guard = self.guard.check()
        hud = result.observed.hud_visible
        if hud and result.regions is not None:
            # The selection from *this* frame, not the tracked (possibly stale) one.
            tc = self.detector.detect(
                frame.image, result.regions, result.observed.selection, frame.timestamp,
                self.perception.ocr, self.perception.min_confidence,
            )
            idle = self.villager_detector.idle(
                frame.image, result.regions, frame.timestamp, self.perception.ocr, self.perception.min_confidence,
            )
            villager = self.villager_detector.selection(frame.image, result.regions, result.observed.selection, frame.timestamp)
        else:
            tc = TownCenterState.unknown(frame.timestamp, "in-game HUD not visible")
            idle = IdleVillagerState.unknown(frame.timestamp, "in-game HUD not visible")
            villager = VillagerSelectionState.unknown(frame.timestamp, "in-game HUD not visible")
        detected = guard.ok and hud
        reason = guard.reason if not guard.ok else ("in game" if hud else f"HUD not visible (score {result.hud_score:.2f})")
        return Observation(frame.timestamp, detected, reason, result.state, tc, frame, idle, villager)

    def close(self) -> None:
        self.perception.close()
        self.source.close()
