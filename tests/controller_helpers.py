"""Builders and fakes for execution / controller tests (no screen, no game, no real input)."""

from __future__ import annotations

from dataclasses import replace

from aoe2_ai.controllers.observation import Observation
from aoe2_ai.perception.game_state import Detection, GameState, Population, Selection, Status
from aoe2_ai.perception.town_center import QueueStatus, TownCenterState

_U = object()  # "unknown" marker for builder arguments


def det(value, ts=0.0, conf=0.95):
    return Detection.unknown(ts, "unreadable") if value is _U or value is None else Detection.valid(value, conf, ts, str(value))


def make_obs(
    *,
    food=200,
    food_status=Status.VALID,
    selected=True,
    queue=QueueStatus.IDLE,
    villager=True,
    game=True,
    population=(4, 10),
    item=_U,
    ts=0.0,
) -> Observation:
    """An observation. Pass ``None`` for any field to make it UNKNOWN."""
    food_det = det(food, ts)
    if food_status is Status.STALE and food is not None:
        food_det = replace(food_det, status=Status.STALE)
    state = GameState(
        ts, "test", game,
        food=food_det,
        population=det(Population(*population) if population else None, ts),
        selection=det(Selection("Forum" if selected else "Chasseur") if selected is not None else None, ts),
    )
    if not game:
        tc = TownCenterState.unknown(ts, "HUD not visible")
    elif selected is None or selected is False:
        sel = det(selected, ts) if selected is not None else Detection.unknown(ts, "selection unknown")
        u = Detection.unknown(ts, "town center not selected")
        tc = TownCenterState(ts, sel, u, u, u, u)
    else:
        q = det(queue, ts)
        tc = TownCenterState(ts, det(True, ts), q, det(villager if villager else None, ts), Detection.unknown(ts, "-"),
                             det("villager", ts) if queue is QueueStatus.PRODUCING and item is _U else det(item if item is not _U else None, ts))
    return Observation(ts, game, "in game" if game else "HUD not visible", state, tc, None)


IDLE = dict(queue=QueueStatus.IDLE)
PRODUCING = dict(queue=QueueStatus.PRODUCING)


class ScriptedObserver:
    """Returns the scripted observations in order, then repeats the last one."""

    def __init__(self, *observations: Observation) -> None:
        self.script = list(observations)
        self.calls = 0

    def push(self, *observations: Observation) -> None:
        self.script.extend(observations)

    def observe(self) -> Observation:
        self.calls += 1
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class MockInput:
    """Records key presses; can be told to raise."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.events: list[tuple] = []
        self.fail_with = fail_with

    def _do(self, event: tuple) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.events.append(event)

    def press(self, key: str) -> None:
        self._do(("press", key))

    def hotkey(self, *keys: str) -> None:
        self._do(("hotkey", keys))

    def click(self, x: int, y: int, button: str = "left") -> None:
        self._do(("click", x, y, button))

    def presses(self, key: str) -> int:
        return sum(1 for e in self.events if e == ("press", key))


# --- idle villager builders -------------------------------------------------------
import numpy as np  # noqa: E402

from aoe2_ai.perception.villager import IdleVillagerState, VillagerFingerprint, VillagerSelectionState  # noqa: E402


def view(seed: int) -> np.ndarray:
    """A distinct, normalised view signature per seed (a different spot on the map)."""
    v = np.random.default_rng(seed).normal(size=32 * 28).astype(np.float32)
    v -= v.mean()
    return v / np.linalg.norm(v)


def fp(seed: int, role: str = "villageois", hp=(25, 25), with_view: bool = True) -> VillagerFingerprint:
    return VillagerFingerprint(role, hp[0], hp[1], view(seed) if with_view else None)


def make_idle_obs(*, present=True, count=_U, villager=None, game=True, ts=0.0) -> Observation:
    """Observation for the idle-villager controller.

    ``present``: True / False / None (unknown button). ``count``: int, None (unread), default
    3 when present and 0 when not. ``villager``: a fingerprint (that villager selected),
    False (something else selected), None (selection unknown).
    """
    base = make_obs(game=game, selected=False, ts=ts)
    if not game or present is None:
        idle = IdleVillagerState.unknown(ts, "idle button unclear")
    else:
        if count is _U:
            count = 3 if present else 0
        idle = IdleVillagerState(ts, det(present, ts), det(count, ts) if count is not None else Detection.unknown(ts, "count unreadable"))
    if not game or villager is None:
        sel = VillagerSelectionState.unknown(ts, "selection unknown")
    elif villager is False:
        sel = VillagerSelectionState(ts, det(False, ts))
    else:
        sel = VillagerSelectionState(ts, Detection.valid(True, 0.95, ts, villager.role), villager)
    return replace(base, idle=idle, villager=sel)
