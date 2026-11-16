"""ActionExecutor: the only place that sends input to the game.

Every call goes through the same gates, in order:

1. emergency stop not triggered,
2. no unverified earlier action with the same ``dedupe_key`` (``await_verification``),
3. the action's cooldown has elapsed,
4. the game is the foreground window (checked again right before sending),

then waits ``pre_delay``, sends the input (unless dry-run), waits ``post_delay``.
Every attempt, blocked or not, is logged and kept in ``history``.

An action sent with ``await_verification=True`` stays *pending* until the
caller calls :meth:`resolve` once it has checked the outcome on screen. While
it is pending, the executor refuses any action with the same dedupe key. The
key is marked pending *before* the input is sent, so an input error halfway
through still leaves it pending: the game may or may not have received it.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Protocol

from .actions import Action, ActionKind, ActionResult, ActionStatus
from .guard import GameGuard
from .stop import EmergencyStop

log = logging.getLogger("aoe2_ai.execution")


class InputBackend(Protocol):
    """What the executor needs from an input layer (``aoe2_ai.input.InputController`` fits)."""

    def press(self, key: str) -> None: ...
    def hotkey(self, *keys: str) -> None: ...
    def click(self, x: int, y: int, button: str = "left") -> None: ...


class ActionExecutor:
    def __init__(
        self,
        backend: InputBackend | None,
        guard: GameGuard,
        *,
        dry_run: bool = False,
        stop: EmergencyStop | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] | None = None,
        history_limit: int = 1000,
    ) -> None:
        if backend is None and not dry_run:
            raise ValueError("an input backend is required unless dry_run=True")
        self.backend = backend
        self.guard = guard
        self.dry_run = dry_run
        self.stop = stop or EmergencyStop()
        self.clock = clock
        self._sleep = sleep or self.stop.wait  # delays end early on emergency stop
        self.history: list[ActionResult] = []
        self.history_limit = history_limit
        self._pending: dict[str, ActionResult] = {}
        self._last_attempt: dict[str, float] = {}

    # --- pending verification -------------------------------------------------
    def is_pending(self, dedupe_key: str) -> bool:
        return dedupe_key in self._pending

    @property
    def pending(self) -> dict[str, ActionResult]:
        return dict(self._pending)

    def resolve(self, dedupe_key: str, outcome: str = "") -> None:
        """The caller has verified (or ruled out) the outcome of the pending action."""
        if self._pending.pop(dedupe_key, None) is not None:
            log.info("resolved %s%s", dedupe_key, f": {outcome}" if outcome else "")

    # --- execution --------------------------------------------------------------
    def execute(self, action: Action, *, await_verification: bool = False) -> ActionResult:
        started = self.clock()
        key = action.guard_key

        if self.stop.triggered:
            return self._record(action, ActionStatus.BLOCKED_STOPPED, started, self.stop.reason)
        if key in self._pending:
            return self._record(action, ActionStatus.BLOCKED_PENDING, started, f"'{key}' is still awaiting verification")
        last = self._last_attempt.get(key)
        if last is not None and action.cooldown and started - last < action.cooldown:
            return self._record(action, ActionStatus.BLOCKED_COOLDOWN, started, f"{started - last:.1f}s since last, cooldown {action.cooldown:.1f}s")
        guard = self.guard.check()
        if not guard.ok:
            return self._record(action, ActionStatus.BLOCKED_NO_GAME, started, guard.reason)

        if action.pre_delay:
            self._sleep(action.pre_delay)
            if self.stop.triggered:
                return self._record(action, ActionStatus.BLOCKED_STOPPED, started, self.stop.reason)
            guard = self.guard.check()  # focus may have changed during the delay
            if not guard.ok:
                return self._record(action, ActionStatus.BLOCKED_NO_GAME, started, guard.reason)

        self._last_attempt[key] = started
        if self.dry_run:
            result = self._record(action, ActionStatus.DRY_RUN, started, "dry run: input not sent")
            if action.post_delay:
                self._sleep(action.post_delay)
            return result

        if await_verification:
            # Before sending: if the send dies halfway, the outcome is still unknown.
            self._pending[key] = ActionResult(action, ActionStatus.EXECUTED, started, started, "sending")
        try:
            self._send(action)
        except Exception as exc:
            if type(exc).__name__ == "FailSafeException":  # pyautogui corner fail-safe
                self.stop.trigger("pyautogui fail-safe (mouse in a screen corner)")
            result = self._record(action, ActionStatus.FAILED, started, f"input error: {exc!r}")
            if await_verification:
                self._pending[key] = result
            return result

        result = self._record(action, ActionStatus.EXECUTED, started, guard.reason)
        if await_verification:
            self._pending[key] = result
        if action.post_delay:
            self._sleep(action.post_delay)
        return result

    def _send(self, action: Action) -> None:
        if action.kind is ActionKind.KEY:
            self.backend.press(action.keys[0])
        elif action.kind is ActionKind.HOTKEY:
            self.backend.hotkey(*action.keys)
        elif action.kind is ActionKind.CLICK:
            x, y = action.position
            self.backend.click(x, y, button=action.button)
        else:  # pragma: no cover - enum is closed
            raise ValueError(f"unknown action kind {action.kind}")

    def _record(self, action: Action, status: ActionStatus, started: float, detail: str) -> ActionResult:
        result = ActionResult(action, status, started, self.clock(), detail)
        self.history.append(result)
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        level = logging.INFO if status in (ActionStatus.EXECUTED, ActionStatus.DRY_RUN) else logging.WARNING
        log.log(level, "action %-16s %-18s %s", status.value, action.describe(), detail)
        return result
