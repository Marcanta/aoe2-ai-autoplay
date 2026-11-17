"""IdleVillagerController: finds idle villagers and selects them, one at a time. Deterministic, no LLM.

Closed loop, one :meth:`IdleVillagerController.cycle` at a time:

    Observe -> Evaluate -> Act -> Wait -> Verify -> (back to) Observe

* **Observe**: the top-bar idle-villager button (yellow = some idle, grey = none,
  plus the count) and the selection panel (is a villager selected, and which one),
  see :mod:`aoe2_ai.perception.villager`.
* **Evaluate** is the pure function :func:`decide`. It only answers SELECT when the
  game is detected, the button is *seen* yellow, and the current pass still has
  villagers left to visit. An unreadable button means WAIT, never "no idle" and
  never "some idle".
* **Act** presses the game's "next idle villager" hotkey (default ``.``) through the
  shared :class:`~aoe2_ai.execution.ActionExecutor`: foreground-window guard,
  emergency stop, cooldown, and no second press while the first is unverified.
* **Verify** polls observations: a villager must be seen selected (by name, read
  this frame). The hotkey's success is never assumed.

**Not selecting the same villager again.** The game's hotkey cycles through the idle
villagers, so ``N`` presses visit ``N`` different ones and the next press wraps around.
The controller works in *passes*: a pass starts when the idle indicator changes
(count or presence) and ends after ``N`` verified selections (``N`` = count read on the
button, or 1 when it could not be read), or as soon as a selection matches a villager
already visited in this pass (role, HP and the view around the camera centre). Then it
waits until the idle set changes. Nothing is assigned to the villagers yet, so without
that rule it would cycle through the same ones forever.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Callable

from aoe2_ai.execution import Action, ActionExecutor, ActionResult, ActionStatus, EmergencyStop
from aoe2_ai.perception.villager import VillagerFingerprint

from .observation import Observation, Observer
from .town_center import DebugRecorder, FailureRecord

log = logging.getLogger("aoe2_ai.controllers.idle_villager")

SELECT_IDLE_KEY = "select_idle_villager"


@dataclass(frozen=True)
class IdleVillagerConfig:
    """Hotkey and timings (seconds). Conservative by default."""

    idle_hotkey: str = "."  # AoE2 DE default: "Go to next idle villager"
    action_post_delay: float = 0.3
    select_cooldown: float = 1.0  # min time between two presses, even when verified
    verify_delay: float = 0.5  # after the press, before the first check
    verify_timeout: float = 5.0  # no villager seen selected by then -> uncertain
    verify_interval: float = 0.4
    other_confirmations: int = 2  # non-villager selection read this many times -> the press did not take
    same_view_threshold: float = 0.9  # view-thumbnail correlation above which two villagers are the same one
    observe_interval: float = 1.0
    failure_backoff: float = 10.0
    min_in_game_observations: int = 2
    max_consecutive_failures: int = 3
    debug_dir: str = "captures/idle_villager"

    @classmethod
    def from_json(cls, path: str | Path) -> "IdleVillagerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def select_action(self) -> Action:
        return Action.key(SELECT_IDLE_KEY, self.idle_hotkey, post_delay=self.action_post_delay,
                          dedupe_key=SELECT_IDLE_KEY, cooldown=self.select_cooldown)


# --- passes ------------------------------------------------------------------
@dataclass
class SelectionPass:
    """Villagers visited since the idle indicator last changed."""

    key: tuple  # (present, count or None) of the indicator that started the pass
    target: int  # how many different villagers the hotkey can reach
    visited: list[VillagerFingerprint] = field(default_factory=list)
    complete: bool = False
    why_complete: str = ""

    @property
    def remaining(self) -> int:
        return 0 if self.complete else max(0, self.target - len(self.visited))


def indicator_key(obs: Observation) -> tuple | None:
    """What identifies "the same idle set" for pass bookkeeping, or None when the button is unreadable."""
    idle = obs.idle
    if not idle.present.is_valid:
        return None
    return (idle.present.value, idle.count.value if idle.count.is_valid else None)


# --- Evaluate ------------------------------------------------------------------
class DecisionKind(str, Enum):
    WAIT = "wait"
    SELECT_IDLE_VILLAGER = "select_idle_villager"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str


def decide(obs: Observation, current: SelectionPass | None) -> Decision:
    """What to do given one observation and the current pass. Pure: no I/O."""
    W, S = DecisionKind.WAIT, DecisionKind.SELECT_IDLE_VILLAGER
    if not obs.game_detected:
        return Decision(W, f"game not detected: {obs.game_reason}")
    idle = obs.idle
    if not idle.present.is_valid:
        return Decision(W, f"idle villager indicator unknown ({idle.present.raw_text})")  # unknown is neither yes nor no
    if idle.present.value is False:
        return Decision(W, "no idle villagers")
    count = f"{idle.count.value}" if idle.count.is_valid else "unknown number of"
    if current is not None and current.remaining == 0:
        return Decision(W, f"{count} idle villager(s), all visited in this pass ({current.why_complete or 'done'}); waiting for a change")
    return Decision(S, f"{count} idle villager(s)")


# --- records -------------------------------------------------------------------
class ControllerState(str, Enum):
    OBSERVING = "observing"
    SELECTING = "selecting"
    VERIFYING = "verifying"
    BACKOFF = "backoff"
    HALTED = "halted"
    STOPPED = "stopped"


class Outcome(str, Enum):
    WAITED = "waited"
    SELECTED = "selected"  # a new idle villager, verified on screen
    PASS_COMPLETE = "pass_complete"  # the hotkey came back to a villager already visited
    DRY_RUN = "dry_run"
    BLOCKED = "blocked"  # executor refused (no game, pending, cooldown, stopped)
    FAILED = "failed"
    UNCERTAIN = "uncertain"  # could not tell what the press did
    HALTED = "halted"
    STOPPED = "stopped"


@dataclass(frozen=True)
class CycleReport:
    outcome: Outcome
    state: ControllerState
    decision: Decision | None = None
    action: ActionResult | None = None
    detail: str = ""
    villager: VillagerFingerprint | None = None


@dataclass
class Stats:
    cycles: int = 0
    selected: int = 0
    passes: int = 0
    failures: int = 0
    uncertain: int = 0


# --- controller ------------------------------------------------------------------
class IdleVillagerController:
    def __init__(
        self,
        observer: Observer,
        executor: ActionExecutor,
        config: IdleVillagerConfig = IdleVillagerConfig(),
        *,
        recorder: DebugRecorder | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self.observer = observer
        self.executor = executor
        self.config = config
        self.stop: EmergencyStop = executor.stop
        self.recorder = recorder if recorder is not None else DebugRecorder(config.debug_dir)
        self.clock = clock
        self._sleep = sleep or self.stop.wait
        self.state = ControllerState.OBSERVING
        self.failures: list[FailureRecord] = []
        self.stats = Stats()
        self.consecutive_failures = 0
        self.current_pass: SelectionPass | None = None
        self.last_selected: VillagerFingerprint | None = None
        self._backoff_until = 0.0
        self._last_obs: Observation | None = None
        self._in_game_streak = 0

    # --- helpers -----------------------------------------------------------------
    def _observe(self) -> Observation:
        obs = self.observer.observe()
        self._last_obs = obs
        self._in_game_streak = self._in_game_streak + 1 if obs.game_detected else 0
        return obs

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)

    def _fail(self, phase: str, reason: str, obs: Observation | None) -> None:
        shot = self.recorder.capture(obs or self._last_obs, f"{phase}_{reason[:40]}")
        self.failures.append(FailureRecord(time.time(), phase, reason, shot))
        self.stats.failures += 1
        self.consecutive_failures += 1
        self._backoff_until = self.clock() + self.config.failure_backoff
        log.error("FAILURE [%s] %s%s", phase, reason, f" (screenshot {shot})" if shot else "")
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.state = ControllerState.HALTED
            log.error("halting after %d consecutive failures", self.consecutive_failures)
        else:
            self.state = ControllerState.BACKOFF

    def _update_pass(self, obs: Observation) -> None:
        """Start a new pass when the idle indicator changes. An unreadable indicator changes nothing."""
        key = indicator_key(obs)
        if key is None:
            return
        present, count = key
        if not present:
            if self.current_pass is not None:
                log.info("no idle villagers left: pass over")
            self.current_pass = None
            return
        if self.current_pass is not None and count is None:
            return  # count unread this frame (seen live: "7" then unread): not a change of the idle set
        if self.current_pass is not None and self.current_pass.key[1] is None:
            # First count read for a pass started without one: same idle set, now with a size.
            self.current_pass.key, self.current_pass.target = key, count
            return
        if self.current_pass is None or self.current_pass.key != key:
            target = count if count is not None else 1  # unknown count: one at a time, re-checked each pass
            self.current_pass = SelectionPass(key, target)
            self.stats.passes += 1
            log.info("new pass: %s idle villager(s) to visit", count if count is not None else "unknown number of (1 per pass)")

    def _same(self, a: VillagerFingerprint, b: VillagerFingerprint) -> bool | None:
        return a.same_as(b, self.config.same_view_threshold)

    # --- main cycle ------------------------------------------------------------------
    def cycle(self) -> CycleReport:
        """One Observe -> Evaluate -> Act -> Wait -> Verify pass. Never raises for game-side problems."""
        self.stats.cycles += 1
        if self.stop.triggered:
            self.state = ControllerState.STOPPED
            return CycleReport(Outcome.STOPPED, self.state, detail=self.stop.reason)
        if self.state is ControllerState.HALTED:
            return CycleReport(Outcome.HALTED, self.state, detail="halted after repeated failures")

        obs = self._observe()
        need = self.config.min_in_game_observations
        if 0 < self._in_game_streak < need:
            return CycleReport(Outcome.WAITED, self.state, detail=f"game just detected ({self._in_game_streak}/{need} observations), letting it settle")

        if obs.game_detected:
            self._update_pass(obs)
        decision = decide(obs, self.current_pass)
        log.debug("decision %s: %s", decision.kind.value, decision.reason)

        if self.state is ControllerState.BACKOFF:
            if self.clock() < self._backoff_until:
                return CycleReport(Outcome.WAITED, self.state, decision, detail=f"in back-off after a failure ({decision.reason})")
            self.state = ControllerState.OBSERVING
        if decision.kind is DecisionKind.WAIT:
            return CycleReport(Outcome.WAITED, self.state, decision, detail=decision.reason)
        return self._select(decision, obs)

    def _select(self, decision: Decision, before: Observation) -> CycleReport:
        self.state = ControllerState.SELECTING
        result = self.executor.execute(self.config.select_action(), await_verification=True)
        if result.blocked:
            self.state = ControllerState.OBSERVING
            return CycleReport(Outcome.BLOCKED, self.state, decision, result, result.detail)
        if result.status is ActionStatus.DRY_RUN:
            self.state = ControllerState.OBSERVING
            log.info("DRY RUN: would select an idle villager (%s)", decision.reason)
            return CycleReport(Outcome.DRY_RUN, self.state, decision, result, "would select an idle villager")
        if result.status is ActionStatus.FAILED:
            # Selecting creates nothing, so there is no cost to settle: the next observation
            # shows whatever is selected. Release the key, record, back off.
            self.executor.resolve(SELECT_IDLE_KEY, "input error")
            self.stats.uncertain += 1
            self._fail("select", f"idle hotkey input error: {result.detail}", before)
            return CycleReport(Outcome.UNCERTAIN, self.state, decision, result, result.detail)

        self.state = ControllerState.VERIFYING
        verdict, detail, after = self._verify(before)
        fp = after.villager.fingerprint if after is not None else None
        current = self.current_pass

        if verdict == "selected":
            self.executor.resolve(SELECT_IDLE_KEY, "villager selected")
            self.consecutive_failures = 0
            self.state = ControllerState.OBSERVING
            repeat = self._visited_match(fp)
            if repeat is not None:
                if current is not None:
                    current.complete, current.why_complete = True, f"came back to a visited villager ({repeat})"
                log.info("idle hotkey came back to a villager already visited (%s): pass complete", repeat)
                self.last_selected = fp
                return CycleReport(Outcome.PASS_COMPLETE, self.state, decision, result, f"selected a villager already visited ({repeat})", fp)
            if current is not None and fp is not None:
                current.visited.append(fp)
                if current.remaining == 0:
                    current.why_complete = f"{len(current.visited)}/{current.target} visited"
            self.last_selected = fp
            self.stats.selected += 1
            log.info("idle villager selected and verified (%s)", detail)
            return CycleReport(Outcome.SELECTED, self.state, decision, result, detail, fp)

        self.executor.resolve(SELECT_IDLE_KEY, verdict)
        if verdict == "uncertain":
            self.stats.uncertain += 1
            self._fail("verify_selection", detail, after)
            return CycleReport(Outcome.UNCERTAIN, self.state, decision, result, detail)
        self._fail("verify_selection", detail, after)
        return CycleReport(Outcome.FAILED, self.state, decision, result, detail)

    def _visited_match(self, fp: VillagerFingerprint | None) -> str | None:
        """Why ``fp`` may be a villager already visited in this pass, or None when it is new.

        "Can't tell" (no view to compare, same role and HP) counts as visited: the safe
        side is to stop selecting, not to cycle through the same villagers.
        """
        current = self.current_pass
        if current is None or fp is None:
            return None
        for i, seen in enumerate(current.visited, start=1):
            same = self._same(fp, seen)
            if same is True:
                return f"same as #{i}: {seen.role}"
            if same is None:
                return f"cannot tell from #{i}: {seen.role}, no view to compare"
        return None

    def _verify(self, before: Observation) -> tuple[str, str, Observation | None]:
        """Poll until a villager is seen selected, something else is, or the timeout.

        Returns ("selected" | "not_selected" | "unchanged" | "uncertain", detail, last observation).
        """
        cfg = self.config
        prev = before.villager.fingerprint if before.villager.is_villager else None
        target = self.current_pass.target if self.current_pass else 1
        self._wait(cfg.verify_delay)
        deadline = self.clock() + cfg.verify_timeout
        others, last, why = 0, None, "no observation"
        while True:
            if self.stop.triggered:
                return "uncertain", f"stopped during verification: {self.stop.reason}", last
            last = self._observe()
            sel = last.villager
            if not last.game_detected:
                why = f"game not detected ({last.game_reason})"
            elif sel.is_villager and sel.fingerprint is not None:
                fp = sel.fingerprint
                if prev is not None and target > 1 and self._same(fp, prev) is True:
                    # Several idle villagers but the same one still selected: the press did not move on (yet).
                    why = f"still the same villager ({fp.role})"
                else:
                    hp = f" {fp.hp}/{fp.hp_max}" if fp.hp is not None else ""
                    return "selected", f"{fp.role}{hp} ({sel.selected.raw_text!r}, conf {sel.selected.confidence:.2f})", last
            elif sel.is_other:
                others += 1
                why = f"selected object is not a villager ({sel.selected.raw_text!r}, {others}/{cfg.other_confirmations})"
                if others >= cfg.other_confirmations:
                    return "not_selected", f"after '{cfg.idle_hotkey}': {why}", last
            else:
                why = f"selection unknown ({sel.selected.raw_text})"
            if self.clock() >= deadline:
                verdict = "unchanged" if why.startswith("still the same") else "uncertain"
                return verdict, f"verification timed out after '{cfg.idle_hotkey}': {why}", last
            self._wait(cfg.verify_interval)

    # --- loop --------------------------------------------------------------------
    def run(self, max_cycles: int | None = None) -> Stats:
        """Repeat cycles until emergency stop, halt, or ``max_cycles``."""
        log.info("idle villager controller started (dry_run=%s, hotkey %r)", self.executor.dry_run, self.config.idle_hotkey)
        n = 0
        while max_cycles is None or n < max_cycles:
            n += 1
            try:
                report = self.cycle()
            except Exception as exc:
                if type(exc).__name__ == "FailSafeException":
                    self.stop.trigger("pyautogui fail-safe")
                    report = CycleReport(Outcome.STOPPED, ControllerState.STOPPED)
                elif self.stop.triggered:
                    log.info("cycle interrupted by the stop: %r", exc)
                    report = CycleReport(Outcome.STOPPED, ControllerState.STOPPED, detail=self.stop.reason)
                else:
                    self._fail("cycle", f"unexpected error: {exc!r}", None)
                    log.exception("unexpected error in cycle")
                    report = CycleReport(Outcome.FAILED, self.state, detail=repr(exc))
            log.info("cycle %d: %s [%s] %s", self.stats.cycles, report.outcome.value, report.state.value, report.detail)
            if report.outcome in (Outcome.STOPPED, Outcome.HALTED) or self.stop.triggered:
                self.state = ControllerState.STOPPED if self.stop.triggered else self.state
                break
            self._wait(self.config.observe_interval)
        log.info("idle villager controller finished: %s", asdict(self.stats))
        return self.stats


# --- CLI ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from aoe2_ai.perception.profile import IDLE_VILLAGER

    from .runtime import add_common_args, run_live, setup_logging

    parser = argparse.ArgumentParser(description="Find idle villagers and select them one at a time (no orders given).")
    parser.add_argument("--config", type=Path, help="JSON file with IdleVillagerConfig fields")
    parser.add_argument("--idle-key", help="the game's 'next idle villager' hotkey (default .)")
    add_common_args(parser)
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    config = IdleVillagerConfig.from_json(args.config) if args.config else IdleVillagerConfig()
    if args.idle_key:
        config = IdleVillagerConfig(**{**asdict(config), "idle_hotkey": args.idle_key})
    run_live(args, lambda observer, executor: IdleVillagerController(observer, executor, config), config.debug_dir, IDLE_VILLAGER)


if __name__ == "__main__":
    main()
