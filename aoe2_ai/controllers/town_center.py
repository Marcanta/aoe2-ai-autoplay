"""TownCenterController: keeps the Town Center producing villagers. Deterministic, no LLM.

Closed loop, one :meth:`TownCenterController.cycle` at a time:

    Observe -> Evaluate -> Act -> Wait -> Verify -> (back to) Observe

* **Evaluate** is the pure function :func:`decide`. It only answers TRAIN when
  every input is positively known: game detected, Town Center selected (by name),
  queue *seen* idle, food read this frame and >= 50, villager button visible.
  Anything unreadable means WAIT (or SELECT when the TC is not selected), never a guess.
* **Act** goes through :class:`~aoe2_ai.execution.ActionExecutor`, which refuses
  input outside the game and refuses a second "train villager" while the first is
  unverified.
* **Verify** polls observations until the queue is seen producing (success), is
  seen idle several times (the command did not take: failure), or times out
  (UNCERTAIN). While UNCERTAIN no training command is sent; the controller only
  observes (and re-selects the TC so it *can* observe) until the queue state
  settles the question.
* On failure: debug screenshot + record, no immediate retry (back-off), and the
  controller halts after repeated failures.

Run it in a skirmish (see ``docs/town_center_controller.md``):

    python -m aoe2_ai.controllers.town_center --dry-run
    python -m aoe2_ai.controllers.town_center
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2

from aoe2_ai.execution import Action, ActionExecutor, ActionResult, ActionStatus, EmergencyStop

from .observation import Observation, Observer

log = logging.getLogger("aoe2_ai.controllers.town_center")

TRAIN_KEY = "train_villager"
SELECT_KEY = "select_town_center"


@dataclass(frozen=True)
class TownCenterConfig:
    """Hotkeys, costs and timings (seconds). Conservative by default."""

    select_hotkey: str = "h"  # AoE2 DE default: "Go to / select Town Center"
    train_hotkey: str = "q"  # AoE2 DE default grid: villager in the TC command card
    villager_food_cost: int = 50
    action_post_delay: float = 0.3  # after each key press
    select_cooldown: float = 2.0  # min time between two selection hotkeys
    train_cooldown: float = 3.0  # min time between two train hotkeys, even when verified
    selection_settle: float = 0.5  # wait after selecting, before re-observing
    verify_delay: float = 0.8  # wait after training, before the first check
    verify_timeout: float = 10.0  # give up verifying (-> UNCERTAIN) after this
    verify_interval: float = 0.5  # between verification observations
    idle_confirmations: int = 2  # idle reads needed to conclude a command did not take
    observe_interval: float = 1.0  # between cycles
    failure_backoff: float = 10.0  # no training attempt for this long after a failure
    min_in_game_observations: int = 2  # consecutive in-game observations before any key (game start, alt-tab back)
    max_consecutive_failures: int = 3  # then halt: something is wrong, a human should look
    max_uncertain_cycles: int = 20  # cycles spent unable to resolve an uncertain command, then halt
    debug_dir: str = "captures/controller"
    unknown_capture_interval: float = 60.0  # save a capture when the selected TC's queue is unreadable, at most this often

    @classmethod
    def from_json(cls, path: str | Path) -> "TownCenterConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def select_action(self) -> Action:
        return Action.key(SELECT_KEY, self.select_hotkey, post_delay=self.action_post_delay, cooldown=self.select_cooldown)

    def train_action(self) -> Action:
        return Action.key(TRAIN_KEY, self.train_hotkey, post_delay=self.action_post_delay, dedupe_key=TRAIN_KEY, cooldown=self.train_cooldown)


# --- Evaluate ------------------------------------------------------------------
class DecisionKind(str, Enum):
    WAIT = "wait"
    SELECT_TOWN_CENTER = "select_town_center"
    TRAIN_VILLAGER = "train_villager"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str


def decide(obs: Observation, config: TownCenterConfig = TownCenterConfig()) -> Decision:
    """What to do given one observation. Pure: no I/O, no memory."""
    W, S, T = DecisionKind.WAIT, DecisionKind.SELECT_TOWN_CENTER, DecisionKind.TRAIN_VILLAGER
    if not obs.game_detected:
        return Decision(W, f"game not detected: {obs.game_reason}")

    tc = obs.town_center
    if not tc.selected.is_valid:
        return Decision(S, f"selection unreadable ({tc.selected.raw_text})")
    if tc.selected.value is not True:
        return Decision(S, f"selected object is not a Town Center ({tc.selected.raw_text!r})")

    if not tc.queue.is_valid:
        return Decision(W, f"production queue unknown ({tc.queue.raw_text})")  # unknown is not idle
    if tc.is_producing:
        item = tc.current_item.value if tc.current_item.is_valid else "unknown item"
        progress = f" {tc.progress.value}%" if tc.progress.is_valid else ""
        return Decision(W, f"town center already producing ({item}{progress})")

    food = obs.state.food
    if not food.is_valid:  # UNKNOWN or STALE: not read in this frame, so not trusted
        return Decision(W, f"food unreadable ({food.status.value}: {food.raw_text!r})")
    if food.value < config.villager_food_cost:
        return Decision(W, f"insufficient food ({food.value} < {config.villager_food_cost})")

    if not (tc.villager_available.is_valid and tc.villager_available.value is True):
        return Decision(W, f"villager training not available ({tc.villager_available.raw_text})")

    pop = obs.state.population
    if pop.is_valid and pop.value.current >= pop.value.cap:
        return Decision(W, f"housed ({pop.value.current}/{pop.value.cap})")

    return Decision(T, f"town center idle, food {food.value}")


# --- records -------------------------------------------------------------------
class ControllerState(str, Enum):
    OBSERVING = "observing"
    SELECTING = "selecting"
    VERIFYING_SELECTION = "verifying_selection"
    TRAINING = "training"
    VERIFYING_TRAINING = "verifying_training"
    UNCERTAIN = "uncertain"  # a training command's outcome is unknown: no new commands
    BACKOFF = "backoff"  # recent failure: observe only until the back-off ends
    HALTED = "halted"  # too many failures; needs a human
    STOPPED = "stopped"  # emergency stop


class Outcome(str, Enum):
    WAITED = "waited"
    SELECTED = "selected"
    TRAINED = "trained"  # verified on screen
    DRY_RUN = "dry_run"
    BLOCKED = "blocked"  # executor refused (no game, pending, cooldown, stopped)
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    RESOLVED = "resolved"  # an uncertain command's outcome became known
    HALTED = "halted"
    STOPPED = "stopped"


@dataclass(frozen=True)
class FailureRecord:
    timestamp: float
    phase: str
    reason: str
    screenshot: str = ""


@dataclass(frozen=True)
class CycleReport:
    outcome: Outcome
    state: ControllerState  # state after the cycle
    decision: Decision | None = None
    action: ActionResult | None = None
    detail: str = ""


@dataclass
class Stats:
    cycles: int = 0
    villagers_queued: int = 0
    failures: int = 0
    uncertain: int = 0


class DebugRecorder:
    """Saves the frame and observation JSON of failures to ``directory``."""

    def __init__(self, directory: str | Path | None) -> None:
        self.directory = Path(directory) if directory else None

    def capture(self, obs: Observation | None, tag: str) -> str:
        if self.directory is None or obs is None:
            return ""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{re.sub(r'[^a-z0-9]+', '_', tag.lower()).strip('_')}"
            (self.directory / f"{stem}.json").write_text(json.dumps(obs.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            if obs.frame is not None:
                path = self.directory / f"{stem}.png"
                cv2.imwrite(str(path), obs.frame.image)
                return str(path)
            return str(self.directory / f"{stem}.json")
        except Exception as exc:  # debugging must never break the loop
            log.error("could not save debug capture: %r", exc)
            return ""


# --- controller ------------------------------------------------------------------
class TownCenterController:
    def __init__(
        self,
        observer: Observer,
        executor: ActionExecutor,
        config: TownCenterConfig = TownCenterConfig(),
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
        self._backoff_until = 0.0
        self._uncertain_since: float | None = None
        self._uncertain_cycles = 0
        self._idle_seen_while_uncertain = 0
        self._last_obs: Observation | None = None
        self._last_unknown_capture = float("-inf")
        self._in_game_streak = 0

    # --- helpers -----------------------------------------------------------------
    def _observe(self) -> Observation:
        obs = self.observer.observe()
        self._last_obs = obs
        self._in_game_streak = self._in_game_streak + 1 if obs.game_detected else 0
        return obs

    def _settling(self) -> str:
        """Why the game is not yet stable enough to send keys, or "" when it is.

        Right after a game loads or the window comes back to the front, the screen can still
        show the previous state (another window on top, a panel not yet redrawn) and the game
        may drop keys. Requiring a few consecutive in-game observations rides that out.
        """
        need = self.config.min_in_game_observations
        if 0 < self._in_game_streak < need:
            return f"game just detected ({self._in_game_streak}/{need} observations), letting it settle"
        return ""

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)

    def _fail(self, phase: str, reason: str, obs: Observation | None) -> None:
        shot = self.recorder.capture(obs or self._last_obs, f"{phase}_{reason[:40]}")
        record = FailureRecord(time.time(), phase, reason, shot)
        self.failures.append(record)
        self.stats.failures += 1
        self.consecutive_failures += 1
        self._backoff_until = self.clock() + self.config.failure_backoff
        log.error("FAILURE [%s] %s%s", phase, reason, f" (screenshot {shot})" if shot else "")
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.state = ControllerState.HALTED
            log.error("halting after %d consecutive failures", self.consecutive_failures)
        elif self.state is not ControllerState.UNCERTAIN:
            self.state = ControllerState.BACKOFF

    def _success(self) -> None:
        self.consecutive_failures = 0

    def _enter_uncertain(self, reason: str, obs: Observation | None) -> None:
        self.state = ControllerState.UNCERTAIN
        self._uncertain_since = self.clock()
        self._uncertain_cycles = 0
        self._idle_seen_while_uncertain = 0
        self.stats.uncertain += 1
        self._fail("verify_training", f"outcome uncertain: {reason}", obs)

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

        settling = self._settling()
        if settling:
            return CycleReport(Outcome.WAITED, self.state, detail=settling)

        if self.state is ControllerState.UNCERTAIN:
            return self._cycle_uncertain(obs)

        decision = decide(obs, self.config)
        log.debug("decision %s: %s", decision.kind.value, decision.reason)

        if decision.kind is DecisionKind.WAIT:
            self._capture_unreadable_queue(obs)
            if self.state is ControllerState.BACKOFF and self.clock() >= self._backoff_until:
                self.state = ControllerState.OBSERVING
            return CycleReport(Outcome.WAITED, self.state, decision, detail=decision.reason)

        if decision.kind is DecisionKind.SELECT_TOWN_CENTER:
            return self._select(decision, obs)

        # TRAIN_VILLAGER
        if self.state is ControllerState.BACKOFF:
            if self.clock() < self._backoff_until:
                return CycleReport(Outcome.WAITED, self.state, decision, detail="in back-off after a failure")
            self.state = ControllerState.OBSERVING
        return self._train(decision, obs)

    def _capture_unreadable_queue(self, obs: Observation) -> None:
        """Not a failure, but worth a look: the TC is selected and still its queue can't be read."""
        tc = obs.town_center
        if not (obs.game_detected and tc.is_selected and not tc.queue.is_valid):
            return
        now = self.clock()
        if now - self._last_unknown_capture < self.config.unknown_capture_interval:
            return
        self._last_unknown_capture = now
        path = self.recorder.capture(obs, "queue_unknown")
        if path:
            log.info("queue unreadable with the town center selected (%s); capture: %s", tc.queue.raw_text, path)

    def _select(self, decision: Decision, obs: Observation) -> CycleReport:
        previous = self.state
        self.state = ControllerState.SELECTING
        result = self.executor.execute(self.config.select_action())
        if result.blocked:
            self.state = previous
            return CycleReport(Outcome.BLOCKED, self.state, decision, result, result.detail)
        if result.status is ActionStatus.FAILED:
            self.state = previous
            self._fail("select", f"selection hotkey failed: {result.detail}", obs)
            return CycleReport(Outcome.FAILED, self.state, decision, result, result.detail)
        if result.status is ActionStatus.DRY_RUN:
            self.state = previous
            return CycleReport(Outcome.DRY_RUN, self.state, decision, result, "would select the town center")

        self.state = ControllerState.VERIFYING_SELECTION
        self._wait(self.config.selection_settle)
        after = self._observe()
        self.state = previous
        if after.town_center.is_selected:
            self._success()
            return CycleReport(Outcome.SELECTED, self.state, decision, result, "town center selected")
        reason = f"after '{self.config.select_hotkey}', selection is {after.town_center.selected.raw_text!r} ({after.town_center.selected.status.value})"
        self._fail("verify_selection", reason, after)
        return CycleReport(Outcome.FAILED, self.state, decision, result, reason)

    def _train(self, decision: Decision, obs: Observation) -> CycleReport:
        self.state = ControllerState.TRAINING
        result = self.executor.execute(self.config.train_action(), await_verification=True)
        if result.blocked:
            self.state = ControllerState.OBSERVING
            return CycleReport(Outcome.BLOCKED, self.state, decision, result, result.detail)
        if result.status is ActionStatus.DRY_RUN:
            self.state = ControllerState.OBSERVING
            log.info("DRY RUN: would train a villager (%s)", decision.reason)
            return CycleReport(Outcome.DRY_RUN, self.state, decision, result, "would train a villager")
        if result.status is ActionStatus.FAILED:
            # The key may or may not have reached the game: do not send anything else
            # until the screen tells us what happened.
            self._enter_uncertain(f"input error {result.detail}", obs)
            return CycleReport(Outcome.UNCERTAIN, self.state, decision, result, result.detail)

        self.state = ControllerState.VERIFYING_TRAINING
        verdict, detail, last = self._verify_training(obs)
        if verdict == "producing":
            self.executor.resolve(TRAIN_KEY, "verified")
            self.stats.villagers_queued += 1
            self._success()
            self.state = ControllerState.OBSERVING
            log.info("villager queued and verified (%s)", detail)
            return CycleReport(Outcome.TRAINED, self.state, decision, result, detail)
        if verdict == "idle":
            # Seen idle repeatedly after the key: the command did not take. Outcome known.
            self.executor.resolve(TRAIN_KEY, "verified not queued")
            self.state = ControllerState.OBSERVING
            reason = f"queue still idle after '{self.config.train_hotkey}': {detail}"
            self._fail("verify_training", reason, last)
            return CycleReport(Outcome.FAILED, self.state, decision, result, reason)
        self._enter_uncertain(detail, last)
        return CycleReport(Outcome.UNCERTAIN, self.state, decision, result, detail)

    def _verify_training(self, before: Observation) -> tuple[str, str, Observation | None]:
        """Poll until the queue is seen producing, seen idle enough times, or the timeout."""
        cfg = self.config
        self._wait(cfg.verify_delay)
        deadline = self.clock() + cfg.verify_timeout
        idle_seen, last, why = 0, None, "no observation"
        while True:
            if self.stop.triggered:
                return "uncertain", f"stopped during verification: {self.stop.reason}", last
            last = self._observe()
            tc = last.town_center
            if not last.game_detected:
                why = f"game not detected ({last.game_reason})"
            elif not tc.is_selected:
                why = f"town center not selected ({tc.selected.raw_text!r})"
            elif tc.is_producing:
                item = tc.current_item.value if tc.current_item.is_valid else "item unread"
                food = _food_delta(before, last)
                return "producing", f"queue producing ({item}){food}", last
            elif tc.is_idle:
                idle_seen += 1
                why = f"queue idle ({idle_seen}/{cfg.idle_confirmations})"
                if idle_seen >= cfg.idle_confirmations:
                    return "idle", why, last
            else:
                why = f"queue unknown ({tc.queue.raw_text})"
            if self.clock() >= deadline:
                return "uncertain", f"verification timed out: {why}", last
            self._wait(cfg.verify_interval)

    def _cycle_uncertain(self, obs: Observation) -> CycleReport:
        """Observe only, until the queue state says whether the uncertain command took."""
        self._uncertain_cycles += 1
        tc = obs.town_center
        if obs.game_detected and tc.is_selected and tc.is_producing:
            return self._resolve_uncertain("queue seen producing")
        if obs.game_detected and tc.is_selected and tc.is_idle:
            self._idle_seen_while_uncertain += 1
            if self._idle_seen_while_uncertain >= self.config.idle_confirmations:
                return self._resolve_uncertain("queue seen idle: the command did not take, or already finished")
        else:
            self._idle_seen_while_uncertain = 0

        if self._uncertain_cycles >= self.config.max_uncertain_cycles:
            self.state = ControllerState.HALTED
            log.error("could not resolve an uncertain training command after %d cycles; halting", self._uncertain_cycles)
            return CycleReport(Outcome.HALTED, self.state, detail="uncertain command never resolved")

        if obs.game_detected and not tc.is_selected:
            # Something else or nothing selected. Selecting is idempotent and creates
            # nothing; it is needed to see the queue.
            result = self.executor.execute(self.config.select_action())
            return CycleReport(Outcome.UNCERTAIN, self.state, None, result, "re-selecting the town center to resolve an uncertain command")
        return CycleReport(Outcome.UNCERTAIN, self.state, None, None, "waiting for a conclusive observation")

    def _resolve_uncertain(self, detail: str) -> CycleReport:
        self.executor.resolve(TRAIN_KEY, detail)
        self._uncertain_since = None
        self._idle_seen_while_uncertain = 0
        self.state = ControllerState.BACKOFF if self.clock() < self._backoff_until else ControllerState.OBSERVING
        log.warning("uncertain command resolved: %s", detail)
        return CycleReport(Outcome.RESOLVED, self.state, None, None, detail)

    # --- loop --------------------------------------------------------------------
    def run(self, max_cycles: int | None = None) -> Stats:
        """Repeat cycles until emergency stop, halt, or ``max_cycles``."""
        log.info("town center controller started (dry_run=%s)", self.executor.dry_run)
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
                    # e.g. Ctrl+C also kills the Tesseract child processes mid-read: a stop, not a failure.
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
        log.info("town center controller finished: %s", asdict(self.stats))
        return self.stats


def _food_delta(before: Observation, after: Observation) -> str:
    a, b = before.state.food, after.state.food
    if a.is_valid and b.is_valid:
        return f", food {a.value} -> {b.value}"
    return ""


# --- CLI ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from .runtime import add_common_args, run_live, setup_logging

    parser = argparse.ArgumentParser(description="Keep the Town Center producing villagers.")
    parser.add_argument("--config", type=Path, help="JSON file with TownCenterConfig fields")
    parser.add_argument("--select-key", help="hotkey that selects the Town Center (default h)")
    parser.add_argument("--train-key", help="hotkey that trains a villager (default q)")
    add_common_args(parser)
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    config = TownCenterConfig.from_json(args.config) if args.config else TownCenterConfig()
    overrides = {k: v for k, v in (("select_hotkey", args.select_key), ("train_hotkey", args.train_key)) if v}
    if overrides:
        config = TownCenterConfig(**{**asdict(config), **overrides})
    run_live(args, lambda observer, executor: TownCenterController(observer, executor, config), config.debug_dir)


if __name__ == "__main__":
    main()
