"""VillagerAssignmentController: sends one idle villager at a time to gather from a sheep. No LLM.

Closed loop, one :meth:`VillagerAssignmentController.cycle` per villager:

    Observe -> Evaluate -> Select -> Verify selection -> Find a sheep -> Re-check -> Right-click
    -> Wait -> Verify the assignment -> (back to) Observe

* **Observe**: the idle-villager button and the selection panel
  (:mod:`aoe2_ai.perception.villager`, same as the idle-villager controller).
* **Evaluate** is the pure function :func:`decide`. ASSIGN only when the game is detected,
  the observation is fresh, the button is *seen* yellow, and the controller is not holding
  (after "no sheep visible" or a villager it already commanded). UNKNOWN is never zero.
* **Select** presses the game's "next idle villager" hotkey (``.``) through the shared
  :class:`~aoe2_ai.execution.ActionExecutor`, then waits until a villager is seen selected
  (by name, this frame). The hotkey also centres the camera on that villager.
* **Find a sheep**: :class:`~aoe2_ai.perception.object_detector.SheepDetector` on the
  frame where the villager was verified. Only ACCEPTED detections (own colour, confident,
  not a carcass, not ambiguous, inside the world area) are targets. Preference: a sheep not
  already given to another villager, then the closest to the reference point (the Town
  Center when a locator is given, otherwise the screen centre, where the selected villager is).
* **Re-check** on a fresh frame, right before clicking: same villager still selected, the
  target still detected within ``target_tolerance`` px, game still detected. Otherwise no click.
* **Right-click** the sheep through the executor (foreground guard, emergency stop, no
  second command while the first is unverified). The click is logged with the villager and
  the target coordinates.
* **Verify the assignment**: the selected villager's name turns into the gatherer role
  ("Shepherd" / "Berger") and the idle count did not go up -> SUCCESS. The same villager
  still selected under its old idle role, with a readable idle count that did not drop, for
  the whole window -> FAILURE. Anything else (an idle-count drop alone, selection lost,
  unreadable frames, a stop) -> UNKNOWN. A sent click is never taken as proof.

**No repeats.** A villager that was sent a command (whatever the result) is remembered by
fingerprint for ``processed_ttl`` seconds; if the hotkey lands on it again, it gets no new
command and the controller holds until the idle set changes. After UNKNOWN or FAILURE the
controller backs off and, after ``max_consecutive_failures``, halts. Sheep already given to a
villager are remembered by the look of their surroundings and avoided while others exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from aoe2_ai.execution import Action, ActionExecutor, ActionResult, ActionStatus, EmergencyStop
from aoe2_ai.perception.object_detector import REFERENCE_SIZE, SheepCandidate, SheepDetector, SheepScan
from aoe2_ai.perception.villager import VillagerFingerprint, signature_similarity, view_signature

from .idle_villager import SELECT_IDLE_KEY, indicator_key
from .observation import Observation, Observer
from .town_center import DebugRecorder, FailureRecord

log = logging.getLogger("aoe2_ai.controllers.villager_assignment")

COMMAND_KEY = "assign_villager"
# Roles a villager takes when it works a herd animal (normalised, as VillagerFingerprint.role).
GATHERER_ROLES = ("shepherd", "berger")


class TargetDetector(Protocol):
    def detect(self, frame: np.ndarray | None, timestamp: float | None = None) -> SheepScan: ...


@dataclass(frozen=True)
class VillagerAssignmentConfig:
    """Hotkey, thresholds and timings (seconds, pixels). Conservative by default."""

    idle_hotkey: str = "."  # AoE2 DE default: "Go to next idle villager"
    action_post_delay: float = 0.3
    select_cooldown: float = 1.0
    command_cooldown: float = 1.0
    select_verify_delay: float = 0.5
    select_verify_timeout: float = 4.0
    other_confirmations: int = 2  # a non-villager selection read this many times -> selection failed
    same_view_threshold: float = 0.9
    verify_interval: float = 0.4
    max_observation_age: float = 3.0  # an observation older than this is not acted on
    target_tolerance: float = 12.0  # px (at 1680 wide): the re-check must find the sheep this close
    command_verify_delay: float = 1.0
    command_verify_timeout: float = 6.0
    success_confirmations: int = 2  # frames showing the gatherer role
    processed_ttl: float = 180.0  # a commanded villager gets no new command for this long
    assigned_target_ttl: float = 180.0
    assigned_target_similarity: float = 0.85  # surroundings correlation above which a sheep is one already given
    target_context: int = 96  # px (at 1680 wide) square around a sheep used to recognise it
    allow_shared_targets: bool = False  # True: when every visible sheep is taken, use one anyway
    max_assignments: int | None = None  # stop after this many SUCCESS results (None = no limit)
    observe_interval: float = 1.0
    no_target_hold: float = 30.0  # after "no sheep" / "already commanded": wait this long or for the idle set to change
    failure_backoff: float = 10.0
    min_in_game_observations: int = 2
    max_consecutive_failures: int = 3
    debug_dir: str = "captures/villager_assignment"

    @classmethod
    def from_json(cls, path: str | Path) -> "VillagerAssignmentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def select_action(self) -> Action:
        return Action.key(SELECT_IDLE_KEY, self.idle_hotkey, post_delay=self.action_post_delay,
                          dedupe_key=SELECT_IDLE_KEY, cooldown=self.select_cooldown)

    def command_action(self, x: int, y: int) -> Action:
        return Action.click("gather_sheep", x, y, button="right", post_delay=self.action_post_delay,
                            dedupe_key=COMMAND_KEY, cooldown=self.command_cooldown)


# --- Evaluate ---------------------------------------------------------------------
class DecisionKind(str, Enum):
    WAIT = "wait"
    ASSIGN = "assign"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str


@dataclass(frozen=True)
class Hold:
    """Do not select again until the idle indicator changes or ``until`` passes."""

    key: tuple | None
    until: float
    reason: str


def decide(obs: Observation, *, age: float, max_age: float, hold: Hold | None, now: float) -> Decision:
    """What to do given one observation. Pure: no I/O. ``age`` = seconds since the frame was grabbed."""
    W, A = DecisionKind.WAIT, DecisionKind.ASSIGN
    if not obs.game_detected:
        return Decision(W, f"game not detected: {obs.game_reason}")
    if age > max_age:
        return Decision(W, f"observation is {age:.1f}s old (> {max_age:.1f}s): not acting on it")
    idle = obs.idle
    if not idle.present.is_valid:
        return Decision(W, f"idle villager indicator unknown ({idle.present.raw_text})")
    if idle.present.value is False:
        return Decision(W, "no idle villagers")
    if hold is not None and now < hold.until and (hold.key is None or hold.key == indicator_key(obs)):
        return Decision(W, f"holding ({hold.reason}) until the idle villagers change or {hold.until - now:.0f}s pass")
    count = f"{idle.count.value}" if idle.count.is_valid else "unknown number of"
    return Decision(A, f"{count} idle villager(s)")


# --- records ------------------------------------------------------------------------
class ControllerState(str, Enum):
    OBSERVING = "observing"
    SELECTING = "selecting"
    TARGETING = "targeting"
    COMMANDING = "commanding"
    VERIFYING = "verifying"
    BACKOFF = "backoff"
    HALTED = "halted"
    STOPPED = "stopped"
    DONE = "done"


class AssignmentResult(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class Outcome(str, Enum):
    WAITED = "waited"
    ASSIGNED = "assigned"  # command sent and verified (SUCCESS)
    FAILED = "failed"  # command sent, verified not to have taken (FAILURE)
    UNCERTAIN = "uncertain"  # command sent (or maybe sent), outcome UNKNOWN
    SELECTION_FAILED = "selection_failed"  # no villager verified selected: no click
    NO_TARGET = "no_target"  # villager selected, no usable sheep: no click
    TARGET_LOST = "target_lost"  # the re-check before clicking failed: no click
    SKIPPED = "skipped"  # the hotkey landed on a villager already commanded: no click
    DRY_RUN = "dry_run"
    BLOCKED = "blocked"  # executor refused (no game, pending, cooldown, stopped)
    HALTED = "halted"
    STOPPED = "stopped"
    DONE = "done"  # max_assignments reached


@dataclass(frozen=True)
class AssignmentRecord:
    """One right-click on a sheep and what came of it (the action log of this controller)."""

    started: float  # wall time of the click
    villager: dict
    target: dict
    result: AssignmentResult
    detail: str
    idle_before: int | None
    idle_after: int | None

    def to_dict(self) -> dict:
        return {**asdict(self), "result": self.result.value}


@dataclass(frozen=True)
class CycleReport:
    outcome: Outcome
    state: ControllerState
    decision: Decision | None = None
    action: ActionResult | None = None
    detail: str = ""
    villager: VillagerFingerprint | None = None
    target: SheepCandidate | None = None
    assignment: AssignmentRecord | None = None


@dataclass
class Stats:
    cycles: int = 0
    selected: int = 0  # villagers verified selected
    selection_failures: int = 0
    commands: int = 0  # right-clicks sent
    success: int = 0
    failure: int = 0
    unknown: int = 0
    no_target: int = 0
    target_lost: int = 0
    skipped: int = 0
    duplicate_commands: int = 0  # commands to a villager already commanded: must stay 0
    shared_targets: int = 0  # commands to a sheep already given to another villager
    observation_ms: list[float] = field(default_factory=list, repr=False)

    def latency(self) -> dict[str, float]:
        ms = sorted(self.observation_ms)
        if not ms:
            return {"n": 0, "mean_ms": 0.0, "p95_ms": 0.0}
        return {"n": len(ms), "mean_ms": float(np.mean(ms)), "p95_ms": ms[min(len(ms) - 1, max(0, round(0.95 * len(ms)) - 1))]}

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "observation_ms"}
        d["observation_latency"] = self.latency()
        return d


@dataclass
class _Commanded:
    fingerprint: VillagerFingerprint
    at: float
    result: AssignmentResult


@dataclass
class _Taken:
    signature: np.ndarray | None
    at: float
    point: tuple[int, int]


# --- controller --------------------------------------------------------------------
class VillagerAssignmentController:
    def __init__(
        self,
        observer: Observer,
        executor: ActionExecutor,
        config: VillagerAssignmentConfig = VillagerAssignmentConfig(),
        *,
        detector: TargetDetector | None = None,
        recorder: DebugRecorder | None = None,
        tc_locator: Callable[[np.ndarray], tuple[int, int] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self.observer = observer
        self.executor = executor
        self.config = config
        self.detector = detector if detector is not None else SheepDetector()
        self.stop: EmergencyStop = executor.stop
        self.recorder = recorder if recorder is not None else DebugRecorder(config.debug_dir)
        self.tc_locator = tc_locator
        self.clock = clock
        self.wall = wall
        self._sleep = sleep or self.stop.wait
        self.state = ControllerState.OBSERVING
        self.failures: list[FailureRecord] = []
        self.assignments: list[AssignmentRecord] = []
        self.stats = Stats()
        self.consecutive_failures = 0
        self.hold: Hold | None = None
        self._commanded: list[_Commanded] = []
        self._taken: list[_Taken] = []
        self._backoff_until = 0.0
        self._last_obs: Observation | None = None
        self._in_game_streak = 0

    # --- helpers ---------------------------------------------------------------------
    def _observe(self) -> Observation:
        obs = self.observer.observe()
        self._last_obs = obs
        self._in_game_streak = self._in_game_streak + 1 if obs.game_detected else 0
        total = obs.timings.get("total") if obs.timings else None
        if total is not None:
            self.stats.observation_ms.append(total * 1000)
        return obs

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)

    def _ui(self, obs: Observation | None) -> float:
        """Pixel scale of the frame relative to the 1680-wide reference."""
        if obs is not None and obs.frame is not None:
            return obs.frame.image.shape[1] / REFERENCE_SIZE[0]
        return 1.0

    def _fail(self, phase: str, reason: str, obs: Observation | None) -> None:
        shot = self.recorder.capture(obs or self._last_obs, f"{phase}_{reason[:40]}")
        self.failures.append(FailureRecord(time.time(), phase, reason, shot))
        self.consecutive_failures += 1
        self._backoff_until = self.clock() + self.config.failure_backoff
        log.error("FAILURE [%s] %s%s", phase, reason, f" (screenshot {shot})" if shot else "")
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.state = ControllerState.HALTED
            log.error("halting after %d consecutive failures", self.consecutive_failures)
        else:
            self.state = ControllerState.BACKOFF

    def _hold(self, obs: Observation, reason: str) -> None:
        self.hold = Hold(indicator_key(obs), self.clock() + self.config.no_target_hold, reason)

    def _prune(self) -> None:
        now = self.clock()
        self._commanded = [c for c in self._commanded if now - c.at < self.config.processed_ttl]
        self._taken = [t for t in self._taken if now - t.at < self.config.assigned_target_ttl]

    def already_commanded(self, fp: VillagerFingerprint) -> _Commanded | None:
        """A villager sent a command recently that ``fp`` may be. "Can't tell" counts as yes."""
        for c in self._commanded:
            if fp.same_as(c.fingerprint, self.config.same_view_threshold) is not False:
                return c
        return None

    def _context(self, obs: Observation, x: int, y: int) -> np.ndarray | None:
        if obs.frame is None:
            return None
        half = max(8, round(self.config.target_context * self._ui(obs) / 2))
        img = obs.frame.image
        h, w = img.shape[:2]
        crop = img[max(0, y - half):min(h, y + half), max(0, x - half):min(w, x + half)]
        if crop.shape[0] < 2 * half or crop.shape[1] < 2 * half:
            return None  # cut by the screen edge: not comparable
        return view_signature(crop)

    def _is_taken(self, obs: Observation, s: SheepCandidate) -> bool | None:
        """True: a sheep already given to a villager; False: not; None: cannot tell."""
        if not self._taken:
            return False
        sig = self._context(obs, s.x, s.y)
        if sig is None:
            return None
        for t in self._taken:
            sim = signature_similarity(sig, t.signature)
            if sim is not None and sim >= self.config.assigned_target_similarity:
                return True
        return False

    def reference_point(self, obs: Observation) -> tuple[float, float]:
        """Where a sheep should preferably be: the Town Center if located, else the screen centre."""
        if self.tc_locator is not None and obs.frame is not None:
            try:
                p = self.tc_locator(obs.frame.image)
            except Exception as exc:  # a broken locator only loses the preference
                log.warning("town center locator failed: %r", exc)
                p = None
            if p is not None:
                return float(p[0]), float(p[1])
        if obs.frame is not None:
            h, w = obs.frame.image.shape[:2]
            return w / 2, h * 0.45  # centre of the world area (the bottom panel is taller than the top bar)
        return REFERENCE_SIZE[0] / 2, REFERENCE_SIZE[1] * 0.45

    def choose_target(self, obs: Observation, scan: SheepScan) -> tuple[SheepCandidate | None, str, bool]:
        """(target, why, shared). Free sheep first, then the closest to the reference point."""
        if not scan.is_valid:
            return None, f"sheep detection unavailable: {scan.reason}", False
        if not scan.sheep:
            rej = f"; rejected: {', '.join(f'{c.verdict.value}@({c.x},{c.y})' for c in scan.rejected)}" if scan.rejected else ""
            return None, f"no sheep visible ({scan.reason}){rej}", False
        rx, ry = self.reference_point(obs)
        free, taken, unsure = [], [], []
        for s in scan.sheep:
            t = self._is_taken(obs, s)
            (taken if t else unsure if t is None else free).append(s)
        if free:
            best = min(free, key=lambda s: s.distance_to(rx, ry))
            return best, f"sheep at ({best.x}, {best.y}) score {best.score:.2f}, {len(free)} free of {len(scan.sheep)}", False
        if self.config.allow_shared_targets and taken:
            best = min(taken, key=lambda s: s.distance_to(rx, ry))
            return best, f"every visible sheep is taken; sharing the one at ({best.x}, {best.y})", True
        return None, f"{len(scan.sheep)} sheep visible, all already assigned or unrecognisable ({len(taken)} taken, {len(unsure)} unsure)", False

    # --- main cycle -------------------------------------------------------------------
    def cycle(self) -> CycleReport:
        """One villager: Observe -> ... -> Verify. Never raises for game-side problems."""
        self.stats.cycles += 1
        if self.stop.triggered:
            self.state = ControllerState.STOPPED
            return CycleReport(Outcome.STOPPED, self.state, detail=self.stop.reason)
        if self.state is ControllerState.HALTED:
            return CycleReport(Outcome.HALTED, self.state, detail="halted after repeated failures")
        cap = self.config.max_assignments
        if cap is not None and self.stats.success >= cap:
            self.state = ControllerState.DONE
            return CycleReport(Outcome.DONE, self.state, detail=f"{self.stats.success} assignment(s) done (limit {cap})")

        obs = self._observe()
        need = self.config.min_in_game_observations
        if 0 < self._in_game_streak < need:
            return CycleReport(Outcome.WAITED, self.state, detail=f"game just detected ({self._in_game_streak}/{need} observations), letting it settle")
        self._prune()
        now = self.clock()
        if self.hold is not None and (now >= self.hold.until or (indicator_key(obs) is not None and indicator_key(obs) != self.hold.key)):
            log.info("hold over (%s)", self.hold.reason)
            self.hold = None
        decision = decide(obs, age=self.wall() - obs.timestamp, max_age=self.config.max_observation_age, hold=self.hold, now=now)
        log.debug("decision %s: %s", decision.kind.value, decision.reason)

        if self.state is ControllerState.BACKOFF:
            if now < self._backoff_until:
                return CycleReport(Outcome.WAITED, self.state, decision, detail=f"in back-off after a failure ({decision.reason})")
            self.state = ControllerState.OBSERVING
        if decision.kind is DecisionKind.WAIT:
            return CycleReport(Outcome.WAITED, self.state, decision, detail=decision.reason)
        return self._assign(decision, obs)

    def _assign(self, decision: Decision, before: Observation) -> CycleReport:
        cfg = self.config
        # 1. Select one idle villager (game hotkey).
        self.state = ControllerState.SELECTING
        result = self.executor.execute(cfg.select_action(), await_verification=True)
        if result.blocked:
            self.state = ControllerState.OBSERVING
            return CycleReport(Outcome.BLOCKED, self.state, decision, result, result.detail)
        if result.status is ActionStatus.DRY_RUN:
            self.state = ControllerState.OBSERVING
            return self._dry_run(decision, before, result)
        if result.status is ActionStatus.FAILED:
            self.executor.resolve(SELECT_IDLE_KEY, "input error")
            self.stats.selection_failures += 1
            self._fail("select", f"idle hotkey input error: {result.detail}", before)
            return CycleReport(Outcome.SELECTION_FAILED, self.state, decision, result, result.detail)

        verdict, detail, selected = self._verify_selection(before)
        self.executor.resolve(SELECT_IDLE_KEY, verdict)
        if self.stop.triggered:
            self.state = ControllerState.STOPPED
            return CycleReport(Outcome.STOPPED, self.state, decision, result, f"stopped after selecting: {self.stop.reason}")
        if verdict != "selected" or selected is None:
            self.stats.selection_failures += 1
            self._fail("verify_selection", detail, selected)
            return CycleReport(Outcome.SELECTION_FAILED, self.state, decision, result, detail)
        fp = selected.villager.fingerprint
        self.stats.selected += 1
        log.info("villager selected: %s", detail)

        # 2. Never command the same villager twice.
        prior = self.already_commanded(fp)
        if prior is not None:
            self.stats.skipped += 1
            self.state = ControllerState.OBSERVING
            why = f"selected villager ({fp.role}) was already commanded ({prior.result.value}); no new command"
            self._hold(selected, "the idle hotkey came back to a villager already commanded")
            log.warning(why)
            return CycleReport(Outcome.SKIPPED, self.state, decision, result, why, fp)

        # 3. Find a sheep in the frame where the villager was verified (the camera is on it now).
        self.state = ControllerState.TARGETING
        scan = self.detector.detect(selected.frame.image if selected.frame else None, selected.timestamp)
        target, why, shared = self.choose_target(selected, scan)
        if target is None:
            self.stats.no_target += 1
            self.state = ControllerState.OBSERVING
            self._hold(selected, "no usable sheep")
            log.info("no target: %s", why)
            self.recorder.capture(selected, f"no_target_{why[:40]}")
            return CycleReport(Outcome.NO_TARGET, self.state, decision, result, why, fp)

        # 4. Re-check on a fresh frame, right before the click.
        if self.stop.triggered:
            self.state = ControllerState.STOPPED
            return CycleReport(Outcome.STOPPED, self.state, decision, result, f"stopped before the click: {self.stop.reason}", fp, target)
        fresh = self._observe()
        problem, target = self._recheck(fresh, fp, target)
        if problem:
            self.stats.target_lost += 1
            self.state = ControllerState.OBSERVING
            log.warning("not clicking: %s", problem)
            self.recorder.capture(fresh, f"target_lost_{problem[:40]}")
            return CycleReport(Outcome.TARGET_LOST, self.state, decision, result, problem, fp, target)

        # 5. Right-click the sheep.
        self.state = ControllerState.COMMANDING
        context = self._context(fresh, target.x, target.y)
        click_at = self.wall()
        action = self.executor.execute(cfg.command_action(target.x, target.y), await_verification=True)
        log.info("command %s: villager %s -> sheep at (%d, %d) [score %.2f, template %s]%s", action.status.value,
                 fp.to_dict(), target.x, target.y, target.score, target.template, " (shared)" if shared else "")
        if action.blocked:
            self.state = ControllerState.OBSERVING
            return CycleReport(Outcome.BLOCKED, self.state, decision, action, action.detail, fp, target)
        # From here the game may have received the command: remember the villager and the sheep.
        self.stats.commands += 1
        if self.already_commanded(fp) is not None:  # cannot happen (checked above); counted to prove it
            self.stats.duplicate_commands += 1
        if shared:
            self.stats.shared_targets += 1
        self._taken.append(_Taken(context, self.clock(), target.point))
        idle_before = before.idle.count.value if before.idle.count.is_valid else None

        if action.status is ActionStatus.FAILED:
            self.executor.resolve(COMMAND_KEY, "input error")
            res, detail, after = AssignmentResult.UNKNOWN, f"right-click input error: {action.detail}", fresh
        else:
            self.state = ControllerState.VERIFYING
            res, detail, after = self._verify_command(before, fp)
            self.executor.resolve(COMMAND_KEY, res.value)
        self._commanded.append(_Commanded(fp, self.clock(), res))
        idle_after = after.idle.count.value if after is not None and after.idle.count.is_valid else None
        record = AssignmentRecord(click_at, fp.to_dict(), target.to_dict(), res, detail, idle_before, idle_after)
        self.assignments.append(record)
        log.info("assignment %s: %s", res.value.upper(), detail)

        if res is AssignmentResult.SUCCESS:
            self.stats.success += 1
            self.consecutive_failures = 0
            self.state = ControllerState.OBSERVING
            return CycleReport(Outcome.ASSIGNED, self.state, decision, action, detail, fp, target, record)
        if res is AssignmentResult.FAILURE:
            self.stats.failure += 1
            self._fail("verify_command", detail, after)
            return CycleReport(Outcome.FAILED, self.state, decision, action, detail, fp, target, record)
        self.stats.unknown += 1
        if self.stop.triggered:
            self.state = ControllerState.STOPPED
            return CycleReport(Outcome.STOPPED, self.state, decision, action, detail, fp, target, record)
        self._fail("verify_command", f"UNKNOWN: {detail}", after)  # back off: no blind retry
        return CycleReport(Outcome.UNCERTAIN, self.state, decision, action, detail, fp, target, record)

    def _dry_run(self, decision: Decision, obs: Observation, result: ActionResult) -> CycleReport:
        """Nothing was pressed, so nothing is selected and the camera did not move: report what
        the detector sees *now*, as a preview of the targeting step."""
        scan = self.detector.detect(obs.frame.image if obs.frame else None, obs.timestamp)
        target, why, _ = self.choose_target(obs, scan)
        msg = f"would select an idle villager ({decision.reason}); in the current view: {why}"
        if target is not None:
            msg += f"; would right-click ({target.x}, {target.y}) once a villager is selected (the camera moves to it first)"
        log.info("DRY RUN: %s", msg)
        return CycleReport(Outcome.DRY_RUN, self.state, decision, result, msg, None, target)

    def _recheck(self, fresh: Observation, fp: VillagerFingerprint, target: SheepCandidate) -> tuple[str, SheepCandidate]:
        """("", updated target) when it is still safe to click, else (why not, target)."""
        if not fresh.game_detected:
            return f"game not detected before the click ({fresh.game_reason})", target
        age = self.wall() - fresh.timestamp
        if age > self.config.max_observation_age:
            return f"re-check observation is {age:.1f}s old", target
        sel = fresh.villager
        if not sel.is_villager or sel.fingerprint is None:
            return f"villager no longer seen selected ({sel.selected.raw_text})", target
        if sel.fingerprint.same_as(fp, self.config.same_view_threshold) is not True:
            return f"a different (or unrecognisable) villager is selected now ({sel.fingerprint.role})", target
        scan = self.detector.detect(fresh.frame.image if fresh.frame else None, fresh.timestamp)
        if not scan.is_valid:
            return f"sheep detection unavailable on the re-check: {scan.reason}", target
        tol = self.config.target_tolerance * self._ui(fresh)
        again = scan.near(target.x, target.y, tol)
        if again is None:
            return f"sheep at ({target.x}, {target.y}) not detected again within {tol:.0f}px ({scan.reason})", target
        return "", again

    def _verify_selection(self, before: Observation) -> tuple[str, str, Observation | None]:
        """Poll until a villager is seen selected. Returns ("selected" | "not_selected" | "uncertain", detail, last obs)."""
        cfg = self.config
        prev = before.villager.fingerprint if before.villager.is_villager else None
        several = before.idle.count.is_valid and (before.idle.count.value or 0) > 1
        self._wait(cfg.select_verify_delay)
        deadline = self.clock() + cfg.select_verify_timeout
        others, last, why = 0, None, "no observation"
        while True:
            if self.stop.triggered:
                return "uncertain", f"stopped during selection: {self.stop.reason}", last
            last = self._observe()
            sel = last.villager
            if not last.game_detected:
                why = f"game not detected ({last.game_reason})"
            elif sel.is_villager and sel.fingerprint is not None:
                f = sel.fingerprint
                if prev is not None and several and f.same_as(prev, cfg.same_view_threshold) is True:
                    why = f"still the same villager ({f.role})"
                else:
                    hp = f" {f.hp}/{f.hp_max}" if f.hp is not None else ""
                    return "selected", f"{f.role}{hp} ({sel.selected.raw_text!r}, conf {sel.selected.confidence:.2f})", last
            elif sel.is_other:
                others += 1
                why = f"selected object is not a villager ({sel.selected.raw_text!r}, {others}/{cfg.other_confirmations})"
                if others >= cfg.other_confirmations:
                    return "not_selected", f"after '{cfg.idle_hotkey}': {why}", last
            else:
                why = f"selection unknown ({sel.selected.raw_text})"
            if self.clock() >= deadline:
                return "uncertain", f"selection not verified after '{cfg.idle_hotkey}': {why}", last
            self._wait(cfg.verify_interval)

    def _verify_command(self, before: Observation, fp: VillagerFingerprint) -> tuple[AssignmentResult, str, Observation | None]:
        """Watch the selection panel and the idle count after the right-click."""
        cfg = self.config
        idle_before = before.idle.count.value if before.idle.count.is_valid else None
        was_gatherer = fp.role in GATHERER_ROLES
        self._wait(cfg.command_verify_delay)
        deadline = self.clock() + cfg.command_verify_timeout
        gatherer_frames = 0
        unchanged_all = True  # every frame: same villager, old role, readable idle count not lower
        frames = 0
        last, why = None, "no observation"
        while True:
            if self.stop.triggered:
                return AssignmentResult.UNKNOWN, f"stopped during verification: {self.stop.reason}", last
            last = self._observe()
            frames += 1
            idle = last.idle
            idle_now = idle.count.value if idle.count.is_valid else (0 if idle.none_idle else None)
            sel = last.villager
            if not last.game_detected:
                unchanged_all = False
                why = f"game not detected ({last.game_reason})"
            elif sel.is_villager and sel.fingerprint is not None:
                f = sel.fingerprint
                same_hp = fp.hp is None or f.hp is None or (f.hp, f.hp_max) == (fp.hp, fp.hp_max)
                idle_up = idle_before is not None and idle_now is not None and idle_now > idle_before
                if f.role in GATHERER_ROLES and not was_gatherer and same_hp and not idle_up:
                    gatherer_frames += 1
                    why = f"selected villager is now a {f.role} (idle {idle_before} -> {idle_now})"
                    if gatherer_frames >= cfg.success_confirmations:
                        return AssignmentResult.SUCCESS, why, last
                else:
                    gatherer_frames = 0
                    still_idle_role = f.role == fp.role and same_hp
                    no_drop = idle_before is not None and idle_now is not None and idle_now >= idle_before
                    if not (still_idle_role and no_drop):
                        unchanged_all = False
                    if was_gatherer:
                        why = f"villager was already a {fp.role}: the role cannot show the new task (idle {idle_before} -> {idle_now})"
                    elif idle_before is not None and idle_now is not None and idle_now < idle_before:
                        why = f"idle count {idle_before} -> {idle_now} but the selected villager still reads {f.role}"
                    else:
                        why = f"selected villager still reads {f.role} (idle {idle_before} -> {idle_now})"
            else:
                unchanged_all = False
                gatherer_frames = 0
                why = f"selection unknown or changed ({sel.selected.raw_text})"
            if self.clock() >= deadline:
                if unchanged_all and frames >= 2 and not was_gatherer:
                    return AssignmentResult.FAILURE, f"command did not take: {why} for {frames} observations", last
                return AssignmentResult.UNKNOWN, f"could not confirm the assignment: {why}", last
            self._wait(cfg.verify_interval)

    # --- loop -------------------------------------------------------------------------
    def run(self, max_cycles: int | None = None) -> Stats:
        """Repeat cycles until emergency stop, halt, the assignment limit, or ``max_cycles``."""
        log.info("villager assignment controller started (dry_run=%s, hotkey %r, max_assignments %s)",
                 self.executor.dry_run, self.config.idle_hotkey, self.config.max_assignments)
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
                    report = CycleReport(Outcome.UNCERTAIN, self.state, detail=repr(exc))
            log.info("cycle %d: %s [%s] %s", self.stats.cycles, report.outcome.value, report.state.value, report.detail)
            if report.outcome in (Outcome.STOPPED, Outcome.HALTED, Outcome.DONE) or self.stop.triggered:
                self.state = ControllerState.STOPPED if self.stop.triggered else self.state
                break
            self._wait(self.config.observe_interval)
        log.info("villager assignment controller finished: %s", json.dumps(self.stats.summary()))
        return self.stats


# --- CLI ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    from aoe2_ai.perception.profile import IDLE_VILLAGER

    from .runtime import add_common_args, run_live, setup_logging

    parser = argparse.ArgumentParser(description="Send idle villagers, one at a time, to gather from visible sheep.")
    parser.add_argument("--config", type=Path, help="JSON file with VillagerAssignmentConfig fields")
    parser.add_argument("--idle-key", help="the game's 'next idle villager' hotkey (default .)")
    parser.add_argument("--max-assignments", type=int, default=1,
                        help="stop after this many verified assignments (default 1; 0 = no limit)")
    parser.add_argument("--assignments-log", type=Path, help="write every right-click and its result to this JSON file at the end")
    add_common_args(parser)
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    config = VillagerAssignmentConfig.from_json(args.config) if args.config else VillagerAssignmentConfig()
    overrides: dict = {"max_assignments": args.max_assignments or None}
    if args.idle_key:
        overrides["idle_hotkey"] = args.idle_key
    config = VillagerAssignmentConfig(**{**asdict(config), **overrides})
    holder: dict = {}

    def make(observer, executor):
        holder["c"] = VillagerAssignmentController(observer, executor, config)
        return holder["c"]

    run_live(args, make, config.debug_dir, IDLE_VILLAGER)
    ctrl = holder.get("c")
    if ctrl is not None:
        print(f"Summary: {json.dumps(ctrl.stats.summary())}")
        if args.assignments_log:
            args.assignments_log.parent.mkdir(parents=True, exist_ok=True)
            args.assignments_log.write_text(json.dumps([a.to_dict() for a in ctrl.assignments], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
