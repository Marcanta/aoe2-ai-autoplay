"""VillagerAssignmentController: decision rules (unit) and the closed loop with scripted
observations, a scripted sheep detector and mocked input (no screen, no game)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aoe2_ai.controllers.idle_villager import SELECT_IDLE_KEY
from aoe2_ai.controllers.town_center import DebugRecorder
from aoe2_ai.controllers.villager_assignment import (
    COMMAND_KEY,
    AssignmentResult,
    ControllerState,
    DecisionKind,
    Hold,
    Outcome,
    VillagerAssignmentConfig,
    VillagerAssignmentController,
    decide,
)
from aoe2_ai.execution import Action, ActionExecutor, CallableGuard, EmergencyStop, GuardResult, StaticGuard
from aoe2_ai.perception.capture import Frame
from aoe2_ai.perception.game_state import Status
from aoe2_ai.perception.object_detector import SheepCandidate, SheepScan, Verdict

from conftest import FRAMES, requires_tesseract
from controller_helpers import FakeClock, MockInput, ScriptedObserver, fp, make_idle_obs

CFG = VillagerAssignmentConfig(debug_dir="", min_in_game_observations=1)
A, B = fp(1), fp(2)
A_SHEPHERD = fp(1, role="berger")
B_SHEPHERD = fp(2, role="berger")
TEXTURE = np.random.default_rng(7).integers(0, 255, (1050, 1680, 3), dtype=np.uint8)  # a "world" to recognise sheep by


def obs(*, frame=True, **kwargs):
    o = make_idle_obs(**kwargs)
    return replace(o, frame=Frame(TEXTURE, 0.0, "test") if frame else None)


def sheep(x, y, score=0.9):
    return SheepCandidate(x, y, (x - 25, y - 20, 50, 40), score, None, "water_buffalo_a", 1.0, Verdict.ACCEPTED)


def scan(*sheep_list, status=Status.VALID, rejected=()):
    return SheepScan(0.0, status, f"{len(sheep_list)} sheep", tuple(sheep_list), tuple(rejected))


class FakeDetector:
    """Returns the scripted scans in order, then repeats the last one."""

    def __init__(self, *scans, on_detect=None):
        self.scans = list(scans)
        self.calls = 0
        self.on_detect = on_detect

    def detect(self, frame, timestamp=None):
        self.calls += 1
        if self.on_detect:
            self.on_detect(self.calls)
        return self.scans.pop(0) if len(self.scans) > 1 else self.scans[0]


def make(*observations, scans=(scan(sheep(840, 470)),), backend=None, guard=None, dry_run=False, config=CFG, on_detect=None):
    clock = FakeClock()
    backend = MockInput() if backend is None and not dry_run else backend
    stop = EmergencyStop()
    executor = ActionExecutor(backend, guard or StaticGuard(True, "test"), dry_run=dry_run, stop=stop, clock=clock, sleep=clock.sleep)
    detector = FakeDetector(*scans, on_detect=on_detect)
    ctrl = VillagerAssignmentController(ScriptedObserver(*observations), executor, config, detector=detector,
                                        recorder=DebugRecorder(None), clock=clock, wall=lambda: 0.0, sleep=clock.sleep)
    return ctrl, backend, detector, clock


def clicks(inp):
    return [e for e in inp.events if e[0] == "click"]


def one_villager_success_script(count=1, villager=A, shepherd=A_SHEPHERD):
    return (
        obs(count=count),  # before: idle, nothing selected
        obs(count=count, villager=villager),  # after '.': villager selected
        obs(count=count, villager=villager),  # re-check before the click
        obs(present=count > 1, count=count - 1 if count > 1 else 0, villager=shepherd),  # after the click
        obs(present=count > 1, count=count - 1 if count > 1 else 0, villager=shepherd),
    )


# --- decide(): unit tests ---------------------------------------------------------
def D(o, age=0.0, hold=None, now=0.0):
    return decide(o, age=age, max_age=3.0, hold=hold, now=now)


def test_zero_idle_villagers_waits():
    d = D(obs(present=False))
    assert d.kind is DecisionKind.WAIT and "no idle" in d.reason


def test_one_idle_villager_assigns():
    assert D(obs(count=1)).kind is DecisionKind.ASSIGN


def test_unknown_idle_indicator_waits_not_zero():
    d = D(obs(present=None))
    assert d.kind is DecisionKind.WAIT and "unknown" in d.reason


def test_unknown_count_with_yellow_button_still_assigns_one():
    d = D(obs(count=None))
    assert d.kind is DecisionKind.ASSIGN and "unknown number" in d.reason


def test_stale_observation_waits():
    d = D(obs(count=2), age=5.0)
    assert d.kind is DecisionKind.WAIT and "old" in d.reason


def test_game_not_detected_waits():
    assert D(obs(game=False)).kind is DecisionKind.WAIT


def test_hold_waits_until_idle_set_changes_or_timeout():
    o = obs(count=2)
    hold = Hold((True, 2), until=100.0, reason="no usable sheep")
    assert D(o, hold=hold, now=50.0).kind is DecisionKind.WAIT
    assert D(obs(count=3), hold=hold, now=50.0).kind is DecisionKind.ASSIGN  # idle set changed
    assert D(o, hold=hold, now=101.0).kind is DecisionKind.ASSIGN  # hold expired


# --- idle villager handling ---------------------------------------------------------
def test_zero_idle_villagers_sends_nothing():
    ctrl, inp, det, _ = make(obs(present=False))
    for _ in range(3):
        assert ctrl.cycle().outcome is Outcome.WAITED
    assert inp.events == [] and det.calls == 0


def test_unknown_idle_count_sends_nothing():
    ctrl, inp, _, _ = make(obs(present=None))
    for _ in range(3):
        assert ctrl.cycle().outcome is Outcome.WAITED
    assert inp.events == []


def test_one_idle_villager_assigned_and_verified():
    ctrl, inp, det, _ = make(*one_villager_success_script())
    r = ctrl.cycle()
    assert r.outcome is Outcome.ASSIGNED, r.detail
    assert inp.events == [("press", "."), ("click", 840, 470, "right")]
    rec = r.assignment
    assert rec.result is AssignmentResult.SUCCESS and rec.target["x"] == 840 and rec.villager["role"] == "villageois"
    assert rec.idle_before == 1 and rec.idle_after == 0
    assert ctrl.stats.success == 1 and ctrl.stats.commands == 1 and ctrl.stats.duplicate_commands == 0
    assert not ctrl.executor.is_pending(COMMAND_KEY) and not ctrl.executor.is_pending(SELECT_IDLE_KEY)
    assert det.calls == 2  # targeting + re-check


def test_multiple_idle_villagers_one_per_cycle_different_sheep():
    first = one_villager_success_script(count=2, villager=A, shepherd=A_SHEPHERD)
    second = (
        obs(count=1), obs(count=1, villager=B), obs(count=1, villager=B),
        obs(present=False, villager=B_SHEPHERD), obs(present=False, villager=B_SHEPHERD),
    )
    two = scan(sheep(840, 470), sheep(1000, 520))
    ctrl, inp, _, clock = make(*first, *second, scans=(two,))
    assert ctrl.cycle().outcome is Outcome.ASSIGNED
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.ASSIGNED
    targets = [e[1:3] for e in clicks(inp)]
    assert targets == [(840, 470), (1000, 520)]  # the second villager is not sent to the first sheep
    assert ctrl.stats.success == 2 and ctrl.stats.shared_targets == 0


def test_all_visible_sheep_taken_stops_safely():
    first = one_villager_success_script(count=2)
    ctrl, inp, _, clock = make(*first, obs(count=1), obs(count=1, villager=B), scans=(scan(sheep(840, 470)),))
    assert ctrl.cycle().outcome is Outcome.ASSIGNED
    clock.now += 2
    r = ctrl.cycle()
    assert r.outcome is Outcome.NO_TARGET and "already assigned" in r.detail
    assert len(clicks(inp)) == 1


def test_shared_targets_when_allowed():
    cfg = replace(CFG, allow_shared_targets=True)
    first = one_villager_success_script(count=2)
    second = (obs(count=1), obs(count=1, villager=B), obs(count=1, villager=B),
              obs(present=False, villager=B_SHEPHERD), obs(present=False, villager=B_SHEPHERD))
    ctrl, inp, _, clock = make(*first, *second, config=cfg)
    ctrl.cycle()
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.ASSIGNED
    assert ctrl.stats.shared_targets == 1


def test_failed_selection_sends_no_click():
    ctrl, inp, det, _ = make(obs(count=1), obs(count=1, villager=False))
    r = ctrl.cycle()
    assert r.outcome is Outcome.SELECTION_FAILED
    assert inp.events == [("press", ".")] and det.calls == 0
    assert ctrl.state is ControllerState.BACKOFF and len(ctrl.failures) == 1


def test_unverified_selection_sends_no_click():
    ctrl, inp, _, _ = make(obs(count=1), obs(count=1, villager=None))  # selection never readable
    assert ctrl.cycle().outcome is Outcome.SELECTION_FAILED and clicks(inp) == []


# --- targets --------------------------------------------------------------------------
def test_no_sheep_visible_no_click_then_holds():
    ctrl, inp, _, clock = make(obs(count=2), obs(count=2, villager=A), scans=(scan(),))
    r = ctrl.cycle()
    assert r.outcome is Outcome.NO_TARGET and "no sheep visible" in r.detail
    clock.now += 2
    r = ctrl.cycle()
    assert r.outcome is Outcome.WAITED and "holding" in r.detail
    assert inp.events == [("press", ".")]


def test_detection_unknown_is_not_no_sheep_but_still_no_click():
    ctrl, inp, _, _ = make(obs(count=1), obs(count=1, villager=A), scans=(scan(status=Status.UNKNOWN),))
    r = ctrl.cycle()
    assert r.outcome is Outcome.NO_TARGET and "unavailable" in r.detail and clicks(inp) == []


def test_ambiguous_detections_are_not_targets():
    amb = SheepCandidate(840, 470, (815, 450, 50, 40), 0.7, 0.68, "a", 1.0, Verdict.AMBIGUOUS)
    ctrl, inp, _, _ = make(obs(count=1), obs(count=1, villager=A), scans=(scan(rejected=(amb,)),))
    r = ctrl.cycle()
    assert r.outcome is Outcome.NO_TARGET and "ambiguous" in r.detail and clicks(inp) == []


def test_prefers_the_sheep_nearest_the_reference_point():
    ctrl, inp, _, _ = make(*one_villager_success_script(), scans=(scan(sheep(200, 200), sheep(850, 480), sheep(1500, 700)),))
    ctrl.cycle()
    assert clicks(inp) == [("click", 850, 480, "right")]


def test_town_center_locator_sets_the_preference():
    ctrl, inp, _, _ = make(*one_villager_success_script(), scans=(scan(sheep(200, 200), sheep(850, 480)),))
    ctrl.tc_locator = lambda image: (210, 190)
    ctrl.cycle()
    assert clicks(inp) == [("click", 200, 200, "right")]


# --- action execution ---------------------------------------------------------------------
def test_target_disappears_before_the_click():
    ctrl, inp, _, _ = make(obs(count=1), obs(count=1, villager=A), obs(count=1, villager=A),
                           scans=(scan(sheep(840, 470)), scan()))
    r = ctrl.cycle()
    assert r.outcome is Outcome.TARGET_LOST and "not detected again" in r.detail
    assert inp.events == [("press", ".")]


def test_target_moved_slightly_clicks_the_new_position():
    ctrl, inp, _, _ = make(*one_villager_success_script(), scans=(scan(sheep(840, 470)), scan(sheep(846, 474))))
    assert ctrl.cycle().outcome is Outcome.ASSIGNED
    assert clicks(inp) == [("click", 846, 474, "right")]


def test_selection_changed_before_the_click():
    ctrl, inp, _, _ = make(obs(count=2), obs(count=2, villager=A), obs(count=2, villager=B))
    r = ctrl.cycle()
    assert r.outcome is Outcome.TARGET_LOST and "different" in r.detail and clicks(inp) == []


def test_game_enters_menu_before_the_click():
    ctrl, inp, _, _ = make(obs(count=1), obs(count=1, villager=A), obs(game=False))
    r = ctrl.cycle()
    assert r.outcome is Outcome.TARGET_LOST and "game not detected" in r.detail and clicks(inp) == []


def test_game_loses_focus_blocks_everything():
    ctrl, inp, _, _ = make(obs(count=1), guard=StaticGuard(False, "foreground is notepad"))
    r = ctrl.cycle()
    assert r.outcome is Outcome.BLOCKED and inp.events == []


def test_focus_lost_between_selection_and_click():
    state = {"ok": True}
    guard = CallableGuard(lambda: GuardResult(state["ok"], "test"))

    def lose_focus(call):
        if call == 2:  # the re-check scan, right before the click
            state["ok"] = False

    ctrl, inp, _, _ = make(*one_villager_success_script(), guard=guard, on_detect=lose_focus)
    r = ctrl.cycle()
    assert r.outcome is Outcome.BLOCKED and inp.events == [("press", ".")]
    assert ctrl.stats.commands == 0


def test_executor_rejects_the_command():
    ctrl, inp, _, _ = make(*one_villager_success_script())
    # An earlier command still awaiting verification: the executor refuses a second one.
    ctrl.executor.execute(Action.click("earlier", 1, 1, button="right", dedupe_key=COMMAND_KEY), await_verification=True)
    r = ctrl.cycle()
    assert r.outcome is Outcome.BLOCKED and "awaiting verification" in r.detail
    assert clicks(inp) == [("click", 1, 1, "right")]


def test_click_input_error_is_unknown_and_never_retried():
    class FailingClicks(MockInput):
        def click(self, x, y, button="left"):
            raise OSError("input broke")

    inp = FailingClicks()
    script = one_villager_success_script(count=2)[:3] + (obs(count=2), obs(count=2, villager=A))
    ctrl, _, _, clock = make(*script, backend=inp)
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and r.assignment.result is AssignmentResult.UNKNOWN
    clock.now += CFG.failure_backoff + 1
    r = ctrl.cycle()
    assert r.outcome is Outcome.SKIPPED  # same villager again: no new command
    assert ctrl.stats.commands == 1


def test_emergency_stop_during_selection_sends_no_click():
    ctrl, inp, _, clock = make(obs(count=1), obs(count=1, villager=A))
    real_sleep = clock.sleep

    def sleep(seconds):
        ctrl.stop.trigger("F12")
        real_sleep(seconds)

    ctrl._sleep = sleep
    r = ctrl.cycle()
    assert r.outcome is Outcome.STOPPED and clicks(inp) == []
    assert ctrl.cycle().outcome is Outcome.STOPPED


def test_emergency_stop_before_the_click():
    ctrl, inp, _, _ = make(*one_villager_success_script(), on_detect=lambda call: ctrl.stop.trigger("F12") if call == 1 else None)
    r = ctrl.cycle()
    assert r.outcome is Outcome.STOPPED and clicks(inp) == []


def test_dry_run_sends_nothing_and_previews_the_target():
    ctrl, _, det, _ = make(obs(count=1), dry_run=True)
    r = ctrl.cycle()
    assert r.outcome is Outcome.DRY_RUN and "would right-click (840, 470)" in r.detail
    assert ctrl.executor.history[-1].status.value == "dry_run"
    assert ctrl.stats.commands == 0


# --- verification -----------------------------------------------------------------------------
def test_failed_assignment_role_and_count_unchanged():
    script = (obs(count=1), obs(count=1, villager=A), obs(count=1, villager=A), obs(count=1, villager=A))
    ctrl, inp, _, _ = make(*script)
    r = ctrl.cycle()
    assert r.outcome is Outcome.FAILED and r.assignment.result is AssignmentResult.FAILURE
    assert ctrl.state is ControllerState.BACKOFF and ctrl.stats.failure == 1


def test_idle_drop_alone_is_unknown():
    # The idle count falls, but the selected villager still reads "villageois": not proof.
    script = (obs(count=2), obs(count=2, villager=A), obs(count=2, villager=A), obs(count=1, villager=A))
    ctrl, _, _, _ = make(*script)
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and "idle count 2 -> 1" in r.detail


def test_selection_lost_after_click_is_unknown():
    script = (obs(count=1), obs(count=1, villager=A), obs(count=1, villager=A), obs(count=1, villager=None))
    ctrl, _, _, _ = make(*script)
    assert ctrl.cycle().outcome is Outcome.UNCERTAIN


def test_villager_already_a_shepherd_is_unknown_not_success():
    shep = fp(1, role="berger")
    script = (obs(count=1), obs(count=1, villager=shep), obs(count=1, villager=shep), obs(present=False, villager=shep))
    ctrl, _, _, _ = make(*script)
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and "already a berger" in r.detail


def test_no_duplicate_command_after_unknown():
    script = (obs(count=2), obs(count=2, villager=A), obs(count=2, villager=A), obs(count=1, villager=A))
    ctrl, inp, _, clock = make(*script)
    assert ctrl.cycle().outcome is Outcome.UNCERTAIN
    for _ in range(3):  # back-off: nothing at all
        clock.now += 2
        assert ctrl.cycle().outcome is Outcome.WAITED
    clock.now += CFG.failure_backoff
    ctrl.observer.script = [obs(count=1), obs(count=1, villager=A)]  # the hotkey lands on A again
    r = ctrl.cycle()
    assert r.outcome is Outcome.SKIPPED and len(clicks(inp)) == 1
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.WAITED  # holds instead of cycling the hotkey
    assert ctrl.stats.duplicate_commands == 0 and ctrl.stats.commands == 1


def test_repeated_failures_halt_the_controller():
    ctrl, inp, _, clock = make(obs(count=1), obs(count=1, villager=False))
    for _ in range(CFG.max_consecutive_failures):
        ctrl.cycle()
        clock.now += CFG.failure_backoff + 1
    assert ctrl.state is ControllerState.HALTED
    n = len(inp.events)
    assert ctrl.cycle().outcome is Outcome.HALTED and len(inp.events) == n


def test_max_assignments_ends_the_run():
    cfg = replace(CFG, max_assignments=1)
    ctrl, inp, _, _ = make(*one_villager_success_script(count=2), config=cfg)
    stats = ctrl.run(max_cycles=10)
    assert stats.success == 1 and len(clicks(inp)) == 1 and ctrl.state is ControllerState.DONE


def test_observation_latency_is_recorded():
    timed = [replace(o, timings={"total": t}) for o, t in zip(one_villager_success_script(), (0.5, 0.6, 0.7, 0.8, 0.9))]
    ctrl, _, _, _ = make(*timed)
    ctrl.cycle()
    lat = ctrl.stats.latency()
    assert lat["n"] >= 4 and 500 <= lat["mean_ms"] <= 900 and lat["p95_ms"] >= lat["mean_ms"]


@pytest.mark.parametrize("bad", ["none", "far"])
def test_recheck_never_clicks_a_moved_or_missing_target(bad):
    second = scan() if bad == "none" else scan(sheep(900, 470))  # 60 px away: a different sheep
    ctrl, inp, _, _ = make(*one_villager_success_script(), scans=(scan(sheep(840, 470)), second))
    assert ctrl.cycle().outcome is Outcome.TARGET_LOST and clicks(inp) == []


# --- end to end on a real frame (real perception + real detector, dry run) -------------------
@requires_tesseract
def test_dry_run_on_a_real_frame_previews_a_real_sheep():
    import cv2

    from aoe2_ai.controllers.observation import LiveObserver
    from aoe2_ai.perception import Perception

    from test_sheep_detector import boxes, inside, place

    # Real HUD (7 idle villagers, one selected) with a real herd pasted where it was captured.
    base = cv2.imread(str(FRAMES / "idle_seven_villager_selected_20261116_2129.png"), cv2.IMREAD_COLOR)
    frame = Frame(place("herd_open", base=base), 1000.0, "test")
    perception = Perception()
    try:
        observation = LiveObserver(StaticGuard(True, "test"), perception=perception, source=object()).observe_frame(frame)
    finally:
        perception.close()
    assert observation.game_detected and observation.idle.has_idle

    clock = FakeClock()
    executor = ActionExecutor(None, StaticGuard(True, "test"), dry_run=True, clock=clock, sleep=clock.sleep)
    ctrl = VillagerAssignmentController(ScriptedObserver(observation), executor, CFG, recorder=DebugRecorder(None),
                                        clock=clock, wall=lambda: 1000.0, sleep=clock.sleep)
    r = ctrl.cycle()
    assert r.outcome is Outcome.DRY_RUN and r.target is not None, r.detail
    assert any(inside(r.target, b) for b in boxes("herd_open").values())
