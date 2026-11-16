"""IdleVillagerController: decision rules (unit) and the closed loop with mocked input (integration)."""

from dataclasses import replace

import numpy as np
import pytest

from aoe2_ai.controllers.idle_villager import (
    SELECT_IDLE_KEY,
    ControllerState,
    DecisionKind,
    IdleVillagerConfig,
    IdleVillagerController,
    Outcome,
    SelectionPass,
    decide,
)
from aoe2_ai.controllers.town_center import DebugRecorder
from aoe2_ai.execution import ActionExecutor, ActionStatus, EmergencyStop, StaticGuard
from aoe2_ai.perception.capture import Frame

from controller_helpers import FakeClock, MockInput, ScriptedObserver, fp, make_idle_obs

CFG = IdleVillagerConfig(debug_dir="", min_in_game_observations=1)  # settling tested separately
A, B, C = fp(1), fp(2), fp(3)


def make_controller(*observations, backend=None, guard=True, dry_run=False, config=CFG, recorder=None):
    clock = FakeClock()
    backend = MockInput() if backend is None and not dry_run else backend
    executor = ActionExecutor(backend, StaticGuard(guard, "test"), dry_run=dry_run, stop=EmergencyStop(), clock=clock, sleep=clock.sleep)
    observer = ScriptedObserver(*observations)
    ctrl = IdleVillagerController(observer, executor, config, recorder=recorder or DebugRecorder(None), clock=clock, sleep=clock.sleep)
    return ctrl, backend, observer, clock


def presses(inp):
    return inp.presses(".")


# --- decide(): unit tests ------------------------------------------------------
def test_idle_villagers_present_selects():
    assert decide(make_idle_obs(present=True, count=2), None).kind is DecisionKind.SELECT_IDLE_VILLAGER


def test_no_idle_villagers_waits():
    d = decide(make_idle_obs(present=False), None)
    assert d.kind is DecisionKind.WAIT and "no idle" in d.reason


@pytest.mark.parametrize(
    "kwargs, text",
    [
        (dict(present=None), "indicator unknown"),  # unreadable button is neither yes nor no
        (dict(game=False), "game not detected"),
    ],
)
def test_unknown_or_no_game_waits(kwargs, text):
    d = decide(make_idle_obs(**kwargs), None)
    assert d.kind is DecisionKind.WAIT and text in d.reason


def test_completed_pass_waits():
    done = SelectionPass((True, 1), 1, [A])
    d = decide(make_idle_obs(count=1), done)
    assert d.kind is DecisionKind.WAIT and "all visited" in d.reason


# --- closed loop -------------------------------------------------------------------
def test_no_idle_villagers_sends_nothing():
    ctrl, inp, _, _ = make_controller(make_idle_obs(present=False))
    for _ in range(3):
        assert ctrl.cycle().outcome is Outcome.WAITED
    assert inp.events == []


def test_one_idle_villager_selected_verified_then_not_reselected():
    ctrl, inp, _, _ = make_controller(make_idle_obs(count=1), make_idle_obs(count=1, villager=A))
    r = ctrl.cycle()
    assert r.outcome is Outcome.SELECTED, r.detail
    assert r.villager == A and ctrl.last_selected == A
    assert inp.events == [("press", ".")]
    assert not ctrl.executor.is_pending(SELECT_IDLE_KEY)
    for _ in range(3):  # still one idle villager, already visited: no more presses
        r = ctrl.cycle()
        assert r.outcome is Outcome.WAITED and "all visited" in r.detail
    assert presses(inp) == 1 and ctrl.stats.selected == 1


def test_multiple_idle_villagers_each_selected_once():
    ctrl, inp, _, clock = make_controller(
        make_idle_obs(count=3),
        make_idle_obs(count=3, villager=A),
        make_idle_obs(count=3, villager=A), make_idle_obs(count=3, villager=B),
        make_idle_obs(count=3, villager=B), make_idle_obs(count=3, villager=C),
        make_idle_obs(count=3, villager=C),
    )
    outcomes, selected = [], []
    for _ in range(5):
        r = ctrl.cycle()
        outcomes.append(r.outcome)
        selected.append(r.villager)
        clock.now += 2
    assert outcomes == [Outcome.SELECTED] * 3 + [Outcome.WAITED] * 2
    assert selected[:3] == [A, B, C]
    assert presses(inp) == 3 and ctrl.stats.selected == 3 and ctrl.stats.passes == 1


def test_hotkey_coming_back_to_a_visited_villager_ends_the_pass():
    # The button says 3, but the game cycles back to A after B (e.g. one got a task meanwhile).
    ctrl, inp, _, clock = make_controller(
        make_idle_obs(count=3), make_idle_obs(count=3, villager=A),
        make_idle_obs(count=3, villager=A), make_idle_obs(count=3, villager=B),
        make_idle_obs(count=3, villager=B), make_idle_obs(count=3, villager=fp(1)),  # A again
        make_idle_obs(count=3, villager=A),
    )
    outcomes = []
    for _ in range(5):
        outcomes.append(ctrl.cycle().outcome)
        clock.now += 2
    assert outcomes == [Outcome.SELECTED, Outcome.SELECTED, Outcome.PASS_COMPLETE, Outcome.WAITED, Outcome.WAITED]
    assert presses(inp) == 3 and ctrl.stats.selected == 2


def test_same_role_and_hp_without_view_counts_as_visited():
    # Can't tell two "Villageois 25/25" apart without the view: stop rather than risk cycling.
    a, b = fp(1, with_view=False), fp(2, with_view=False)
    ctrl, inp, _, clock = make_controller(
        make_idle_obs(count=2), make_idle_obs(count=2, villager=a),
        make_idle_obs(count=2, villager=a), make_idle_obs(count=2, villager=b),
    )
    assert ctrl.cycle().outcome is Outcome.SELECTED
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.PASS_COMPLETE
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.WAITED and presses(inp) == 2


def test_different_role_is_a_different_villager_even_without_view():
    a, b = fp(1, with_view=False), fp(1, role="bucheron", with_view=False)
    ctrl, _, _, clock = make_controller(
        make_idle_obs(count=2), make_idle_obs(count=2, villager=a),
        make_idle_obs(count=2, villager=a), make_idle_obs(count=2, villager=b),
    )
    assert ctrl.cycle().outcome is Outcome.SELECTED
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.SELECTED


def test_new_pass_when_idle_count_changes():
    ctrl, inp, observer, clock = make_controller(make_idle_obs(count=1), make_idle_obs(count=1, villager=A), make_idle_obs(count=1, villager=A))
    assert ctrl.cycle().outcome is Outcome.SELECTED
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.WAITED
    observer.script = [make_idle_obs(count=2, villager=A), make_idle_obs(count=2, villager=B)]  # a second villager went idle
    clock.now += 2
    r = ctrl.cycle()
    assert r.outcome is Outcome.SELECTED and r.villager == B
    assert ctrl.stats.passes == 2 and presses(inp) == 2


def test_unknown_count_selects_one_per_pass():
    ctrl, inp, _, clock = make_controller(make_idle_obs(count=None), make_idle_obs(count=None, villager=A), make_idle_obs(count=None, villager=A))
    assert ctrl.cycle().outcome is Outcome.SELECTED
    clock.now += 2
    assert ctrl.cycle().outcome is Outcome.WAITED and presses(inp) == 1


def test_failed_selection_records_failure_backs_off_and_halts(tmp_path):
    frame = Frame(np.zeros((10, 10, 3), np.uint8), 0.0, "test")
    other = replace(make_idle_obs(count=2, villager=False), frame=frame)  # e.g. the Town Center stays selected
    config = replace(CFG, max_consecutive_failures=2)
    ctrl, inp, _, clock = make_controller(make_idle_obs(count=2), other, recorder=DebugRecorder(tmp_path), config=config)
    r = ctrl.cycle()
    assert r.outcome is Outcome.FAILED and "not a villager" in r.detail
    assert ctrl.state is ControllerState.BACKOFF
    assert ctrl.failures[0].phase == "verify_selection" and ctrl.failures[0].screenshot.endswith(".png")
    assert not ctrl.executor.is_pending(SELECT_IDLE_KEY)
    # No blind retry during the back-off.
    assert ctrl.cycle().outcome is Outcome.WAITED and presses(inp) == 1
    clock.now += CFG.failure_backoff
    assert ctrl.cycle().outcome is Outcome.FAILED
    assert ctrl.state is ControllerState.HALTED
    assert ctrl.cycle().outcome is Outcome.HALTED and presses(inp) == 2


def test_selection_that_does_not_move_on_is_a_failure():
    # Two idle villagers, A already selected; after the press A is still selected.
    ctrl, inp, _, _ = make_controller(make_idle_obs(count=2, villager=A), make_idle_obs(count=2, villager=A))
    r = ctrl.cycle()
    assert r.outcome is Outcome.FAILED and "same villager" in r.detail
    assert ctrl.stats.selected == 0 and presses(inp) == 1


def test_single_idle_villager_already_selected_is_accepted():
    # Only one idle villager and it is the one already selected: the press keeps it selected.
    ctrl, _, _, _ = make_controller(make_idle_obs(count=1, villager=A), make_idle_obs(count=1, villager=A))
    assert ctrl.cycle().outcome is Outcome.SELECTED


def test_unknown_hud_state_never_presses():
    ctrl, inp, _, _ = make_controller(make_idle_obs(present=None))
    for _ in range(3):
        r = ctrl.cycle()
        assert r.outcome is Outcome.WAITED and "unknown" in r.detail
    assert inp.events == []


def test_unreadable_selection_after_press_is_uncertain():
    ctrl, inp, _, _ = make_controller(make_idle_obs(count=2), make_idle_obs(count=2, villager=None))
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and "timed out" in r.detail
    assert not ctrl.executor.is_pending(SELECT_IDLE_KEY)  # selecting creates nothing: released, but backed off
    assert ctrl.state is ControllerState.BACKOFF and ctrl.stats.uncertain == 1
    assert ctrl.cycle().outcome is Outcome.WAITED and presses(inp) == 1


def test_game_losing_focus():
    # Perception says the game is not in front: no decision to act.
    ctrl, inp, _, _ = make_controller(make_idle_obs(game=False))
    assert ctrl.cycle().outcome is Outcome.WAITED and inp.events == []
    # Perception still thinks it is, but the window guard disagrees at send time: blocked, nothing pending.
    ctrl2, inp2, _, _ = make_controller(make_idle_obs(count=2), guard=False)
    r = ctrl2.cycle()
    assert r.outcome is Outcome.BLOCKED and r.action.status is ActionStatus.BLOCKED_NO_GAME
    assert inp2.events == [] and not ctrl2.executor.is_pending(SELECT_IDLE_KEY)


def test_focus_lost_during_verification_is_uncertain_not_success():
    ctrl, _, _, _ = make_controller(make_idle_obs(count=2), make_idle_obs(game=False))
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and "game not detected" in r.detail
    assert ctrl.stats.selected == 0


def test_input_error_is_uncertain_and_releases_the_key():
    ctrl, inp, _, _ = make_controller(make_idle_obs(count=2), backend=MockInput(fail_with=OSError("SendInput failed")))
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and ctrl.failures[-1].phase == "select"
    assert not ctrl.executor.is_pending(SELECT_IDLE_KEY)


def test_executor_refuses_a_second_press_while_unverified():
    ctrl, inp, _, _ = make_controller(make_idle_obs(count=2))
    action = ctrl.config.select_action()
    assert ctrl.executor.execute(action, await_verification=True).status is ActionStatus.EXECUTED
    assert ctrl.executor.execute(action, await_verification=True).status is ActionStatus.BLOCKED_PENDING
    assert presses(inp) == 1


def test_dry_run_sends_nothing():
    ctrl, _, _, _ = make_controller(make_idle_obs(count=2), dry_run=True)
    r = ctrl.cycle()
    assert r.outcome is Outcome.DRY_RUN and r.action.status is ActionStatus.DRY_RUN
    assert not ctrl.executor.is_pending(SELECT_IDLE_KEY)


def test_emergency_stop_ends_run():
    ctrl, inp, _, _ = make_controller(make_idle_obs(count=2))
    ctrl.stop.trigger("F12")
    stats = ctrl.run(max_cycles=10)
    assert stats.cycles == 1 and ctrl.state is ControllerState.STOPPED and inp.events == []


def test_waits_for_game_to_settle_before_any_key():
    config = replace(CFG, min_in_game_observations=2)
    ctrl, inp, _, _ = make_controller(make_idle_obs(game=False), make_idle_obs(count=1), make_idle_obs(count=1), make_idle_obs(count=1, villager=A), config=config)
    assert ctrl.cycle().outcome is Outcome.WAITED
    r = ctrl.cycle()
    assert r.outcome is Outcome.WAITED and "settle" in r.detail and inp.events == []
    assert ctrl.cycle().outcome is Outcome.SELECTED


def test_config_from_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"idle_hotkey": "f1"}')
    assert IdleVillagerConfig.from_json(p).select_action().keys == ("f1",)
    p.write_text('{"typo": 1}')
    with pytest.raises(ValueError):
        IdleVillagerConfig.from_json(p)
