"""TownCenterController: decision rules (unit) and the closed loop with mocked input (integration)."""

from dataclasses import replace

import pytest

from aoe2_ai.controllers.town_center import (
    TRAIN_KEY,
    ControllerState,
    DebugRecorder,
    DecisionKind,
    Outcome,
    TownCenterConfig,
    TownCenterController,
    decide,
)
from aoe2_ai.execution import ActionExecutor, ActionStatus, EmergencyStop, StaticGuard
from aoe2_ai.perception.capture import Frame
from aoe2_ai.perception.game_state import Status
from aoe2_ai.perception.town_center import QueueStatus

from controller_helpers import IDLE, PRODUCING, FakeClock, MockInput, ScriptedObserver, make_obs

import numpy as np

CFG = TownCenterConfig(debug_dir="", min_in_game_observations=1)  # settling tested separately


# --- decide(): unit tests ------------------------------------------------------
def test_idle_town_center_with_food_trains():
    d = decide(make_obs(food=200, **IDLE))
    assert d.kind is DecisionKind.TRAIN_VILLAGER


def test_exactly_50_food_is_enough():
    assert decide(make_obs(food=50, **IDLE)).kind is DecisionKind.TRAIN_VILLAGER


def test_insufficient_food_waits():
    d = decide(make_obs(food=49, **IDLE))
    assert d.kind is DecisionKind.WAIT and "insufficient food" in d.reason


def test_already_producing_waits():
    d = decide(make_obs(**PRODUCING))
    assert d.kind is DecisionKind.WAIT and "already producing" in d.reason


def test_wrong_building_selected_selects_town_center():
    d = decide(make_obs(selected=False))
    assert d.kind is DecisionKind.SELECT_TOWN_CENTER and "not a Town Center" in d.reason


@pytest.mark.parametrize(
    "kwargs, expected, text",
    [
        (dict(food=None), DecisionKind.WAIT, "food unreadable"),  # OCR failure is not 0, and not enough either
        (dict(food=300, food_status=Status.STALE), DecisionKind.WAIT, "food unreadable"),  # last known value not trusted
        (dict(queue=None), DecisionKind.WAIT, "queue unknown"),  # missing queue detection is not idle
        (dict(villager=None), DecisionKind.WAIT, "not available"),
        (dict(selected=None), DecisionKind.SELECT_TOWN_CENTER, "unreadable"),
        (dict(game=False), DecisionKind.WAIT, "game not detected"),
    ],
)
def test_unknown_perception_values_never_train(kwargs, expected, text):
    d = decide(make_obs(**{**IDLE, **kwargs}))
    assert d.kind is expected and text in d.reason


def test_housed_waits_but_unknown_population_does_not_block():
    assert decide(make_obs(population=(10, 10), **IDLE)).kind is DecisionKind.WAIT
    assert decide(make_obs(population=None, **IDLE)).kind is DecisionKind.TRAIN_VILLAGER


# --- closed loop: integration with mocked input ------------------------------------
def make_controller(*observations, backend=None, guard=True, dry_run=False, config=CFG, recorder=None):
    clock = FakeClock()
    backend = MockInput() if backend is None and not dry_run else backend
    stop = EmergencyStop()
    executor = ActionExecutor(backend, StaticGuard(guard, "test"), dry_run=dry_run, stop=stop, clock=clock, sleep=clock.sleep)
    observer = ScriptedObserver(*observations)
    ctrl = TownCenterController(observer, executor, config, recorder=recorder or DebugRecorder(None), clock=clock, sleep=clock.sleep)
    return ctrl, backend, observer, clock


def test_trains_and_verifies():
    ctrl, inp, _, _ = make_controller(make_obs(**IDLE), make_obs(food=150, **PRODUCING))
    r = ctrl.cycle()
    assert r.outcome is Outcome.TRAINED, r.detail
    assert inp.presses("q") == 1
    assert ctrl.stats.villagers_queued == 1
    assert not ctrl.executor.is_pending(TRAIN_KEY)
    assert "food 200 -> 150" in r.detail
    assert ctrl.state is ControllerState.OBSERVING


def test_does_nothing_while_producing():
    ctrl, inp, _, _ = make_controller(make_obs(**PRODUCING))
    for _ in range(5):
        assert ctrl.cycle().outcome is Outcome.WAITED
    assert inp.events == []


def test_insufficient_food_sends_nothing():
    ctrl, inp, _, _ = make_controller(make_obs(food=20, **IDLE))
    assert ctrl.cycle().outcome is Outcome.WAITED and inp.events == []


def test_wrong_building_selects_then_verifies_selection():
    ctrl, inp, _, _ = make_controller(make_obs(selected=False), make_obs(**PRODUCING))
    r = ctrl.cycle()
    assert r.outcome is Outcome.SELECTED
    assert inp.events == [("press", "h")]  # only selection: training waits for the next observation


def test_selection_failure_is_recorded_without_training(tmp_path):
    frame = Frame(np.zeros((10, 10, 3), np.uint8), 0.0, "test")
    wrong = replace(make_obs(selected=False), frame=frame)
    ctrl, inp, _, _ = make_controller(wrong, wrong, recorder=DebugRecorder(tmp_path))
    r = ctrl.cycle()
    assert r.outcome is Outcome.FAILED
    assert inp.presses("q") == 0
    assert ctrl.failures and ctrl.failures[0].phase == "verify_selection"
    assert ctrl.failures[0].screenshot.endswith(".png")
    assert list(tmp_path.glob("*.png")) and list(tmp_path.glob("*.json"))


def test_failed_keyboard_input_blocks_further_commands_until_resolved():
    ctrl, inp, obs, _ = make_controller(make_obs(**IDLE), make_obs(**IDLE), backend=MockInput(fail_with=OSError("SendInput failed")))
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and ctrl.state is ControllerState.UNCERTAIN
    assert ctrl.failures[-1].phase == "verify_training"
    inp.fail_with = None
    # Uncertain: observes only, even though the TC looks idle with plenty of food.
    assert ctrl.cycle().outcome is Outcome.UNCERTAIN
    assert inp.presses("q") == 0
    # Second idle read resolves it: the key did not take.
    assert ctrl.cycle().outcome is Outcome.RESOLVED
    assert not ctrl.executor.is_pending(TRAIN_KEY)


def test_uncertain_resolved_by_seeing_production():
    ctrl, inp, obs, _ = make_controller(make_obs(**IDLE), make_obs(**PRODUCING), backend=MockInput(fail_with=OSError("boom")))
    assert ctrl.cycle().outcome is Outcome.UNCERTAIN
    inp.fail_with = None
    r = ctrl.cycle()
    assert r.outcome is Outcome.RESOLVED and "producing" in r.detail
    assert inp.events == []


def test_failed_verification_records_failure_and_does_not_retry_immediately(tmp_path):
    frame = Frame(np.zeros((10, 10, 3), np.uint8), 0.0, "test")
    idle = replace(make_obs(**IDLE), frame=frame)
    ctrl, inp, _, clock = make_controller(idle, recorder=DebugRecorder(tmp_path))
    r = ctrl.cycle()
    assert r.outcome is Outcome.FAILED and "still idle" in r.detail
    assert ctrl.state is ControllerState.BACKOFF
    assert ctrl.failures[0].screenshot and list(tmp_path.glob("*.png"))
    assert not ctrl.executor.is_pending(TRAIN_KEY)  # outcome is known: not queued
    # Back to observation, but no blind retry during the back-off.
    assert ctrl.cycle().outcome is Outcome.WAITED
    assert inp.presses("q") == 1
    clock.now += CFG.failure_backoff
    ctrl.cycle()
    assert inp.presses("q") == 2


def test_verification_timeout_is_uncertain():
    unknown_queue = make_obs(queue=None)
    ctrl, inp, _, _ = make_controller(make_obs(**IDLE), unknown_queue)
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and "timed out" in r.detail
    assert ctrl.executor.is_pending(TRAIN_KEY)
    for _ in range(3):
        ctrl.cycle()
    assert inp.presses("q") == 1  # no new training command while uncertain


def test_uncertain_never_resolved_halts():
    ctrl, inp, _, _ = make_controller(make_obs(**IDLE), make_obs(queue=None), config=replace(CFG, max_uncertain_cycles=3, max_consecutive_failures=10))
    ctrl.cycle()
    outcomes = [ctrl.cycle().outcome for _ in range(3)]
    assert outcomes[-1] is Outcome.HALTED and ctrl.state is ControllerState.HALTED
    assert ctrl.cycle().outcome is Outcome.HALTED
    assert inp.presses("q") == 1


def test_duplicate_command_prevention_at_executor_level():
    ctrl, inp, _, _ = make_controller(make_obs(**IDLE), make_obs(queue=None))
    ctrl.cycle()  # uncertain, pending
    # Even if something bypassed the controller state, the executor refuses a second train.
    res = ctrl.executor.execute(ctrl.config.train_action(), await_verification=True)
    assert res.status is ActionStatus.BLOCKED_PENDING
    assert inp.presses("q") == 1


def test_continues_monitoring_and_queues_next_villager():
    ctrl, inp, _, clock = make_controller(
        make_obs(**IDLE), make_obs(**PRODUCING),  # train 1 + verify
        make_obs(**PRODUCING), make_obs(**PRODUCING),  # villager in production: wait
        make_obs(food=150, **IDLE), make_obs(food=100, **PRODUCING),  # finished: train 2 + verify
        make_obs(food=100, **PRODUCING),
    )
    outcomes = []
    for _ in range(5):
        outcomes.append(ctrl.cycle().outcome)
        clock.now += 10  # time passes between cycles (a villager takes 25 s)
    assert outcomes == [Outcome.TRAINED, Outcome.WAITED, Outcome.WAITED, Outcome.TRAINED, Outcome.WAITED]
    assert inp.presses("q") == 2 and ctrl.stats.villagers_queued == 2


def test_never_acts_when_game_not_detected():
    ctrl, inp, _, _ = make_controller(make_obs(game=False))
    assert ctrl.cycle().outcome is Outcome.WAITED
    # And the executor's own guard blocks even a TRAIN decision if the window is wrong.
    ctrl2, inp2, _, _ = make_controller(make_obs(**IDLE), guard=False)
    assert ctrl2.cycle().outcome is Outcome.BLOCKED
    assert inp.events == [] and inp2.events == []
    assert not ctrl2.executor.is_pending(TRAIN_KEY)


def test_dry_run_sends_nothing():
    ctrl, _, _, _ = make_controller(make_obs(**IDLE), dry_run=True)
    r = ctrl.cycle()
    assert r.outcome is Outcome.DRY_RUN and r.action.status is ActionStatus.DRY_RUN
    assert not ctrl.executor.is_pending(TRAIN_KEY)


def test_emergency_stop_ends_run():
    ctrl, inp, _, _ = make_controller(make_obs(**PRODUCING))
    ctrl.stop.trigger("F12")
    stats = ctrl.run(max_cycles=10)
    assert stats.cycles == 1 and ctrl.state is ControllerState.STOPPED and inp.events == []


def test_halts_after_repeated_failures():
    config = replace(CFG, max_consecutive_failures=2, failure_backoff=0.0)
    ctrl, inp, _, _ = make_controller(make_obs(**IDLE), config=config)
    stats = ctrl.run(max_cycles=10)
    assert ctrl.state is ControllerState.HALTED
    assert inp.presses("q") == 2 and stats.failures == 2


def test_config_from_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"select_hotkey": "f2", "train_hotkey": "c"}')
    cfg = TownCenterConfig.from_json(p)
    assert cfg.select_action().keys == ("f2",) and cfg.train_action().keys == ("c",)
    p.write_text('{"typo": 1}')
    with pytest.raises(ValueError):
        TownCenterConfig.from_json(p)


def test_uncertain_reselects_when_nothing_selected():
    ctrl, inp, _, _ = make_controller(make_obs(**IDLE), make_obs(queue=None), make_obs(selected=None))
    assert ctrl.cycle().outcome is Outcome.UNCERTAIN  # verification never conclusive
    inp.events.clear()
    r = ctrl.cycle()
    assert r.outcome is Outcome.UNCERTAIN and "re-selecting" in r.detail
    assert inp.events == [("press", "h")]  # select only, never train


def test_error_caused_by_stop_is_not_a_failure():
    class Boom:
        def __init__(self, stop):
            self.stop = stop

        def observe(self):
            self.stop.trigger("Ctrl+C")  # Ctrl+C also kills the Tesseract children
            raise RuntimeError("TesseractError(3221225786)")

    ctrl, _, _, _ = make_controller(make_obs(**IDLE))
    ctrl.observer = Boom(ctrl.stop)
    stats = ctrl.run(max_cycles=5)
    assert stats.failures == 0 and ctrl.state is ControllerState.STOPPED and stats.cycles == 1


def test_unreadable_queue_with_tc_selected_is_captured_rate_limited(tmp_path):
    frame = Frame(np.zeros((10, 10, 3), np.uint8), 0.0, "test")
    obs = replace(make_obs(queue=None), frame=frame)
    ctrl, inp, _, clock = make_controller(obs, recorder=DebugRecorder(tmp_path))
    for _ in range(3):
        assert ctrl.cycle().outcome is Outcome.WAITED
    assert len(list(tmp_path.glob("*queue_unknown*.png"))) == 1 and not ctrl.failures
    clock.now += CFG.unknown_capture_interval
    ctrl.cycle()
    assert len(list(tmp_path.glob("*queue_unknown*.png"))) == 2 and inp.events == []


def test_waits_for_game_to_settle_before_any_key():
    # Game just loaded / window just came back: first in-game observation sends nothing.
    config = replace(CFG, min_in_game_observations=2)
    ctrl, inp, _, _ = make_controller(make_obs(game=False), make_obs(selected=False), make_obs(selected=False), make_obs(**PRODUCING), config=config)
    assert ctrl.cycle().outcome is Outcome.WAITED  # not in game
    r = ctrl.cycle()
    assert r.outcome is Outcome.WAITED and "settle" in r.detail and inp.events == []
    assert ctrl.cycle().outcome is Outcome.SELECTED  # second consecutive in-game observation
    assert inp.events == [("press", "h")]
