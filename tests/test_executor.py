"""ActionExecutor gates: game detection, dry run, duplicate prevention, cooldown, input failures, stop."""

import logging
import time

import pytest

from aoe2_ai.execution import (
    Action,
    ActionExecutor,
    ActionKind,
    ActionStatus,
    CallableGuard,
    EmergencyStop,
    ForegroundWindowGuard,
    GuardResult,
    StaticGuard,
)
from aoe2_ai.execution.guard import ForegroundWindow
from aoe2_ai.execution.stop import StopKeyWatcher, console_handler, parse_keys, start_exit_watchdog, vk_code

from controller_helpers import FakeClock, MockInput

IN_GAME = StaticGuard(True, "game")
NO_GAME = StaticGuard(False, "notepad.exe in front")
TRAIN = Action.key("train_villager", "q", dedupe_key="train_villager", post_delay=0.3)


def make(backend=None, guard=IN_GAME, **kwargs):
    clock = FakeClock()
    backend = backend if backend is not None else MockInput()
    ex = ActionExecutor(backend, guard, clock=clock, sleep=clock.sleep, **kwargs)
    return ex, backend, clock


def test_executes_key_and_logs(caplog):
    ex, inp, clock = make()
    with caplog.at_level(logging.INFO, logger="aoe2_ai.execution"):
        r = ex.execute(TRAIN)
    assert r.status is ActionStatus.EXECUTED and r.sent
    assert inp.events == [("press", "q")]
    assert clock.now == pytest.approx(1000.3)  # post_delay waited
    assert "train_villager" in caplog.text and "executed" in caplog.text
    assert ex.history == [r]


def test_hotkey_and_click_dispatch():
    ex, inp, _ = make()
    ex.execute(Action.hotkey("group", "ctrl", "1"))
    ex.execute(Action.click("tc", 10, 20, button="right"))
    assert inp.events == [("hotkey", ("ctrl", "1")), ("click", 10, 20, "right")]


def test_blocked_when_game_not_detected():
    ex, inp, _ = make(guard=NO_GAME)
    r = ex.execute(TRAIN)
    assert r.status is ActionStatus.BLOCKED_NO_GAME and r.blocked
    assert inp.events == []
    assert "notepad" in r.detail


def test_guard_rechecked_after_pre_delay():
    answers = iter([GuardResult(True, "game"), GuardResult(False, "alt-tabbed")])
    ex, inp, _ = make(guard=CallableGuard(lambda: next(answers)))
    r = ex.execute(Action.key("x", "x", pre_delay=1.0))
    assert r.status is ActionStatus.BLOCKED_NO_GAME and inp.events == []


def test_guard_error_blocks():
    def boom():
        raise OSError("no window")

    ex, inp, _ = make(guard=CallableGuard(boom))
    assert ex.execute(TRAIN).status is ActionStatus.BLOCKED_NO_GAME and inp.events == []


def test_dry_run_sends_nothing_but_still_checks_game():
    ex, _, _ = make(backend=None, dry_run=True)
    assert ex.execute(TRAIN).status is ActionStatus.DRY_RUN
    ex2, _, _ = make(backend=None, guard=NO_GAME, dry_run=True)
    assert ex2.execute(TRAIN).status is ActionStatus.BLOCKED_NO_GAME


def test_backend_required_unless_dry_run():
    with pytest.raises(ValueError):
        ActionExecutor(None, IN_GAME)


def test_duplicate_prevented_until_resolved():
    ex, inp, _ = make()
    assert ex.execute(TRAIN, await_verification=True).sent
    second = ex.execute(TRAIN, await_verification=True)
    assert second.status is ActionStatus.BLOCKED_PENDING
    assert ex.execute(TRAIN).status is ActionStatus.BLOCKED_PENDING  # even without await
    assert inp.presses("q") == 1
    ex.resolve("train_villager", "verified")
    assert ex.execute(TRAIN, await_verification=True).sent
    assert inp.presses("q") == 2


def test_cooldown_blocks_rapid_repeat():
    ex, inp, clock = make()
    a = Action.key("select_tc", "h", cooldown=2.0, post_delay=0)
    assert ex.execute(a).sent
    assert ex.execute(a).status is ActionStatus.BLOCKED_COOLDOWN
    clock.now += 2.0
    assert ex.execute(a).sent
    assert inp.presses("h") == 2


def test_failed_input_is_uncertain_and_stays_pending():
    ex, inp, _ = make(backend=MockInput(fail_with=OSError("SendInput failed")))
    r = ex.execute(TRAIN, await_verification=True)
    assert r.status is ActionStatus.FAILED and r.uncertain and not r.sent
    assert ex.is_pending("train_villager")  # outcome unknown: no repeat allowed
    inp.fail_with = None
    assert ex.execute(TRAIN, await_verification=True).status is ActionStatus.BLOCKED_PENDING
    assert inp.events == []


def test_failsafe_triggers_emergency_stop():
    class FailSafeException(Exception):
        pass

    ex, _, _ = make(backend=MockInput(fail_with=FailSafeException("corner")))
    assert ex.execute(TRAIN).status is ActionStatus.FAILED
    assert ex.stop.triggered
    assert ex.execute(Action.key("other", "x")).status is ActionStatus.BLOCKED_STOPPED


def test_emergency_stop_blocks_everything():
    stop = EmergencyStop()
    ex, inp, _ = make(stop=stop)
    stop.trigger("F12")
    assert ex.execute(TRAIN).status is ActionStatus.BLOCKED_STOPPED
    assert inp.events == []
    assert stop.reason == "F12"


def test_action_validation():
    with pytest.raises(ValueError):
        Action("bad", ActionKind.KEY, ())
    with pytest.raises(ValueError):
        Action("bad", ActionKind.KEY, ("ctrl", "1"))
    with pytest.raises(ValueError):
        Action("bad", ActionKind.CLICK)
    with pytest.raises(ValueError):
        Action.hotkey("bad")
    with pytest.raises(ValueError):
        Action.key("bad", "q", post_delay=-1)


def test_foreground_guard_matches_process_or_title():
    g = ForegroundWindowGuard(probe=lambda: ForegroundWindow("Age of Empires II: Definitive Edition", r"C:\Steam\AoE2DE_s.exe"))
    assert g.check().ok
    g = ForegroundWindowGuard(probe=lambda: ForegroundWindow("Age of Empires II fan page - Chrome", r"C:\chrome.exe"))
    assert not g.check().ok  # title alone does not win over a known, different process
    g = ForegroundWindowGuard(probe=lambda: ForegroundWindow("Age of Empires II: Definitive Edition", ""))
    assert g.check().ok  # process unreadable: fall back to the title
    assert not ForegroundWindowGuard(probe=lambda: None).check().ok


def test_stop_key_codes():
    assert vk_code("f12") == 0x7B and vk_code("pause") == 0x13 and vk_code("x") == ord("X")
    with pytest.raises(ValueError):
        vk_code("ctrl+x")
    assert parse_keys("F12, pause") == ("f12", "pause")
    with pytest.raises(ValueError):
        parse_keys(" , ")


class FakeKeys:
    """Scripted GetAsyncKeyState: ``down`` is the set of VKs currently held."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.down: set[int] = set()
        self.fail = fail

    def __call__(self, vk: int) -> int:
        if self.fail:
            raise self.fail
        return -32768 if vk in self.down else 0


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()


def test_stop_key_watcher_any_of_several_keys():
    stop, keys = EmergencyStop(), FakeKeys()
    with StopKeyWatcher(stop, "f12,pause", poll=0.005, key_state=keys) as w:
        assert w.alive and not stop.triggered
        keys.down.add(0x13)  # Pause
        assert wait_until(lambda: stop.triggered)
    assert "PAUSE" in stop.reason


def test_stop_key_watcher_failure_triggers_stop():
    stop = EmergencyStop()
    with StopKeyWatcher(stop, "f12", poll=0.005, key_state=FakeKeys(fail=OSError("no access"))):
        assert wait_until(lambda: stop.triggered)
    assert "watcher failed" in stop.reason


def test_console_ctrl_c_triggers_stop():
    stop = EmergencyStop()
    assert console_handler(stop)(0) is True
    assert stop.triggered and stop.reason == "Ctrl+C"


def test_exit_watchdog_forces_exit_after_grace():
    stop, codes = EmergencyStop(), []
    start_exit_watchdog(stop, grace=0.01, exit=codes.append)
    time.sleep(0.05)
    assert codes == []  # nothing happens before a stop
    stop.trigger("F12")
    assert wait_until(lambda: codes == [2])
