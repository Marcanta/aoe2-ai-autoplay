import json
from dataclasses import replace

import pytest

from aoe2_ai.perception.game_state import Age, Detection, GameState, Population, StateTracker, Status


def state(ts, **fields):
    return replace(GameState.empty(ts, reason="not read"), hud_visible=True, **fields)


def test_unknown_has_no_value():
    det = Detection.unknown(1.0, "garbage")
    assert det.value is None and det.status is Status.UNKNOWN and not det.is_valid


def test_valid_requires_value_and_clamps_confidence():
    with pytest.raises(ValueError):
        Detection.valid(None, 0.9, 1.0)
    assert Detection.valid(5, 1.7, 1.0).confidence == 1.0


def test_empty_state_never_reports_zero():
    s = GameState.empty(10.0)
    for det in s.detections().values():
        assert det.value is None and det.status is Status.UNKNOWN


def test_json_round_trip():
    s = state(1.0, wood=Detection.valid(200, 0.9, 1.0, "200"), population=Detection.valid(Population(4, 5), 0.9, 1.0), age=Detection.valid(Age.DARK, 0.8, 1.0))
    data = json.loads(s.to_json())
    assert data["wood"] == {"value": 200, "confidence": 0.9, "timestamp": 1.0, "status": "valid", "raw_text": "200"}
    assert data["population"]["value"] == {"current": 4, "cap": 5}
    assert data["age"]["value"] == "dark"
    assert data["food"]["value"] is None and data["food"]["status"] == "unknown"


def test_tracker_valid_then_stale_then_unknown():
    tracker = StateTracker(stale_after=5.0)
    s1 = tracker.update(state(100.0, wood=Detection.valid(200, 0.9, 100.0)))
    assert s1.wood.status is Status.VALID and s1.wood.value == 200

    s2 = tracker.update(state(103.0, wood=Detection.unknown(103.0, "")))
    assert s2.wood.status is Status.STALE
    assert s2.wood.value == 200 and s2.wood.timestamp == 100.0  # when it was actually seen

    s3 = tracker.update(state(106.0, wood=Detection.unknown(106.0, "")))
    assert s3.wood.status is Status.UNKNOWN and s3.wood.value is None


def test_tracker_new_value_replaces_old():
    tracker = StateTracker()
    tracker.update(state(1.0, wood=Detection.valid(200, 0.9, 1.0)))
    s = tracker.update(state(2.0, wood=Detection.valid(350, 0.9, 2.0)))
    assert s.wood.value == 350 and s.wood.status is Status.VALID


def test_tracker_unknown_from_start_stays_unknown():
    s = StateTracker().update(state(1.0))
    assert all(d.status is Status.UNKNOWN and d.value is None for d in s.detections().values())


def test_tracker_rejects_clock_going_backwards():
    tracker = StateTracker(stale_after=5.0, clock_tolerance=2.0)
    tracker.update(state(1.0, game_time=Detection.valid(600, 0.9, 1.0)))
    s = tracker.update(state(2.0, game_time=Detection.valid(500, 0.9, 2.0)))
    assert s.game_time.status is Status.STALE and s.game_time.value == 600
    s = tracker.update(state(3.0, game_time=Detection.valid(602, 0.9, 3.0)))
    assert s.game_time.status is Status.VALID and s.game_time.value == 602


def test_tracker_accepts_clock_reset_after_gap():
    tracker = StateTracker(stale_after=5.0)
    tracker.update(state(1.0, game_time=Detection.valid(600, 0.9, 1.0)))
    s = tracker.update(state(20.0, game_time=Detection.valid(5, 0.9, 20.0)))  # new game
    assert s.game_time.status is Status.VALID and s.game_time.value == 5
