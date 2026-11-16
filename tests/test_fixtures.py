"""End-to-end detection on real (HUD-only) screenshots with Tesseract.

Add frames with ``python -m aoe2_ai.perception.capture --name <label>`` and
describe them in ``tests/fixtures/labels.json``.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pytest

from aoe2_ai.perception.capture import load_frame
from aoe2_ai.perception.detectors import _normalize
from aoe2_ai.perception.game_state import Status
from aoe2_ai.perception.pipeline import Perception

from conftest import FRAMES, load_labels, requires_tesseract

LABELS = load_labels()
FRAME_LABELS = LABELS["frames"]

pytestmark = requires_tesseract


@pytest.fixture(scope="module")
def perception() -> Perception:
    return Perception()


def _actual(det):
    value = det.value
    if hasattr(value, "current"):
        return [value.current, value.cap]
    if hasattr(value, "value"):  # enum
        return value.value
    return value


def _check(field, det, want, where="", may_miss=()):
    if field in may_miss and det.status is not Status.VALID:
        return  # known OCR miss: unknown is acceptable, a wrong value is not
    assert det.status is Status.VALID, f"{where}{field}: {det}"
    if field == "selection":
        # Names are raw OCR (English model on a French UI loses some accents): fuzzy match.
        got = _normalize(det.value.name)
        ratio = SequenceMatcher(None, got, _normalize(want["name"])).ratio()
        assert ratio >= 0.8, f"{where}selection name {det.value.name!r} vs {want['name']!r} ({ratio:.2f})"
        if "hp" in want:
            assert [det.value.hp, det.value.hp_max] == want["hp"], f"{where}selection hp: {det}"
    else:
        assert _actual(det) == want, f"{where}{field}: {det}"


def test_every_fixture_frame_is_labelled():
    on_disk = {p.name for p in FRAMES.glob("*.png")}
    assert on_disk == set(FRAME_LABELS), "label new frames in tests/fixtures/labels.json"


@pytest.mark.parametrize("name", sorted(FRAME_LABELS))
def test_frame(perception, name):
    expected = FRAME_LABELS[name]
    perception.reset()
    state = perception.analyze(load_frame(FRAMES / name, timestamp=0.0)).state

    assert state.hud_visible == expected.get("hud", True)
    for field, want in expected.items():
        if field.startswith("_") or field == "hud":
            continue
        det = getattr(state, field)
        if want is None:
            assert det.status is not Status.VALID and det.value is None, f"{field}: expected unreadable, got {det}"
        else:
            _check(field, det, want, may_miss=expected.get("_ocr_misses", ()))
            assert det.status is not Status.VALID or det.confidence >= 0.6


def test_sequence_tracks_changes(perception):
    """Frames in capture order: each one's values must replace the previous ones."""
    sequence = LABELS.get("sequence", [])
    changed = {f for a, b in zip(sequence, sequence[1:]) for f in ("wood", "food", "population", "game_time", "age") if FRAME_LABELS[a].get(f) != FRAME_LABELS[b].get(f)}
    if len(sequence) < 2 or not changed:
        pytest.skip("need at least two sequence frames with different values (capture more fixtures)")
    assert {"wood", "food", "population", "game_time", "age"} <= changed  # the fixture set really exercises changes

    perception.reset()
    previous_clock = -1
    for t, name in enumerate(sequence):
        # Frames are seconds apart in this test, so the tracker never fills gaps with STALE values:
        # every expected value has to come from the frame itself.
        state = perception.analyze(load_frame(FRAMES / name, timestamp=t * 60.0)).state
        for field, want in FRAME_LABELS[name].items():
            if field.startswith("_") or field == "hud":
                continue
            det = getattr(state, field)
            if want is None:
                assert det.value is None, f"{name} {field}: {det}"
            else:
                _check(field, det, want, f"{name} ", FRAME_LABELS[name].get("_ocr_misses", ()))
        assert state.game_time.value > previous_clock
        previous_clock = state.game_time.value
