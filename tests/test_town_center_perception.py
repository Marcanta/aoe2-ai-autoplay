"""Town Center panel perception: real fixture frames + pure helpers."""

import numpy as np
import pytest

from aoe2_ai.controllers.observation import Observation
from aoe2_ai.controllers.town_center import DecisionKind, decide
from aoe2_ai.perception import Perception, load_frame
from aoe2_ai.perception.game_state import Detection, Selection, Status
from aoe2_ai.perception.regions import RegionSet
from aoe2_ai.perception.town_center import (
    QueueStatus,
    TownCenterDetector,
    fuzzy_name_match,
    parse_percent,
    progress_bar_fill,
)

from conftest import FRAMES, requires_tesseract

PRODUCING_FRAME = FRAMES / "feudal_20261116_185009.png"  # TC selected, Feudal Age research 22%
IDLE_FRAME = FRAMES / "feudal_20261116_185119.png"  # TC selected, empty queue
UNIT_FRAME = FRAMES / "unit_selected_20261116_184841.png"  # villager selected
MENU_FRAME = FRAMES / "main_menu_20261116_175453.png"

FORUM = Detection.valid(Selection("Forum", 2400, 2400), 0.9, 1.0, "Forum")


@pytest.fixture(scope="module")
def detector():
    return TownCenterDetector()


def regions_for(image):
    return RegionSet.default().for_frame(image)


def test_queue_producing_on_research_frame(detector):
    img = load_frame(PRODUCING_FRAME).image
    tc = detector.detect(img, regions_for(img), FORUM, 1.0)
    assert tc.is_selected and tc.is_producing and not tc.is_idle
    assert tc.villager_available.value is True


def test_queue_idle_on_empty_frame(detector):
    img = load_frame(IDLE_FRAME).image
    tc = detector.detect(img, regions_for(img), FORUM, 1.0)
    assert tc.is_idle and tc.villager_available.value is True
    assert tc.progress.status is Status.UNKNOWN  # not producing: nothing to read


def test_villager_button_absent_for_unit_panel(detector):
    img = load_frame(UNIT_FRAME).image
    det, score = detector.villager_available(img, regions_for(img), 1.0)
    assert det.status is Status.UNKNOWN and score < 0.6


def test_dark_screen_is_not_a_progress_bar(detector):
    img = load_frame(MENU_FRAME).image
    q, scores = detector.queue(img, regions_for(img), 1.0)
    assert q.status is Status.UNKNOWN and "panel not visible" in q.raw_text
    assert scores["bar_fill"] > 0.5  # the bar test alone would have been fooled


def test_covered_queue_area_is_unknown_not_idle(detector):
    img = load_frame(IDLE_FRAME).image.copy()
    r = regions_for(img)["tc_queue_slot"]
    rng = np.random.default_rng(0)
    img[r.y : r.y + 40, r.x : r.x + 120] = rng.integers(120, 255, (40, 120, 3), dtype=np.uint8)  # e.g. a tooltip
    q, _ = detector.queue(img, regions_for(img), 1.0)
    assert q.status is Status.UNKNOWN


def test_not_selected_or_unknown_selection(detector):
    img = load_frame(IDLE_FRAME).image
    other = detector.detect(img, regions_for(img), Detection.valid(Selection("Chasseur"), 0.9, 1.0), 1.0)
    assert other.selected.value is False and other.queue.status is Status.UNKNOWN
    unknown = detector.detect(img, regions_for(img), Detection.unknown(1.0, "nothing read"), 1.0)
    assert unknown.selected.status is Status.UNKNOWN and unknown.queue.status is Status.UNKNOWN


def test_fuzzy_names():
    assert fuzzy_name_match("Forum", ("Forum",)) == 1.0
    assert fuzzy_name_match("Town Centre", ("Town Center",)) > 0.8
    assert fuzzy_name_match("Chasseur", ("Forum", "Town Center")) == 0.0
    assert fuzzy_name_match("", ("Forum",)) == 0.0


def test_parse_percent():
    assert parse_percent("Développement 22%") == 22
    assert parse_percent("Création 100 %") == 100
    assert parse_percent("Création") is None and parse_percent("250%") is None


def test_progress_bar_fill():
    bar = np.zeros((8, 100, 3), np.uint8)
    bar[:, :30] = (0, 0, 255)  # red fill
    bar[:, 30:] = (27, 28, 57)  # dark track
    assert progress_bar_fill(bar) == pytest.approx(1.0)
    parchment = np.full((8, 100, 3), (127, 161, 198), np.uint8)
    assert progress_bar_fill(parchment) == 0.0


@requires_tesseract
@pytest.mark.parametrize("path, expected", [(IDLE_FRAME, DecisionKind.TRAIN_VILLAGER), (PRODUCING_FRAME, DecisionKind.WAIT), (UNIT_FRAME, DecisionKind.SELECT_TOWN_CENTER)])
def test_end_to_end_decision_on_fixture(path, expected, detector):
    """Real OCR + real TC detector on a fixture frame -> decision."""
    perception = Perception()
    try:
        frame = load_frame(path, 1.0)
        result = perception.analyze(frame)
        tc = detector.detect(frame.image, result.regions, result.observed.selection, 1.0, perception.ocr)
        obs = Observation(1.0, result.state.hud_visible, "fixture", result.state, tc, frame)
        assert decide(obs).kind is expected
        if path == PRODUCING_FRAME:
            assert tc.progress.value == 22
    finally:
        perception.close()


VILLAGER_FRAME = FRAMES / "tc_villager_20261116_203424.png"  # TC training a villager, 61%


def test_villager_in_production(detector):
    img = load_frame(VILLAGER_FRAME).image
    tc = detector.detect(img, regions_for(img), FORUM, 1.0)
    assert tc.is_producing and tc.villager_available.value is True


def test_queue_still_read_when_strip_under_bar_is_covered(detector):
    img = load_frame(VILLAGER_FRAME).image.copy()
    r = regions_for(img)["tc_queue_context"]
    img[r.y : r.y + r.h, r.x : r.x + r.w // 2] = (40, 40, 40)  # e.g. queued-item icons under the bar
    q, scores = detector.queue(img, regions_for(img), 1.0)
    assert q.value is QueueStatus.PRODUCING and scores["context_below"] < 0.8


@requires_tesseract
def test_villager_progress_and_item_ocr(detector):
    perception = Perception()
    try:
        frame = load_frame(VILLAGER_FRAME, 1.0)
        result = perception.analyze(frame)
        tc = detector.detect(frame.image, result.regions, result.observed.selection, 1.0, perception.ocr)
        assert tc.is_selected and tc.progress.value == 61 and tc.current_item.value == "villager"
        assert decide(Observation(1.0, True, "fixture", result.state, tc, frame)).kind is DecisionKind.WAIT
    finally:
        perception.close()
