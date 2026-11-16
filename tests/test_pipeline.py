"""Pipeline behaviour with a scripted OCR (no Tesseract needed)."""

from dataclasses import replace

from aoe2_ai.perception.capture import Frame
from aoe2_ai.perception.game_state import Age, Population, Status
from aoe2_ai.perception.pipeline import Perception
from aoe2_ai.perception.regions import RegionSet

from conftest import FakeOcr, tagged_frame

GOOD = {"wood": "200", "food": "150", "gold": "100", "stone": "200", "population": "4/5", "age": "Âge sombre"}


def make(texts, **kwargs):
    regions = RegionSet.default()
    ocr = FakeOcr(regions, texts)
    return Perception(regions, ocr, hud_check=False, **kwargs), tagged_frame(regions), ocr


def test_reads_all_calibrated_fields():
    perception, frame, _ = make(GOOD)
    s = perception.analyze(Frame(frame, 1.0, "test")).state
    assert (s.wood.value, s.food.value, s.gold.value, s.stone.value) == (200, 150, 100, 200)
    assert s.population.value == Population(4, 5)
    assert s.age.value is Age.DARK
    assert all(d.status is Status.VALID for d in (s.wood, s.food, s.gold, s.stone, s.population, s.age))


def test_uncalibrated_regions_are_unknown_and_not_read():
    base = RegionSet.default()
    regions = RegionSet(base.reference_size, {**base.regions, "game_time": replace(base["game_time"], calibrated=False)})
    ocr = FakeOcr(regions, {**GOOD, "game_time": "00:01:00"})
    s = Perception(regions, ocr, hud_check=False).analyze(Frame(tagged_frame(regions), 1.0, "test")).state
    assert s.game_time.status is Status.UNKNOWN and s.game_time.value is None
    assert "game_time" not in ocr.calls


def test_timer_and_selection():
    perception, frame, _ = make({**GOOD, "game_time": "00:12:43", "selection_name": "Chasseur", "selection_hp": "25/25"})
    s = perception.analyze(Frame(frame, 1.0, "test")).state
    assert s.game_time.value == 763
    assert (s.selection.value.name, s.selection.value.hp, s.selection.value.hp_max) == ("Chasseur", 25, 25)


def test_empty_selection_panel_is_unknown():
    perception, frame, _ = make(GOOD)  # nothing read in the selection regions
    s = perception.analyze(Frame(frame, 1.0, "test")).state
    assert s.selection.status is Status.UNKNOWN and s.selection.value is None


def test_ocr_failure_is_never_zero():
    perception, frame, _ = make({**GOOD, "wood": "", "food": "x", "gold": "0100"})
    s = perception.analyze(Frame(frame, 1.0, "test")).state
    for det in (s.wood, s.food, s.gold):
        assert det.status is Status.UNKNOWN and det.value is None
    assert s.stone.value == 200


def test_low_confidence_rejected():
    perception, frame, ocr = make(GOOD, min_confidence=0.6)
    ocr.confidence = 0.4
    s = perception.analyze(Frame(frame, 1.0, "test")).state
    assert s.wood.status is Status.UNKNOWN and s.wood.value is None


def test_values_change_across_frames_and_failures_go_stale():
    perception, frame, ocr = make(GOOD, stale_after=5.0)
    s = perception.analyze(Frame(frame, 0.0, "t0")).state
    assert s.wood.value == 200

    ocr.texts = {**GOOD, "wood": "345", "population": "7/10"}
    s = perception.analyze(Frame(frame, 2.0, "t1")).state
    assert (s.wood.value, s.wood.status) == (345, Status.VALID)
    assert s.population.value == Population(7, 10)

    ocr.texts = {**GOOD, "wood": ""}  # unreadable for a moment
    s = perception.analyze(Frame(frame, 4.0, "t2")).state
    assert (s.wood.value, s.wood.status, s.wood.timestamp) == (345, Status.STALE, 2.0)

    s = perception.analyze(Frame(frame, 8.0, "t3")).state  # still unreadable, too old now
    assert (s.wood.value, s.wood.status) == (None, Status.UNKNOWN)


def test_hud_missing_skips_ocr():
    regions = RegionSet.default()
    ocr = FakeOcr(regions, GOOD)
    perception = Perception(regions, ocr, hud_check=True)
    result = perception.analyze(Frame(tagged_frame(regions), 1.0, "menu"))
    assert not result.state.hud_visible and ocr.calls == []
    assert all(d.value is None for d in result.state.detections().values())


def test_observed_vs_tracked_state():
    perception, frame, ocr = make(GOOD)
    perception.analyze(Frame(frame, 0.0, "t0"))
    ocr.texts = {**GOOD, "gold": ""}
    result = perception.analyze(Frame(frame, 1.0, "t1"))
    assert result.observed.gold.status is Status.UNKNOWN
    assert result.state.gold.status is Status.STALE and result.state.gold.value == 100
