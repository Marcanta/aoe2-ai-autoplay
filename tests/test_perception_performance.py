"""Perception profiles, region cache and timing instrumentation."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from aoe2_ai.benchmark import FileSource, check
from aoe2_ai.controllers.observation import LiveObserver
from aoe2_ai.execution import StaticGuard
from aoe2_ai.perception import Perception, load_frame
from aoe2_ai.perception.cache import RegionCache, fingerprint
from aoe2_ai.perception.capture import Frame
from aoe2_ai.perception.game_state import Status
from aoe2_ai.perception.metrics import StageTimer, TimedOcr
from aoe2_ai.perception.ocr import OcrResult
from aoe2_ai.perception.profile import (
    ECONOMY,
    FULL,
    IDLE_VILLAGER,
    NOT_REQUESTED,
    TOWN_CENTER,
    PerceptionProfile,
)
from aoe2_ai.perception.regions import RegionSet
from aoe2_ai.perception.villager import VillagerDetector

from conftest import FRAMES, FakeOcr, load_labels, requires_tesseract, tagged_frame

GOOD = {"wood": "200", "food": "150", "gold": "100", "stone": "200", "population": "4/5", "age": "Âge sombre",
        "game_time": "00:01:00", "selection_name": "Villageois", "selection_hp": "25/25"}
RESOURCES = ("wood", "food", "gold", "stone")


def make(texts=GOOD, **kwargs):
    regions = RegionSet.default()
    ocr = FakeOcr(regions, texts)
    return Perception(regions, ocr, hud_check=False, **kwargs), tagged_frame(regions), ocr


def touch(frame: np.ndarray, regions: RegionSet, name: str) -> np.ndarray:
    """Copy of ``frame`` with one pixel of region ``name`` changed (its tag pixel at [0, 0] kept)."""
    out = frame.copy()
    r = regions[name]
    out[r.y + r.h - 1, r.x + r.w - 1, 2] ^= 0x40
    return out


# --- profiles ----------------------------------------------------------------
def test_profile_dependencies_and_names():
    assert {"food", "population", "selection", "town_center"} == set(TOWN_CENTER.fields)
    assert {"idle", "selection", "villager"} == set(IDLE_VILLAGER.fields)
    assert set(ECONOMY.fields) == {*RESOURCES, "population"}
    assert FULL.is_full and not ECONOMY.is_full
    assert "selection" in PerceptionProfile.of("x", "town_center_text")  # panels pull in the name they read
    assert "town_center" in PerceptionProfile.of("x", "town_center_text")
    assert PerceptionProfile.named("Town-Center") is TOWN_CENTER
    with pytest.raises(ValueError):
        PerceptionProfile.of("x", "lumber")
    with pytest.raises(ValueError):
        PerceptionProfile.named("nope")


def test_default_profile_reads_every_field():
    perception, frame, ocr = make()
    s = perception.analyze(Frame(frame, 1.0, "t")).state
    assert all(s.detections()[f].is_valid for f in (*RESOURCES, "population", "game_time", "age", "selection"))
    assert {"wood", "food", "gold", "stone", "population", "game_time", "age", "selection_name", "selection_hp"} <= set(ocr.calls)


def test_profile_reads_only_requested_fields():
    perception, frame, ocr = make()
    result = perception.analyze(Frame(frame, 1.0, "t"), ECONOMY)
    assert set(ocr.calls) == {*RESOURCES, "population"}
    s = result.state
    assert all(getattr(s, f).is_valid for f in (*RESOURCES, "population"))
    for f in ("game_time", "age", "selection"):
        det = getattr(s, f)
        assert det.status is Status.UNKNOWN and det.value is None and det.raw_text == NOT_REQUESTED


def test_unrequested_field_is_never_valid_and_ages_out():
    perception, frame, _ = make(stale_after=5.0)
    perception.analyze(Frame(frame, 0.0, "t0"))  # full: age read
    s = perception.analyze(Frame(frame, 2.0, "t1"), ECONOMY).state
    assert s.age.status is Status.STALE  # last value kept as STALE, like any unread field
    s = perception.analyze(Frame(frame, 10.0, "t2"), ECONOMY).state
    assert s.age.status is Status.UNKNOWN and s.age.value is None


def test_live_observer_profile_skips_panels():
    regions = RegionSet.default()
    perception = Perception(regions, FakeOcr(regions, GOOD), hud_check=False)
    frame = tagged_frame(regions)

    class Source:
        def grab(self):
            return Frame(frame, 1.0, "t")

        def close(self):
            pass

    obs = LiveObserver(StaticGuard(True, "t"), perception=perception, source=Source(), profile=ECONOMY).observe()
    assert obs.town_center.selected.raw_text == NOT_REQUESTED and not obs.town_center.selected.is_valid
    assert obs.idle.present.raw_text == NOT_REQUESTED and not obs.idle.present.is_valid
    assert obs.villager.selected.raw_text == NOT_REQUESTED and not obs.villager.selected.is_valid
    assert obs.state.food.value == 150 and obs.state.selection.raw_text == NOT_REQUESTED

    obs = LiveObserver(StaticGuard(True, "t"), perception=perception, source=Source(), profile=IDLE_VILLAGER).observe()
    assert obs.villager.selected.is_valid  # the name was read for the panel
    assert obs.state.food.status is not Status.VALID


# --- cache -------------------------------------------------------------------
def test_fingerprint_is_pixel_exact():
    a = np.zeros((9, 30, 3), np.uint8)
    b = a.copy()
    assert fingerprint(a) == fingerprint(b)
    b[4, 4, 1] = 1
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a) != fingerprint(a.reshape(30, 9, 3))  # same bytes, other shape


def test_cache_off_by_default():
    perception, frame, ocr = make()
    perception.analyze(Frame(frame, 0.0, "t0"))
    n = len(ocr.calls)
    perception.analyze(Frame(frame, 1.0, "t1"))
    assert len(ocr.calls) == 2 * n


def test_identical_pixels_reuse_the_detection_unchanged():
    perception, frame, ocr = make(cache=True)
    first = perception.analyze(Frame(frame, 0.0, "t0"))
    calls = len(ocr.calls)
    second = perception.analyze(Frame(frame.copy(), 3.0, "t1"))
    assert len(ocr.calls) == calls  # no OCR at all
    for f in ("wood", "population", "age", "selection"):
        a, b = getattr(first.observed, f), getattr(second.observed, f)
        assert b == a and b.timestamp == 0.0 and b.status is Status.VALID  # original timestamp, confidence, status
    assert set(second.cache_hits) >= {"wood", "population", "age", "selection"}
    assert second.observed.timestamp == 3.0


def test_changed_pixels_are_read_again():
    perception, frame, ocr = make(cache=True)
    perception.analyze(Frame(frame, 0.0, "t0"))
    ocr.calls.clear()
    ocr.texts = {**GOOD, "wood": "345", "stone": "999"}  # stone's pixels do not change: its new text must not appear
    s = perception.analyze(Frame(touch(frame, perception.regions, "wood"), 1.0, "t1")).observed
    assert ocr.calls == ["wood"]
    assert (s.wood.value, s.wood.timestamp) == (345, 1.0)
    assert (s.stone.value, s.stone.timestamp) == (200, 0.0)


def test_multi_region_fields_depend_on_every_region():
    perception, frame, ocr = make(cache=True)
    perception.analyze(Frame(frame, 0.0, "t0"))
    ocr.calls.clear()
    ocr.texts = {**GOOD, "selection_hp": "20/25"}
    s = perception.analyze(Frame(touch(frame, perception.regions, "selection_hp"), 1.0, "t1")).observed
    assert set(ocr.calls) == {"selection_name", "selection_hp"}
    assert s.selection.value.hp == 20


def test_unknown_is_cached_as_unknown_never_valid():
    perception, frame, ocr = make({**GOOD, "wood": "", "food": "x"}, cache=True)
    perception.analyze(Frame(frame, 0.0, "t0"))
    ocr.texts = GOOD  # would now read fine, but the pixels are the same: the result must be the same
    s = perception.analyze(Frame(frame, 1.0, "t1")).observed
    for det in (s.wood, s.food):
        assert det.status is Status.UNKNOWN and det.value is None


def test_low_confidence_kept_with_its_confidence():
    perception, frame, ocr = make(cache=True)
    ocr.confidence = 0.4
    first = perception.analyze(Frame(frame, 0.0, "t0")).observed.wood
    second = perception.analyze(Frame(frame, 1.0, "t1")).observed.wood
    assert second == first and second.status is Status.UNKNOWN and second.confidence == pytest.approx(0.4)


def test_cache_invalidated_by_configuration_changes():
    perception, frame, ocr = make(cache=True)
    perception.analyze(Frame(frame, 0.0, "t0"))

    def reads(image, ts) -> int:
        before = len(ocr.calls)
        perception.analyze(Frame(image, ts, "t"))
        return len(ocr.calls) - before

    assert reads(frame, 1.0) == 0
    perception.min_confidence = 0.7  # validation setting
    assert reads(frame, 2.0) > 0
    assert reads(frame, 3.0) == 0

    regions = perception.regions  # region configuration (same pixels inside the moved region)
    perception.regions = RegionSet(regions.reference_size, {**regions.regions, "wood": replace(regions["wood"], w=regions["wood"].w - 1)})
    assert reads(frame, 4.0) > 0

    h, w = frame.shape[:2]  # resolution
    big = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    assert reads(big, 5.0) > 0
    assert reads(big, 6.0) == 0
    assert reads(frame, 7.0) > 0  # back to the first resolution: not reused either

    perception.reset()
    assert len(perception.cache) == 0
    assert reads(frame, 8.0) > 0


def test_panel_ocr_cache_preserves_detection():
    image = load_frame(FRAMES / "idle_seven_villager_selected_20261116_2129.png").image
    regions = RegionSet.default().for_frame(image)

    class Ocr:
        calls = 0

        def read(self, crop, **_):
            Ocr.calls += 1
            return OcrResult("7", 0.9, ("7",))

    detector, cache = VillagerDetector(), RegionCache()
    first = detector.idle(image, regions, 0.0, Ocr(), cache=cache)
    second = detector.idle(image, regions, 5.0, Ocr(), cache=cache)
    assert Ocr.calls == 1
    assert second.count == first.count and second.count.timestamp == 0.0 and second.count.value == 7
    assert second.present.timestamp == 5.0  # the colour check always runs


def test_cache_rejects_none():
    with pytest.raises(ValueError):
        RegionCache().put("x", "k", None)


# --- timing ------------------------------------------------------------------
def test_stage_timer_and_timed_ocr():
    regions = RegionSet.default()
    timer = StageTimer()
    ocr = TimedOcr(FakeOcr(regions, GOOD), timer)
    ocr.read(tagged_frame(regions)[regions["wood"].y:, regions["wood"].x:])
    assert timer.calls()["ocr"] == 1 and timer.seconds()["ocr"] >= 0.0
    assert ocr.texts is GOOD  # attributes of the wrapped engine stay visible

    perception, frame, _ = make()
    result = perception.analyze(Frame(frame, 0.0, "t"))
    assert {"regions", "ocr", "parse", "track", "total"} <= set(result.timings)
    assert result.ocr_launches is None  # FakeOcr does not count launches


# --- real frames -------------------------------------------------------------
LABELS = load_labels()["frames"]


@pytest.fixture(scope="module")
def cached_perception():
    p = Perception(cache=True)
    yield p
    p.close()


@requires_tesseract
@pytest.mark.parametrize("name", sorted(LABELS))
def test_cache_is_transparent_on_real_frames(cached_perception, name):
    """Second observation of an unchanged frame: no Tesseract launch, same detections, still correct."""
    perception = cached_perception
    perception.reset()
    observer = LiveObserver(StaticGuard(True, "test"), perception=perception, source=FileSource())
    first = observer.observe_frame(load_frame(FRAMES / name, timestamp=0.0))
    second = observer.observe_frame(load_frame(FRAMES / name, timestamp=1.0))

    assert second.ocr_launches == 0
    same = lambda d: (d.value, d.status, d.confidence, d.raw_text)  # noqa: E731
    assert {k: same(d) for k, d in first.state.detections().items()} == {k: same(d) for k, d in second.state.detections().items()}
    # Panel values without OCR (colour/template checks) are recomputed each frame: compare content, not timestamps.
    panels = lambda o: tuple(same(d) for d in (o.idle.present, o.idle.count, o.town_center.selected,  # noqa: E731
                                               o.town_center.queue, o.town_center.villager_available, o.villager.selected))
    assert panels(second) == panels(first)
    verdicts = check(second, LABELS[name])
    assert "wrong" not in verdicts.values() and "unknown" not in verdicts.values(), verdicts


@requires_tesseract
def test_changed_region_is_recognised_again_on_real_frame(cached_perception):
    perception = cached_perception
    perception.reset()
    image = load_frame(FRAMES / "eco_20261116_184200.png").image
    first = perception.analyze(Frame(image, 0.0, "t0"))
    assert first.ocr_launches > 0 and {"ocr", "total"} <= set(first.timings)

    changed = image.copy()
    r = first.regions["stone"]
    changed[r.y, r.x, 0] ^= 1  # one corner pixel, one intensity level
    second = perception.analyze(Frame(changed, 1.0, "t1"))
    assert 0 < second.ocr_launches < first.ocr_launches
    assert "stone" not in second.cache_hits and "wood" in second.cache_hits
    assert second.observed.stone.timestamp == 1.0 and second.observed.stone.value == first.observed.stone.value
    assert second.observed.wood.timestamp == 0.0
