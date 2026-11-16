"""Idle-villager perception: real skirmish frames end to end, plus the pure helpers."""

import numpy as np
import pytest

from aoe2_ai.controllers.idle_villager import DecisionKind, decide
from aoe2_ai.controllers.observation import LiveObserver
from aoe2_ai.execution import StaticGuard
from aoe2_ai.perception import Perception, load_frame
from aoe2_ai.perception.game_state import Detection, Selection, Status
from aoe2_ai.perception.regions import RegionSet
from aoe2_ai.perception.villager import (
    VillagerDetector,
    VillagerFingerprint,
    best_role,
    idle_icon_colors,
    parse_idle_count,
    view_signature,
)

from conftest import FRAMES, load_labels, requires_tesseract

LABELS = load_labels()["frames"]
IDLE_LABELLED = sorted(n for n, label in LABELS.items() if "_idle" in label)


class FileSource:
    def __init__(self, path):
        self.path = path

    def grab(self):
        return load_frame(self.path)

    def close(self):
        pass


@pytest.fixture(scope="module")
def perception():
    p = Perception()
    yield p
    p.close()


def observe(perception, name, guard=True):
    """The real LiveObserver on a saved frame (fake screen, fixed guard verdict)."""
    perception.reset()
    observer = LiveObserver(StaticGuard(guard, "test"), perception=perception, source=FileSource(FRAMES / name))
    return observer.observe()


def _check(det, want, name, field, may_miss=False):
    if want is None:
        assert det.status is not Status.VALID, f"{name} {field}: expected unknown, got {det}"
    elif may_miss and det.status is not Status.VALID:
        return
    else:
        assert det.is_valid and det.value == want, f"{name} {field}: want {want}, got {det}"


def test_every_frame_has_idle_labels():
    assert set(IDLE_LABELLED) == set(LABELS)


@requires_tesseract
@pytest.mark.parametrize("name", IDLE_LABELLED)
def test_fixture_frame(perception, name):
    label = LABELS[name]
    obs = observe(perception, name)
    _check(obs.idle.present, label["_idle"]["present"], name, "idle present")
    _check(obs.idle.count, label["_idle"]["count"], name, "idle count", "idle_count" in label.get("_ocr_misses", ()))
    _check(obs.villager.selected, label["_villager"], name, "villager selected")
    if obs.villager.is_villager:
        # Fixtures are HUD-only (play area blacked out), so the view part is empty here.
        assert obs.villager.fingerprint is not None and obs.villager.fingerprint.view is None


@requires_tesseract
def test_decisions_on_real_frames(perception):
    assert decide(observe(perception, "start_20261116_184105.png"), None).kind is DecisionKind.SELECT_IDLE_VILLAGER  # 3 idle at start
    assert decide(observe(perception, "eco_20261116_184242.png"), None).kind is DecisionKind.WAIT  # grey button: none idle
    assert decide(observe(perception, "focus_lost_dimmed_20261116_204359.png"), None).kind is DecisionKind.WAIT
    lost = observe(perception, "start_20261116_184105.png", guard=False)  # game window not in front
    assert not lost.game_detected and decide(lost, None).kind is DecisionKind.WAIT


@requires_tesseract
def test_ocr_misread_role_still_canonical(perception):
    obs = observe(perception, "idle_one_villager_selected_20261116_204145.png")
    assert obs.villager.fingerprint.role == "villageois"  # OCR reads "Villa eois"


# --- without Tesseract ------------------------------------------------------------
@pytest.fixture(scope="module")
def detector():
    return VillagerDetector()


def regions_for(image):
    return RegionSet.default().for_frame(image)


@pytest.mark.parametrize("name, expected", [
    ("start_20261116_184105.png", True),
    ("feudal_20261116_185119.png", True),
    ("eco_20261116_184242.png", False),
    ("focus_lost_dimmed_20261116_204359.png", None),
    ("main_menu_20261116_175453.png", None),
])
def test_idle_button_colour_alone(detector, name, expected):
    img = load_frame(FRAMES / name).image
    state = detector.idle(img, regions_for(img), 1.0, ocr=None)
    if expected is None:
        assert state.present.status is Status.UNKNOWN and state.count.status is Status.UNKNOWN
    else:
        assert state.present.value is expected
        # The grey button gives 0 without OCR; a yellow one needs OCR for the count.
        assert state.count.value == (None if expected else 0)


def test_build_button_separates_villager_panels(detector):
    for name, villager in [("unit_selected_20261116_184841.png", True), ("eco_20261116_203933.png", True),
                           ("feudal_20261116_185119.png", False), ("eco_20261116_184242.png", False),
                           ("idle_one_villager_selected_20261116_204145.png", False)]:  # build sub-page
        img = load_frame(FRAMES / name).image
        score = detector.build_button_score(img, regions_for(img))
        assert (score >= 0.8) is villager, (name, score)


def test_multi_selection_is_not_a_villager(detector):
    img = load_frame(FRAMES / "eco_20261116_203933.png").image
    state = detector.selection(img, regions_for(img), Detection.unknown(1.0, "no text"), 1.0)
    assert state.selected.status is Status.UNKNOWN and "multi" in state.selected.raw_text


def test_name_and_button_disagree_is_unknown(detector):
    img = load_frame(FRAMES / "unit_selected_20261116_184841.png").image  # villager command card
    state = detector.selection(img, regions_for(img), Detection.valid(Selection("Forum"), 0.9, 1.0), 1.0)
    assert state.selected.status is Status.UNKNOWN


def test_other_unit_is_not_a_villager(detector):
    img = load_frame(FRAMES / "eco_20261116_184242.png").image  # scout cavalry
    state = detector.selection(img, regions_for(img), Detection.valid(Selection("Cavalerie d'éclairage", 45, 45), 0.9, 1.0), 1.0)
    assert state.is_other


def test_villager_name_without_button_still_counts(detector):
    img = load_frame(FRAMES / "idle_one_villager_selected_20261116_204145.png").image
    state = detector.selection(img, regions_for(img), Detection.valid(Selection("Villageois", 25, 25), 0.9, 1.0), 1.0)
    assert state.is_villager and state.fingerprint.hp == 25


def test_idle_icon_colors_synthetic():
    yellow = np.zeros((27, 27, 3), np.uint8)
    yellow[:14] = (40, 200, 250)  # BGR
    grey = np.zeros((27, 27, 3), np.uint8)
    grey[:14] = (180, 180, 180)
    assert idle_icon_colors(yellow)[0] > 0.4 and idle_icon_colors(yellow)[1] == 0
    assert idle_icon_colors(grey)[1] > 0.4 and idle_icon_colors(grey)[0] == 0
    assert idle_icon_colors(np.zeros((0, 0, 3), np.uint8)) == (0.0, 0.0)


@pytest.mark.parametrize("text, value", [("3", 3), ("18", 18), ("0", 0), ("", None), ("03", None), ("a", None), ("9999", None)])
def test_parse_idle_count(text, value):
    assert parse_idle_count(text) == value


@pytest.mark.parametrize("text, role", [
    ("Villageois", "villageois"), ("Villa eois", "villageois"), ("Bucheron", "bucheron"), ("Chasseur", "chasseur"),
    ("Lumberjack", "lumberjack"), ("Forum", None), ("Cavalerie d'eclairage", None), ("", None),
])
def test_best_role(text, role):
    match = best_role(text)
    assert (match[0] if match else None) == role


def test_fingerprints():
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, (140, 160, 3), np.uint8)
    other = rng.integers(0, 255, (140, 160, 3), np.uint8)
    a = VillagerFingerprint("villageois", 25, 25, view_signature(scene))
    a2 = VillagerFingerprint("villageois", 25, 25, view_signature(scene))
    b = VillagerFingerprint("villageois", 25, 25, view_signature(other))
    assert a.same_as(a2) is True and a.same_as(b) is False
    assert a.same_as(VillagerFingerprint("bucheron", 25, 25, a.view)) is False  # different role
    assert a.same_as(VillagerFingerprint("villageois", 20, 25, a.view)) is False  # different HP
    assert a.same_as(VillagerFingerprint("villageois", 25, 25, None)) is None  # can't tell
    assert view_signature(np.zeros((140, 160, 3), np.uint8)) is None  # black screen says nothing
