"""The controllers read numbers with HybridOcr by default; ``--ocr tesseract`` switches back."""

import argparse

import pytest

from aoe2_ai.benchmark import FileSource, check
from aoe2_ai.controllers.observation import LiveObserver
from aoe2_ai.controllers.runtime import add_common_args
from aoe2_ai.execution import StaticGuard
from aoe2_ai.perception import load_frame
from aoe2_ai.perception.digit_recognizer import OCR_ENGINES, HybridOcr, make_ocr
from aoe2_ai.perception.ocr import TesseractOcr

from conftest import FRAMES, load_labels, requires_tesseract

LABELS = load_labels()["frames"]
NUMERIC = ("wood", "food", "gold", "stone", "population", "game_time")


def observer(**kwargs) -> LiveObserver:
    return LiveObserver(StaticGuard(True, "test"), source=FileSource(), **kwargs)


def test_make_ocr():
    hybrid = make_ocr()
    assert isinstance(hybrid, HybridOcr) and isinstance(hybrid.fallback, TesseractOcr)
    assert hybrid.recognizer.supports("digits") and hybrid.recognizer.supports("clock")
    assert isinstance(make_ocr("tesseract"), TesseractOcr)
    assert set(OCR_ENGINES) == {"hybrid", "tesseract"}
    with pytest.raises(ValueError, match="unknown OCR engine"):
        make_ocr("easyocr")


def test_live_observer_defaults_to_hybrid_with_cache():
    obs = observer()
    try:
        assert isinstance(obs.perception.ocr, HybridOcr)
        assert obs.perception.cache is not None
    finally:
        obs.perception.close()


def test_live_observer_tesseract_only():
    obs = observer(ocr="tesseract")
    try:
        assert isinstance(obs.perception.ocr, TesseractOcr)
        assert obs.perception.cache is not None
    finally:
        obs.perception.close()


def test_explicit_perception_keeps_its_engine():
    from aoe2_ai.perception import Perception

    perception = Perception()
    try:
        assert observer(perception=perception, ocr="hybrid").perception.ocr is perception.ocr
        assert isinstance(perception.ocr, TesseractOcr)  # Perception() itself is unchanged
    finally:
        perception.close()


def test_cli_ocr_flag():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    assert parser.parse_args([]).ocr == "hybrid"
    assert parser.parse_args(["--ocr", "tesseract"]).ocr == "tesseract"
    with pytest.raises(SystemExit):
        parser.parse_args(["--ocr", "easyocr"])


@pytest.fixture(scope="module")
def default_observer():
    obs = observer()
    yield obs
    obs.perception.close()


@requires_tesseract
@pytest.mark.parametrize("name", sorted(LABELS))
def test_default_observer_on_real_frames(default_observer, name):
    """The controllers' default observer: every labelled field correct, nothing UNKNOWN, numbers without Tesseract."""
    default_observer.perception.reset()
    stats = default_observer.perception.ocr.stats
    fallback_before = stats.fallback
    obs = default_observer.observe_frame(load_frame(FRAMES / name, timestamp=0.0))
    verdicts = check(obs, LABELS[name])
    assert "wrong" not in verdicts.values() and "unknown" not in verdicts.values(), verdicts
    # A frame with every numeric field labelled is read without Tesseract. Elsewhere (dimmed window,
    # clock hidden at game start) OpenCV refuses the crop and Tesseract finds nothing either.
    if all(LABELS[name].get(f) is not None for f in NUMERIC):
        assert stats.fallback == fallback_before


@requires_tesseract
def test_live_metrics_record_digit_reads_fields_and_frames(tmp_path):
    from aoe2_ai.controllers.metrics import LiveMetrics

    name = "eco_20261116_184242.png"

    class Source:
        def __init__(self):
            self.t = 0.0

        def grab(self):
            self.t += 10.0
            return load_frame(FRAMES / name, timestamp=self.t)

    obs = LiveObserver(StaticGuard(True, "test"), source=Source())

    class Controller:
        observer = obs

        def cycle(self):
            self.observer.observe()
            return type("R", (), {"outcome": "waited", "decision": None, "detail": ""})()

    ctrl = Controller()
    metrics = LiveMetrics(tmp_path, frame_every=15.0)
    metrics.attach(obs, ctrl)
    try:
        for _ in range(3):
            ctrl.cycle()
    finally:
        summary = metrics.close()
        obs.perception.close()
    first, second = metrics.observations[0], metrics.observations[1]
    assert first["digits"]["opencv"] >= 6 and first["digits"]["fallback"] == 0
    assert second["digits"]["opencv"] == 0  # unchanged frame: region cache, no read at all
    assert first["fields"]["wood"]["status"] == "valid" and first["fields"]["wood"]["value"] == LABELS[name]["wood"]
    assert summary["digit_reads"]["fallback_rate"] == 0.0 and summary["numeric_not_valid_in_game"] == {}
    assert summary["frames_saved"] == 2 and (tmp_path / "frame_0001.png").exists()  # t=10 and t=30
