"""Tesseract pass memo: a pass whose binarised image was already read is not launched again."""

import numpy as np
import pytest

import aoe2_ai.perception.ocr as ocr_module
from aoe2_ai.controllers.observation import OCR_MEMO, LiveObserver
from aoe2_ai.execution import StaticGuard
from aoe2_ai.perception import load_frame
from aoe2_ai.perception.digit_recognizer import make_ocr
from aoe2_ai.perception.metrics import ocr_memo_hits
from aoe2_ai.perception.ocr import TesseractOcr, preprocess
from aoe2_ai.perception.regions import RegionSet

from conftest import FRAMES, requires_tesseract


def text_crop(value: int = 200) -> np.ndarray:
    """A light 'glyph' (two bars) on a dark background, HUD-like."""
    crop = np.full((14, 40, 3), 30, np.uint8)
    crop[3:11, 8:11] = value
    crop[3:11, 20:23] = value
    return crop


@pytest.fixture
def fake_tesseract(monkeypatch):
    """TesseractOcr whose launches are recorded instead of run (no binary needed)."""
    monkeypatch.setattr(ocr_module, "find_tesseract", lambda cmd=None: "tesseract")

    def make(**kwargs) -> TesseractOcr:
        engine = TesseractOcr(**kwargs)
        engine.seen = []

        def launch(image, config):
            engine.launches += 1
            engine.seen.append((image.tobytes(), config))
            return f"t{int((image < 128).sum())}", 0.9  # text depends only on the binarised image

        engine._launch = launch
        return engine

    return make


def test_memo_off_by_default(fake_tesseract):
    engine = fake_tesseract()
    engine.read(text_crop())
    first = engine.launches
    engine.read(text_crop())
    assert first >= 2 and engine.launches == 2 * first and engine.memo_hits == 0


def test_identical_image_is_not_launched_again_and_gives_the_same_result(fake_tesseract):
    engine = fake_tesseract(memo=16)
    a = engine.read(text_crop())
    launches, hits = engine.launches, engine.memo_hits  # thresholds giving the same image already hit
    b = engine.read(text_crop())
    assert engine.launches == launches and engine.memo_hits == hits + len(b.votes)
    assert (a.text, a.confidence, a.votes) == (b.text, b.confidence, b.votes)


def test_grey_level_noise_that_does_not_change_the_binarised_image_hits(fake_tesseract):
    engine = fake_tesseract(memo=16)
    crop = text_crop()
    noisy = crop.copy()
    noisy[0, 0] += 1  # background pixel, far from every threshold
    noisy[5, 9] -= 1  # glyph pixel, far from every threshold
    assert not np.array_equal(crop, noisy)
    for p in engine.passes:
        assert np.array_equal(preprocess(crop, p, scale=engine.scale), preprocess(noisy, p, scale=engine.scale))
    a = engine.read(crop)
    launches = engine.launches
    assert engine.read(noisy) == a and engine.launches == launches


def test_changed_text_or_config_misses(fake_tesseract):
    engine = fake_tesseract(memo=16)
    engine.read(text_crop())
    launches = engine.launches
    other = text_crop()
    other[3:11, 30:33] = 200  # a third bar: different binarised image
    engine.read(other)
    assert engine.launches > launches
    launches = engine.launches
    engine.read(text_crop(), whitelist="0123456789")  # same image, different Tesseract config
    assert engine.launches > launches


def test_memo_keeps_only_the_most_recent_entries(fake_tesseract):
    engine = fake_tesseract(memo=2, passes=(ocr_module.Pass("min", 170),), early_stop=2)
    crops = [text_crop(), text_crop(), text_crop()]
    crops[1][3:11, 30:33] = 200
    crops[2][3:11, 2:4] = 200
    for c in crops:
        engine.read(c)
    assert len(engine._memo) == 2 and engine.launches == 3
    engine.read(crops[0])  # evicted: launched again
    assert engine.launches == 4
    engine.read(crops[2])  # still remembered
    assert engine.launches == 4


def test_make_ocr_memo_and_observer_default(fake_tesseract):
    assert make_ocr("tesseract").memo_size == 0  # benchmarks time real launches
    assert make_ocr("hybrid").fallback.memo_size == 0
    assert make_ocr("tesseract", memo=8).memo_size == 8
    assert make_ocr("hybrid", memo=8).fallback.memo_size == 8
    obs = LiveObserver(StaticGuard(True, "test"), source=object())
    try:
        assert obs.perception.ocr.fallback.memo_size == OCR_MEMO > 0
        assert ocr_memo_hits(obs.perception.ocr) == 0
    finally:
        obs.perception.close()
    assert ocr_memo_hits(object()) is None


@requires_tesseract
def test_selection_read_with_pixel_noise_skips_tesseract_and_reads_the_same():
    """Real frame: the selected unit's name and HP, re-read with 1-grey-level noise (as on screen
    between two observations), give the same text with no new Tesseract launch."""
    frame = load_frame(FRAMES / "eco_20261116_184242.png").image
    regions = RegionSet.default().for_frame(frame)
    rng = np.random.default_rng(0)
    plain, memo = TesseractOcr(), TesseractOcr(memo=64)
    for name, whitelist in (("selection_name", None), ("selection_hp", "0123456789/")):
        region = regions[name]
        crop = region.crop(frame)
        noise = rng.integers(-1, 2, crop.shape)
        noisy = np.clip(crop.astype(int) + noise, 0, 255).astype(np.uint8)
        assert not np.array_equal(crop, noisy)
        expected = plain.read(noisy, whitelist=whitelist, dark=region.dark_text)
        memo.read(crop, whitelist=whitelist, dark=region.dark_text)
        before = memo.launches
        got = memo.read(noisy, whitelist=whitelist, dark=region.dark_text)
        assert (got.text, got.confidence, got.votes) == (expected.text, expected.confidence, expected.votes)
        if all(np.array_equal(preprocess(crop, p), preprocess(noisy, p)) for p in (memo.dark_passes if region.dark_text else memo.passes)):
            assert memo.launches == before
