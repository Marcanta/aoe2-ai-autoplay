"""OpenCV digit recognizer: segmentation, matching, validation, Tesseract fallback.

Synthetic fields are composed from *real* glyph pixels (with their anti-aliased
fringe) cut from the labelled fixture screenshots, so numbers that never
appear in the fixtures can be tested with controlled spacing, contrast and
blur. Accuracy on real screenshots is covered by the fixture tests at the end
and by ``python -m aoe2_ai.digit_benchmark``.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import pytest

from aoe2_ai.perception import Frame, Perception, RegionSet, Status
from aoe2_ai.perception.digit_recognizer import (
    CANVAS,
    DEFAULT_KINDS,
    DigitReading,
    DigitRecognizer,
    HybridOcr,
    RecognizerConfig,
    TemplateSet,
    build_templates,
    channel,
    expected_text,
    labelled_crops,
    normalize_glyph,
    segment,
)
from aoe2_ai.perception.ocr import OcrResult
from aoe2_ai.perception.profile import PerceptionProfile

from conftest import FRAMES, load_labels, requires_tesseract

LABELS = load_labels()["frames"]


@lru_cache(maxsize=None)
def recognizer() -> DigitRecognizer:
    return DigitRecognizer()


# --- glyph bank: real glyph pixels from the fixtures ------------------------------
@lru_cache(maxsize=None)
def glyph_bank(font: str) -> dict[str, tuple[np.ndarray, int]]:
    """char -> (grey patch incl. a 1 px fringe, offset of the ink top below the text-line top)."""
    cfg = RecognizerConfig()
    bank: dict[str, tuple[np.ndarray, int]] = {}
    for _, _, kind, crop, value in labelled_crops(LABELS, FRAMES):
        spec = DEFAULT_KINDS[kind]
        if spec.font != font:
            continue
        gray = channel(crop, spec.channel)
        seg, _ = segment(gray, cfg, merge_pieces=spec.merge_pieces, strict=spec.strict)
        chars = expected_text(kind, value)
        if seg is None or len(seg.boxes) != len(chars):
            continue
        top = min(b[1] for b in seg.boxes)
        for ch, (x, y, w, h) in zip(chars, seg.boxes):
            if ch not in bank:
                bank[ch] = (gray[y - 1 : y + h + 1, x - 1 : x + w + 1].copy(), y - top)
    return bank


def compose(text: str, *, font: str = "hud", gap: int = 2, size: tuple[int, int] = (53, 16), bg: int = 22,
            seed: int = 0) -> np.ndarray:
    """Grey crop of ``text`` drawn with real glyphs, ``gap`` px between the ink of neighbours
    (0 = touching, negative = overlapping). ``font="clock"`` draws ``HH:MM:SS`` on a fixed pitch."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.clip(rng.normal(bg, 3, (h, w)), 0, 255).astype(np.uint8)
    bank = glyph_bank(font)
    x, top = 4, 4
    if font == "clock":
        # fixed pitch: cells 8 px apart, a colon adds 4 px between pairs; narrow glyphs are centred
        img = np.clip(rng.normal(120, 12, (h, w)), 0, 255).astype(np.uint8)
        digits = text.replace(":", "")
        def cell(i: int) -> int:  # as measured on the fixtures: digits at x = 4, 12, 24, 32, 44, 52
            return 4 + 8 * i + 4 * (i // 2)

        for i, ch in enumerate(digits):
            patch, dy = bank[ch]
            px = cell(i) + max(0, (7 - (patch.shape[1] - 2)) // 2) - 1
            img[top + dy - 1 : top + dy - 1 + patch.shape[0], px : px + patch.shape[1]] = patch
        for i in (1, 3):  # colon dots between the pairs
            cx = cell(i) + 9
            img[7:9, cx : cx + 2] = 255
            img[11:13, cx : cx + 2] = 255
        return img
    for ch in text:
        patch, dy = bank[ch]
        ph, pw = patch.shape
        y0 = top + dy - 1
        region = img[y0 : y0 + ph, x - 1 : x - 1 + pw]
        np.maximum(region, patch[: region.shape[0], : region.shape[1]], out=region)
        x += pw - 2 + gap
    return img


def bgr(gray: np.ndarray, color: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Tint a grey text image (``color`` = B, G, R multipliers of the ink above the background)."""
    out = np.stack([gray.astype(np.float32) * c for c in color], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def read(crop: np.ndarray, kind: str = "digits", rec: DigitRecognizer | None = None) -> DigitReading:
    return (rec or recognizer()).recognize(crop, kind)


# --- templates ---------------------------------------------------------------------
def test_templates_cover_every_character_of_every_kind():
    rec = recognizer()
    for kind, spec in DEFAULT_KINDS.items():
        assert rec.supports(kind), kind
        assert set(spec.charset) <= rec.templates.chars(spec.font)
        for char in spec.charset:
            for sample in rec.templates.fonts[spec.font][char]:
                assert sample.shape == (CANVAS, CANVAS)


def test_committed_templates_match_a_rebuild_from_the_fixtures():
    rebuilt = build_templates(LABELS, FRAMES)
    committed = TemplateSet.load()
    assert {f: {c: len(s) for c, s in ch.items()} for f, ch in rebuilt.fonts.items()} == \
           {f: {c: len(s) for c, s in ch.items()} for f, ch in committed.fonts.items()}
    for font, chars in rebuilt.fonts.items():
        for char, samples in chars.items():
            for a, b in zip(samples, committed.fonts[font][char]):
                assert np.abs(a - b).max() <= 1 / 255 + 1e-6  # PNG round trip


def test_template_set_save_load_round_trip(tmp_path):
    t = recognizer().templates
    t2 = TemplateSet.load(t.save(tmp_path / "t"))
    assert t2.chars("hud") == t.chars("hud") and t2.chars("clock") == t.chars("clock")


# --- digits ----------------------------------------------------------------------------
@pytest.mark.parametrize("digit", list("0123456789"))
def test_every_single_digit(digit):
    r = read(bgr(compose(digit)))
    assert r.status is Status.VALID, r.reason
    assert r.value == int(digit) and r.text == digit
    assert 0.85 <= r.confidence <= 1.0


@pytest.mark.parametrize("text", ["10", "205", "1250", "36789", "999999", "100000", "47"])
def test_multi_digit_values(text):
    r = read(bgr(compose(text, size=(64, 16))))
    assert r.is_valid, r.reason
    assert r.value == int(text)
    assert len(r.glyphs) == len(text)


@pytest.mark.parametrize("text", ["111", "1010", "717", "1181"])
def test_different_digit_widths(text):
    r = read(bgr(compose(text)))
    assert r.value == int(text), r.reason
    for g in r.glyphs:  # the narrow 1 keeps its aspect ratio (it is not stretched to a digit's width)
        assert (g.box[2] < 0.5 * g.box[3]) == (g.char == "1")


@pytest.mark.parametrize("gap", [1, 2, 3, 4])
def test_different_spacing(gap):
    assert read(bgr(compose("2048", gap=gap, size=(64, 16)))).value == 2048


@pytest.mark.parametrize("text", ["200", "48", "905", "77"])
def test_touching_digits_are_split(text):
    r = read(bgr(compose(text, gap=0)))
    assert r.is_valid, r.reason
    assert r.value == int(text)


@pytest.mark.parametrize("text", [str(n) for n in (10, 23, 45, 67, 89, 100, 386, 512, 2794, 6150)])
def test_overlapping_digits_are_never_misread(text):
    """1 px of overlap: either split correctly or rejected, never another number."""
    r = read(bgr(compose(text, gap=-1, size=(64, 16))))
    assert r.value in (None, int(text))


@pytest.mark.parametrize("scale", [0.6, 0.45])
def test_low_contrast_digits(scale):
    gray = compose("380").astype(np.float32)
    dim = (22 + (gray - 22) * scale).clip(0, 255).astype(np.uint8)  # ink peak ~22 + 233 * scale
    r = read(bgr(dim))
    assert r.value == 380, r.reason


def test_too_low_contrast_is_unknown():
    gray = compose("380").astype(np.float32)
    dim = (22 + (gray - 22) * 0.2).astype(np.uint8)  # contrast ~47, like the dimmed (unfocused) game
    r = read(bgr(dim))
    assert r.status is Status.UNKNOWN and r.value is None
    assert "contrast" in r.reason


@pytest.mark.parametrize("sigma", [0.4, 0.6])
def test_anti_aliased_blurred_digits(sigma):
    gray = cv2.GaussianBlur(compose("1259", size=(64, 16)), (0, 0), sigma)
    r = read(bgr(gray))
    assert r.value in (None, 1259)  # never another number...
    if sigma <= 0.4:
        assert r.value == 1259, r.reason  # ...and light extra smoothing is still read


def test_tinted_digits_red_overlay():
    r = read(bgr(compose("150"), (0.75, 0.8, 1.0)))
    assert r.value == 150, r.reason


def test_leading_zero_is_invalid():
    r = read(bgr(compose("0150")))
    assert r.status is Status.UNKNOWN and "not a valid" in r.reason


# --- empty / invalid regions ---------------------------------------------------------------
@pytest.mark.parametrize("crop", [
    np.zeros((0, 0, 3), np.uint8),
    np.zeros((1, 1, 3), np.uint8),
    np.zeros((16, 53, 3), np.uint8),
    np.full((16, 53, 3), 200, np.uint8),
    np.random.default_rng(1).integers(0, 255, (16, 53, 3), dtype=np.uint8),
], ids=["empty", "1px", "black", "uniform", "noise"])
def test_empty_or_invalid_region_is_unknown(crop):
    for kind in ("digits", "fraction", "clock"):
        r = read(crop, kind)
        assert r.status is Status.UNKNOWN and r.value is None and r.reason


def test_unsupported_kind_is_unknown():
    r = read(bgr(compose("5")), "hex")
    assert r.status is Status.UNKNOWN and "unsupported" in r.reason


def test_text_regions_are_not_read_as_numbers():
    regions = RegionSet.default()
    for name in ("feudal_20261116_185119.png", "unit_selected_20261116_184841.png"):
        frame = cv2.imread(str(FRAMES / name))
        for region in ("age", "selection_name"):
            assert read(regions[region].crop(frame)).status is Status.UNKNOWN


def test_glyph_clipped_by_region_edge_is_unknown():
    gray = compose("75")
    r = read(bgr(gray[:, 6:]))  # cut through the 7
    assert r.status is Status.UNKNOWN and "clipped" in r.reason


def test_unexplained_ink_is_unknown_not_dropped():
    gray = compose("15")
    gray[8:11, 30:32] = 255  # a short blob in the text line, next to the digits
    r = read(bgr(gray))
    assert r.status is Status.UNKNOWN


def test_broken_glyph_pieces_are_merged():
    """Anti-aliasing breaks a thin 5 into stacked pieces; they are one glyph, not noise."""
    gray = compose("5/5", size=(60, 16))
    patch, _ = glyph_bank("hud")["5"]
    x0 = 4 - 1
    gray[4 + 2, x0 : x0 + patch.shape[1]] = 22  # erase one row through the first 5
    r = read(bgr(gray), "fraction")
    assert r.value in (None, (5, 5))
    assert r.text != "/5"


# --- ambiguity and confidence ----------------------------------------------------------------
def _confusable(templates: TemplateSet, a: str, b: str) -> TemplateSet:
    fonts = {f: {c: list(s) for c, s in chars.items()} for f, chars in templates.fonts.items()}
    fonts["hud"][b] = fonts["hud"][b] + fonts["hud"][a]  # b now also looks exactly like a
    return TemplateSet(fonts)


def test_ambiguous_digit_is_rejected():
    rec = DigitRecognizer(_confusable(recognizer().templates, "3", "8"))
    r = rec.recognize(bgr(compose("38")), "digits")
    assert r.status is Status.UNKNOWN and r.value is None
    assert "unknown glyphs" in r.reason


def test_blend_of_two_digits_is_not_guessed():
    a, _ = glyph_bank("hud")["3"]
    b, _ = glyph_bank("hud")["8"]
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    blend = ((a[:h, :w].astype(np.float32) + b[:h, :w]) / 2).astype(np.uint8)
    gray = np.full((16, 30), 22, np.uint8)
    gray[3 : 3 + h, 6 : 6 + w] = np.maximum(gray[3 : 3 + h, 6 : 6 + w], blend)
    assert read(bgr(gray)).value in (None, 3, 8)
    strict = DigitRecognizer(recognizer().templates, RecognizerConfig(min_margin=0.15))
    assert strict.recognize(bgr(gray), "digits").status is Status.UNKNOWN


def test_scores_equal_cv2_match_template():
    """The vectorised scorer computes TM_CCOEFF_NORMED over +-shift px, like one matchTemplate per sample."""
    rec = recognizer()
    gray = compose("2794")
    seg, _ = segment(gray, rec.config)
    for box in seg.boxes:
        canvas = normalize_glyph(seg.level, box)
        best, score, runner, second = rec.classify(canvas, "hud", "0123456789")
        padded = cv2.copyMakeBorder(canvas, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        ref = {c: max(float(cv2.matchTemplate(padded, s, cv2.TM_CCOEFF_NORMED).max()) for s in samples)
               for c, samples in rec.templates.fonts["hud"].items() if c.isdigit()}
        assert best == max(ref, key=ref.get)
        assert score == pytest.approx(ref[best], abs=1e-4) and second == pytest.approx(ref[runner], abs=1e-4)


def test_confidence_is_the_weakest_glyph_score():
    r = read(bgr(compose("2794")))
    assert r.confidence == pytest.approx(min(g.score for g in r.glyphs))
    assert all(g.margin >= RecognizerConfig().min_margin for g in r.glyphs)


def test_confidence_threshold_rejects_below_min_score():
    crop = bgr(compose("2794"))
    score = read(crop).confidence
    assert DigitRecognizer(recognizer().templates, RecognizerConfig(min_score=min(1.0, score + 0.001))).recognize(crop, "digits").status is Status.UNKNOWN
    assert DigitRecognizer(recognizer().templates, RecognizerConfig(min_score=score - 0.001)).recognize(crop, "digits").is_valid


def test_margin_threshold_rejects_close_runner_up():
    crop = bgr(compose("2794"))
    margin = min(g.margin for g in read(crop).glyphs)
    rec = DigitRecognizer(recognizer().templates, RecognizerConfig(min_margin=margin + 0.001))
    assert rec.recognize(crop, "digits").status is Status.UNKNOWN


def test_missing_template_character_disables_the_kind():
    fonts = {f: {c: list(s) for c, s in chars.items() if c != "9"} for f, chars in recognizer().templates.fonts.items()}
    rec = DigitRecognizer(TemplateSet(fonts))
    r = rec.recognize(bgr(compose("100")), "digits")  # no 9 in the text, but a 9 could not be told apart
    assert r.status is Status.UNKNOWN and "no templates" in r.reason
    assert not rec.supports("digits")


# --- population ----------------------------------------------------------------------------------
@pytest.mark.parametrize("text, value", [("7/15", (7, 15)), ("22/25", (22, 25)), ("4/5", (4, 5)), ("4/10", (4, 10)),
                                         ("125/200", (125, 200)), ("5/5", (5, 5))])
def test_population_values(text, value):
    r = read(bgr(compose(text, size=(60, 16))), "fraction")
    assert r.is_valid, r.reason
    assert r.value == value and r.text == text


@pytest.mark.parametrize("color", [(0.0, 1.0, 1.0), (0.2, 0.2, 1.0)], ids=["yellow", "red"])
def test_coloured_population(color):
    r = read(bgr(compose("22/25", size=(60, 16)), color), "fraction")
    assert r.value == (22, 25), r.reason


@pytest.mark.parametrize("text", ["45", "/5", "4/", "4//5"])
def test_malformed_population_is_unknown(text):
    r = read(bgr(compose(text, size=(60, 16))), "fraction")
    assert r.status is Status.UNKNOWN and r.value is None


def test_slash_is_not_accepted_in_a_digits_field():
    assert read(bgr(compose("4/5", size=(60, 16))), "digits").status is Status.UNKNOWN


# --- clock -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, seconds", [("00:13:22", 802), ("01:23:45", 5025), ("06:07:08", 22_028),
                                           ("00:59:29", 3569), ("00:00:11", 11), ("00:01:11", 71)])
def test_clock_formatting(text, seconds):
    r = read(bgr(compose(text, font="clock", size=(63, 17))), "clock")
    assert r.is_valid, r.reason
    assert r.text == text and r.value == seconds


def test_clock_with_out_of_range_minutes_is_unknown():
    r = read(bgr(compose("00:61:00", font="clock", size=(63, 17))), "clock")
    assert r.status is Status.UNKNOWN and "range" in r.reason


def test_clock_with_a_missing_digit_is_unknown():
    gray = compose("00:13:22", font="clock", size=(63, 17))
    gray[:, 50:] = 120  # last digit gone (and the edge with it)
    r = read(bgr(gray), "clock")
    assert r.status is Status.UNKNOWN


def test_clock_digits_must_be_grouped_in_pairs():
    gray = np.full((17, 63), 120, np.uint8)
    bank = glyph_bank("clock")
    for i, ch in enumerate("001322"):  # evenly spaced, no colons: not a clock layout
        patch, dy = bank[ch]
        x = 4 + 9 * i
        gray[3 + dy : 3 + dy + patch.shape[0], x : x + patch.shape[1]] = patch
    r = read(bgr(gray), "clock")
    assert r.status is Status.UNKNOWN and "pairs" in r.reason


# --- real screenshots ----------------------------------------------------------------------------------
REAL = [(n, f, k, c, v) for n, f, k, c, v in labelled_crops(LABELS, FRAMES)]


@pytest.mark.parametrize("name, field, kind, crop, value", REAL, ids=[f"{n[:-4]}-{f}" for n, f, *_ in REAL])
def test_real_fixture_fields(name, field, kind, crop, value):
    r = read(crop, kind)
    want = tuple(value) if kind == "fraction" else value
    assert r.is_valid, r.reason
    assert r.value == want


@pytest.mark.parametrize("name", ["main_menu_20261116_175453.png", "focus_lost_dimmed_20261116_204359.png"])
def test_real_frames_without_readable_hud_give_unknown(name):
    frame = cv2.imread(str(FRAMES / name))
    regions = RegionSet.default()
    for field, kind in (("wood", "digits"), ("population", "fraction"), ("game_time", "clock")):
        assert read(regions[field].crop(frame), kind).status is Status.UNKNOWN


@pytest.mark.parametrize("scale", [0.8, 1.25])
def test_other_resolutions_never_misread(scale):
    """Regions scale with the frame; templates are size-normalised. A read may be UNKNOWN, never wrong."""
    regions = RegionSet.default()
    name = "unit_selected_20261116_184841.png"
    frame = cv2.imread(str(FRAMES / name))
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    rs = regions.for_frame(frame)
    label = LABELS[name]
    for field, kind in (("wood", "digits"), ("food", "digits"), ("population", "fraction"), ("game_time", "clock")):
        want = tuple(label[field]) if kind == "fraction" else label[field]
        assert read(rs[field].crop(frame), kind).value in (None, want)


# --- HybridOcr: Tesseract fallback -------------------------------------------------------------------
class RecordingOcr:
    def __init__(self, text: str = "fallback", confidence: float = 0.9) -> None:
        self.text, self.confidence = text, confidence
        self.calls: list[dict] = []
        self.launches = 0

    @property
    def config_key(self):
        return ("recording",)

    def read(self, crop, *, whitelist=None, psm=7, dark=False) -> OcrResult:
        self.calls.append({"whitelist": whitelist, "psm": psm, "dark": dark, "shape": crop.shape})
        self.launches += 1
        return OcrResult(self.text, self.confidence, (self.text,))


def test_hybrid_answers_numeric_reads_without_the_fallback():
    fb = RecordingOcr()
    ocr = HybridOcr(fb, recognizer())
    r = ocr.read(bgr(compose("380")), whitelist="0123456789")
    assert (r.text, fb.calls) == ("380", [])
    assert r.confidence >= 0.85 and r.votes == ("opencv:380",)
    pop = ocr.read(bgr(compose("22/25", size=(60, 16))), whitelist="0123456789/")
    assert pop.text == "22/25"
    clock = ocr.read(bgr(compose("00:13:22", font="clock", size=(63, 17))), whitelist="0123456789:")
    assert clock.text == "00:13:22"
    assert ocr.stats.opencv == 3 and ocr.stats.fallback == 0 and ocr.launches == 0


def test_hybrid_falls_back_on_unknown_with_the_same_arguments():
    fb = RecordingOcr("123")
    ocr = HybridOcr(fb, recognizer())
    crop = np.zeros((16, 53, 3), np.uint8)
    r = ocr.read(crop, whitelist="0123456789", psm=8)
    assert r.text == "123"
    assert fb.calls == [{"whitelist": "0123456789", "psm": 8, "dark": False, "shape": crop.shape}]
    assert ocr.stats.fallback == 1 and ocr.stats.fallback_rate == 1.0 and ocr.launches == 1


@pytest.mark.parametrize("kwargs", [{"whitelist": None}, {"whitelist": "0123456789/", "dark": True}, {"whitelist": "ABC"}])
def test_hybrid_sends_unsupported_reads_to_the_fallback(kwargs):
    fb = RecordingOcr()
    ocr = HybridOcr(fb, recognizer())
    assert ocr.read(bgr(compose("380")), **kwargs).text == "fallback"
    assert len(fb.calls) == 1 and ocr.stats.unsupported == 1 and ocr.stats.opencv == 0


def test_hybrid_without_fallback_returns_empty_text_on_unknown():
    ocr = HybridOcr(None, recognizer())
    r = ocr.read(np.zeros((16, 53, 3), np.uint8), whitelist="0123456789")
    assert r.empty and r.confidence == 0.0 and "unknown" in r.votes[0]
    assert ocr.launches == 0


def test_hybrid_config_key_tracks_the_fallback_and_recognizer():
    a = HybridOcr(RecordingOcr(), recognizer())
    b = HybridOcr(RecordingOcr(), DigitRecognizer(recognizer().templates, RecognizerConfig(min_score=0.9)))
    assert a.config_key != b.config_key
    assert a.config_key[-1] == ("recording",)


@requires_tesseract
def test_hybrid_with_real_tesseract_fallback():
    from aoe2_ai.perception.ocr import TesseractOcr

    tess = TesseractOcr()
    ocr = HybridOcr(tess, recognizer())
    frame = cv2.imread(str(FRAMES / "unit_selected_20261116_184841.png"))
    regions = RegionSet.default()
    assert ocr.read(regions["food"].crop(frame), whitelist="0123456789").text == "868"
    assert tess.launches == 0
    hp = ocr.read(regions["selection_hp"].crop(frame), whitelist="0123456789/", dark=True)  # dark text: Tesseract
    assert hp.text == "25/25" and tess.launches > 0


# --- through Perception (no Tesseract needed) ---------------------------------------------------------
NUMERIC = PerceptionProfile.of("numeric", "wood", "food", "gold", "stone", "population", "game_time")


@pytest.mark.parametrize("name", sorted(LABELS))
def test_perception_with_hybrid_ocr_matches_labels(name):
    """Perception() API unchanged: only the engine differs. Without a fallback, every value comes
    from OpenCV: each VALID field equals its label, and nothing unreadable becomes a value."""
    perception = Perception(ocr=HybridOcr(None), profile=NUMERIC)
    try:
        frame = Frame(cv2.imread(str(FRAMES / name)), 5.0, name)
        state = perception.analyze(frame).state
    finally:
        perception.close()
    label = LABELS[name]
    for field in ("wood", "food", "gold", "stone", "population", "game_time"):
        det = getattr(state, field)
        if field not in label:
            continue
        want = label[field]
        if want is None:
            assert det.status is not Status.VALID, field
            continue
        assert det.status is Status.VALID, (field, det.raw_text)
        value = [det.value.current, det.value.cap] if field == "population" else det.value
        assert value == want
        assert det.timestamp == 5.0 and 0.85 <= det.confidence <= 1.0


def test_unknown_digits_never_become_zero():
    perception = Perception(ocr=HybridOcr(None), hud_check=False, profile=NUMERIC)
    try:
        state = perception.analyze(Frame(np.zeros((1050, 1680, 3), np.uint8), 1.0, "black")).state
    finally:
        perception.close()
    for field in ("wood", "food", "gold", "stone", "population", "game_time"):
        det = getattr(state, field)
        assert det.status is Status.UNKNOWN and det.value is None
