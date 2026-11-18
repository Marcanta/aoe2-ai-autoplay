"""OpenCV digit recognizer for the numeric HUD fields (no Tesseract launch).

The top-bar numbers of AoE2 DE (wood, food, gold, stone, population, the game
clock and the idle-villager count) are drawn in two small bitmap-like fonts.
Reading them with Tesseract costs a process launch per pass (~100-200 ms);
matching their glyphs against templates cut from real screenshots costs well
under a millisecond. This module is independent of the detectors:

* :class:`DigitRecognizer` reads one crop for one *kind* of field and returns a
  :class:`DigitReading` (text, parsed value, confidence, ``Status``, reason).
* :class:`HybridOcr` is an :class:`~aoe2_ai.perception.ocr.OcrEngine`: numeric
  reads go to the recognizer, and anything it rejects or does not support goes
  to the fallback engine (Tesseract). Pass it as ``Perception(ocr=...)``; the
  detectors, parsers and validation rules do not change.

Recognition, step by step (:meth:`DigitRecognizer.recognize`):

1. **Channel**: one grey channel of the crop (``min`` of B, G, R keeps white
   text; ``max`` also keeps the yellow/red population of a housed player).
2. **Contrast**: background = median, ink peak = 99th percentile. A crop with
   less than ``min_contrast`` between them has no readable text -> UNKNOWN
   (empty region, menu, dimmed window).
3. **Segmentation**: the crop is binarised half-way between background and
   peak, and split into connected components. Components at least 60 % of the
   tallest one are glyphs; smaller ones are merged into a glyph they overlap
   horizontally (anti-aliasing can break a stroke, e.g. the top bar of a 5)
   or else dropped as noise (background texture, the clock's colon dots).
   A glyph touching the crop edge means the text is clipped -> UNKNOWN.
4. **Splitting**: a glyph wider than any digit is two or more touching
   glyphs; it is cut at low-ink columns, and the cut is kept only if every
   part matches a template confidently.
5. **Normalisation**: each glyph is scaled to a fixed height keeping its
   aspect ratio (a "1" stays narrow) and its intensity rescaled to 0..1, so
   brightness, tint and resolution do not matter.
6. **Matching**: normalised cross-correlation (``TM_CCOEFF_NORMED``, +-2 px
   shift) against every template sample of the field's font. A glyph is
   accepted when its best score is at least ``min_score`` *and* beats the best
   score of every other character by ``min_margin``; otherwise the reading is
   ambiguous -> UNKNOWN. Nothing is guessed.
7. **Reconstruction and validation**: glyphs are joined left to right and
   checked against the field's format (no leading zero, ``a/b``, a clock of
   ``HH:MM:SS`` grouped by the gaps between digit pairs). The confidence of
   the reading is its weakest glyph's score.

Templates live in ``digit_templates/<font>/<char>_<n>.png`` (normalised glyph
images, ``slash`` for ``/``). They are cut from labelled screenshots:

    python -m aoe2_ai.perception.digit_recognizer build      # rebuild from tests/fixtures
    python -m aoe2_ai.perception.digit_recognizer read --image shot.png
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .game_state import Status
from .ocr import OcrEngine, OcrResult

TEMPLATE_DIR = Path(__file__).with_name("digit_templates")
_CHAR_NAMES = {"/": "slash", ":": "colon"}
_NAME_CHARS = {v: k for k, v in _CHAR_NAMES.items()}

CANVAS = 20  # normalised glyph height and canvas size (px)


# --- configuration -------------------------------------------------------------
@dataclass(frozen=True)
class KindSpec:
    """How to read one kind of numeric field.

    ``font``: template set; ``channel``: ``min`` (white text only) or ``max``
    (coloured text too); ``charset``: characters a glyph may be.
    ``merge_pieces``: join vertically stacked pieces of one glyph (thin
    anti-aliased fonts). ``strict``: ink in the text line that is no glyph makes
    the reading UNKNOWN; off for the clock, whose colon dots and textured
    background leave small blobs, and whose fixed six-digit layout catches a
    missing digit instead.
    """

    font: str
    channel: str
    charset: str
    merge_pieces: bool = True
    strict: bool = True


DEFAULT_KINDS: dict[str, KindSpec] = {
    "digits": KindSpec("hud", "min", "0123456789"),  # resources, idle-villager count
    "fraction": KindSpec("hud", "max", "0123456789/"),  # population: white, yellow or red
    "clock": KindSpec("clock", "min", "0123456789", merge_pieces=False, strict=False),  # colons are located by the gaps, not matched
}

# Whitelists the detectors pass to ``OcrEngine.read`` -> kind. Light text only:
# dark text on the parchment panel (``dark=True``, e.g. selection HP) is another font.
WHITELIST_KINDS: dict[str | None, str] = {
    "0123456789": "digits",
    "0123456789/": "fraction",
    "0123456789:": "clock",
}


@dataclass(frozen=True)
class RecognizerConfig:
    min_score: float = 0.85  # best template correlation a glyph needs
    min_margin: float = 0.06  # ... and its lead over the best other character
    min_contrast: float = 70.0  # ink peak - background (grey levels) for a crop to hold text
    ink_level: float = 0.5  # binarisation point between background (0) and peak (1)
    glyph_height_ratio: float = 0.6  # components at least this share of the tallest are glyphs
    min_glyph_px: int = 4  # tallest component shorter than this: no text
    max_aspect: float = 0.95  # glyph width / height above this: touching glyphs, try to split
    max_parts: int = 3  # most glyphs one touching blob may be split into
    # A piece cut out of a touching blob must be about as wide (for its height) as the character it
    # matched: its width/height within [narrowest sample * slack, widest sample / slack]. Stops a cut
    # through one digit (half an 8) from passing as a narrow 3 or 1.
    split_aspect_slack: float = 0.75
    max_glyph_overlap: int = 1  # px two neighbouring glyph boxes may share (anti-aliased edges)
    max_glyphs: int = 8
    shift: int = 2  # canvas px of shift tolerated when matching

    def key(self) -> tuple:
        return tuple(sorted(vars(self).items()))


# --- results -----------------------------------------------------------------------
@dataclass(frozen=True)
class GlyphMatch:
    char: str
    score: float  # best correlation with a template of ``char``
    runner_up: str  # best other character
    margin: float  # score - runner-up score
    box: tuple[int, int, int, int]  # x, y, w, h in the crop

    @property
    def confidence(self) -> float:
        return self.score


@dataclass(frozen=True)
class DigitReading:
    """Result of one recognition. ``value`` is None unless ``status`` is VALID."""

    status: Status
    text: str = ""  # reconstructed text when VALID, best effort otherwise (debugging)
    value: Any = None  # int (digits), (current, cap) (fraction) or seconds (clock)
    confidence: float = 0.0  # weakest glyph's score when VALID, how close a rejected read came otherwise
    reason: str = ""  # why it is UNKNOWN
    glyphs: tuple[GlyphMatch, ...] = ()
    kind: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status is Status.VALID

    @classmethod
    def unknown(cls, kind: str, reason: str, *, text: str = "", confidence: float = 0.0,
                glyphs: tuple[GlyphMatch, ...] = ()) -> "DigitReading":
        return cls(Status.UNKNOWN, text, None, confidence, reason, glyphs, kind)


# --- templates ------------------------------------------------------------------
@dataclass
class TemplateSet:
    """Per font, per character: normalised glyph samples (``CANVAS`` x ``CANVAS`` float32)."""

    fonts: dict[str, dict[str, list[np.ndarray]]] = field(default_factory=dict)

    def chars(self, font: str) -> set[str]:
        return set(self.fonts.get(font, {}))

    def add(self, font: str, char: str, canvas: np.ndarray, *, dedupe: float = 0.985, cap: int = 8) -> bool:
        """Keep ``canvas`` as a sample unless a near-identical one exists or ``char`` has ``cap`` samples."""
        samples = self.fonts.setdefault(font, {}).setdefault(char, [])
        if len(samples) >= cap or any(_correlate(canvas, s, 0) >= dedupe for s in samples):
            return False
        samples.append(canvas.astype(np.float32))
        return True

    @classmethod
    def load(cls, directory: str | Path = TEMPLATE_DIR) -> "TemplateSet":
        directory = Path(directory)
        fonts: dict[str, dict[str, list[np.ndarray]]] = {}
        for path in sorted(directory.glob("*/*.png")):
            name = path.stem.rsplit("_", 1)[0]
            char = _NAME_CHARS.get(name, name)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (CANVAS, CANVAS):
                raise ValueError(f"bad digit template {path}")
            fonts.setdefault(path.parent.name, {}).setdefault(char, []).append(image.astype(np.float32) / 255.0)
        return cls(fonts)

    def save(self, directory: str | Path = TEMPLATE_DIR) -> Path:
        directory = Path(directory)
        for font, chars in self.fonts.items():
            font_dir = directory / font
            font_dir.mkdir(parents=True, exist_ok=True)
            for old in font_dir.glob("*.png"):
                old.unlink()
            for char, samples in sorted(chars.items()):
                for i, s in enumerate(samples):
                    cv2.imwrite(str(font_dir / f"{_CHAR_NAMES.get(char, char)}_{i}.png"), np.round(s * 255).astype(np.uint8))
        return directory


# --- image helpers -----------------------------------------------------------------
def channel(crop: np.ndarray, name: str) -> np.ndarray:
    if crop.ndim == 2:
        return crop
    if name == "min":
        return crop.min(axis=2)
    if name == "max":
        return crop.max(axis=2)
    raise ValueError(f"unknown channel {name!r}")


def normalize_glyph(level: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Glyph pixels (already 0..1) scaled to height ``CANVAS``, aspect kept, centred on a square canvas."""
    x, y, w, h = box
    patch = level[y : y + h, x : x + w]
    new_w = max(1, min(CANVAS, round(w * CANVAS / h)))
    patch = cv2.resize(patch, (new_w, CANVAS), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((CANVAS, CANVAS), np.float32)
    left = (CANVAS - new_w) // 2
    canvas[:, left : left + new_w] = patch
    return canvas


def _correlate(canvas: np.ndarray, template: np.ndarray, shift: int) -> float:
    if shift:
        canvas = cv2.copyMakeBorder(canvas, shift, shift, shift, shift, cv2.BORDER_CONSTANT, value=0)
    if float(template.std()) == 0.0 or float(canvas.std()) == 0.0:
        return 0.0
    return float(cv2.matchTemplate(canvas, template, cv2.TM_CCOEFF_NORMED).max())


def _shift_windows(canvas: np.ndarray, shift: int) -> np.ndarray:
    """Every ``CANVAS`` x ``CANVAS`` window of the zero-padded canvas within ``shift`` px, as
    zero-mean unit-norm rows (a flat window becomes a zero row: correlation 0)."""
    padded = np.pad(canvas.astype(np.float32), shift)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (CANVAS, CANVAS)).reshape(-1, CANVAS * CANVAS)
    windows = windows - windows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(windows, axis=1, keepdims=True)
    return np.divide(windows, norms, out=np.zeros_like(windows), where=norms > 0)


@dataclass(frozen=True)
class _Segmentation:
    level: np.ndarray  # crop intensity rescaled to 0..1 (background 0, ink peak 1)
    mask: np.ndarray  # binarised ink
    boxes: list[tuple[int, int, int, int]]  # glyph candidates, left to right
    contrast: float


def segment(gray: np.ndarray, cfg: RecognizerConfig, *, merge_pieces: bool = True, strict: bool = True) -> tuple[_Segmentation | None, str]:
    """Glyph boxes of a grey crop, or (None, reason)."""
    if gray.size == 0 or min(gray.shape[:2]) < 3:
        return None, "empty region"
    g = gray.astype(np.float32)
    background, peak = float(np.median(g)), float(np.percentile(g, 99))
    contrast = peak - background
    if contrast < cfg.min_contrast:
        return None, f"no text (contrast {contrast:.0f})"
    level = np.clip((g - background) / contrast, 0.0, 1.0)
    mask = (level >= cfg.ink_level).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [tuple(int(v) for v in stats[i][:4]) for i in range(1, count)]
    if not comps:
        return None, "no text"
    tallest = max(h for _, _, _, h in comps)
    if tallest < cfg.min_glyph_px:
        return None, "no text (only specks)"

    if merge_pieces:
        # Anti-aliasing can break one glyph into pieces stacked vertically (the top bar of a
        # 5): join components that overlap horizontally by half the narrower one and are at
        # most a quarter of the text height apart vertically.
        comps = _merge_stacked(comps, max(1, round(0.25 * tallest)))
        tallest = max(h for _, _, _, h in comps)

    glyph_min = cfg.glyph_height_ratio * tallest
    tall = sorted(list(c) for c in comps if c[3] >= glyph_min)
    small = [c for c in comps if c[3] < glyph_min]
    height, width = mask.shape
    for x, y, w, h in tall:
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            return None, "text clipped by the region edge"

    top = min(b[1] for b in tall)
    bottom = max(b[1] + b[3] for b in tall)
    tiny = max(2, round(0.03 * tallest * tallest))
    for x, y, w, h in small:
        outside_band = y + h <= top or y >= bottom
        overlap = max((min(mx + mw, x + w) - max(mx, x) for mx, _, mw, _ in tall), default=0)
        if not outside_band and overlap >= 0.5 * w:
            i = max(range(len(tall)), key=lambda k: min(tall[k][0] + tall[k][2], x + w) - max(tall[k][0], x))
            tall[i] = _union(tall[i], (x, y, w, h))  # a detached accent of that glyph
        elif strict and not outside_band and w * h > tiny:
            # Ink in the text line that is no glyph: never drop it silently (a dropped
            # broken digit would turn "15" into "1").
            return None, f"unexplained ink at x={x}"
    line_top, line_bottom = _band(tall)
    boxes = [_trim_protrusions(mask, b, line_top, line_bottom) for b in tall]
    return _Segmentation(level, mask, boxes, contrast), ""


def _band(boxes) -> tuple[int, int]:
    """Median top and bottom (exclusive) of the glyph boxes: the text line."""
    return int(np.median([b[1] for b in boxes])), int(np.median([b[1] + b[3] for b in boxes]))


def _trim_protrusions(mask: np.ndarray, box, top: int, bottom: int) -> tuple[int, int, int, int]:
    """Drop rows above/below the text line holding a single ink pixel: a speck of background
    texture touching the glyph, which would otherwise squash it when normalised."""
    x, y, w, h = box
    y0, y1 = y, y + h
    while y0 < top and mask[y0, x : x + w].sum() <= 1:
        y0 += 1
    while y1 > bottom and mask[y1 - 1, x : x + w].sum() <= 1:
        y1 -= 1
    return x, y0, w, y1 - y0


def _merge_stacked(comps: list[tuple[int, int, int, int]], max_gap: int) -> list[tuple[int, int, int, int]]:
    boxes = [list(c) for c in comps]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                h_overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                v_gap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                if h_overlap >= 0.5 * min(a[2], b[2]) and v_gap <= max_gap:
                    boxes[i] = _union(a, b)
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def _union(a, b) -> list[int]:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def _trim(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> tuple[int, int, int, int] | None:
    """Tight box of the ink in mask[y0:y1, x0:x1]."""
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    return x0 + int(xs.min()), y0 + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


# --- recognizer ----------------------------------------------------------------
_FORMATS = {
    "digits": re.compile(r"^(0|[1-9]\d{0,5})$"),
    "fraction": re.compile(r"^(0|[1-9]\d{0,3})/(0|[1-9]\d{0,3})$"),
}


class DigitRecognizer:
    """Template-matching reader of numeric HUD crops. Thread-safe; create once and reuse."""

    def __init__(self, templates: TemplateSet | None = None, config: RecognizerConfig = RecognizerConfig(),
                 kinds: dict[str, KindSpec] | None = None) -> None:
        self.templates = templates if templates is not None else TemplateSet.load()
        self.config = config
        self.kinds = dict(kinds or DEFAULT_KINDS)
        self._matrices: dict[str, tuple[list[str], np.ndarray]] = {}  # per font, see _matrix
        self._aspects: dict[str, dict[str, tuple[float, float]]] = {}  # per font, see _aspect_range

    @property
    def config_key(self) -> tuple:
        sizes = tuple(sorted((f, c, len(s)) for f, chars in self.templates.fonts.items() for c, s in chars.items()))
        return ("digits", self.config.key(), tuple(sorted(self.kinds.items())), sizes, id(self.templates))

    def supports(self, kind: str) -> bool:
        spec = self.kinds.get(kind)
        return spec is not None and set(spec.charset) <= self.templates.chars(spec.font)

    # public ------------------------------------------------------------------------
    def recognize(self, crop: np.ndarray, kind: str) -> DigitReading:
        """Read ``crop`` (BGR or grey) as a field of ``kind`` (``digits``, ``fraction`` or ``clock``)."""
        spec = self.kinds.get(kind)
        if spec is None:
            return DigitReading.unknown(kind, f"unsupported kind {kind!r}")
        if not self.supports(kind):
            missing = sorted(set(spec.charset) - self.templates.chars(spec.font))
            return DigitReading.unknown(kind, f"no templates for {''.join(missing)!r} in font {spec.font!r}")
        if crop is None or crop.size == 0:
            return DigitReading.unknown(kind, "empty region")

        seg, problem = segment(channel(crop, spec.channel), self.config, merge_pieces=spec.merge_pieces, strict=spec.strict)
        if seg is None:
            return DigitReading.unknown(kind, problem)
        if len(seg.boxes) > self.config.max_glyphs:
            return DigitReading.unknown(kind, f"{len(seg.boxes)} glyphs: not a {kind} field")

        glyphs: list[GlyphMatch] = []
        for box in seg.boxes:
            parts = self._read_blob(seg, box, spec)
            if parts is None:
                return DigitReading.unknown(kind, "touching or unknown glyphs", text="".join(g.char for g in glyphs) + "?",
                                            glyphs=tuple(glyphs))
            glyphs.extend(parts)
        # Glyphs on one line do not overlap. Two that share columns are one glyph counted twice
        # (a digit broken into pieces that joined both neighbours), never two characters.
        boxes = sorted(g.box for g in glyphs)
        for a, b in zip(boxes, boxes[1:]):
            if a[0] + a[2] - b[0] > self.config.max_glyph_overlap:
                return DigitReading.unknown(kind, "overlapping glyphs", text="".join(g.char for g in glyphs),
                                            glyphs=tuple(glyphs))
        return self._assemble(kind, glyphs)

    def classify(self, canvas: np.ndarray, font: str, charset: str) -> tuple[str, float, str, float]:
        """(best char, its score, runner-up char, its score) of a normalised glyph.

        Score = ``TM_CCOEFF_NORMED`` (zero-mean normalised cross-correlation) of the
        template with the glyph, best over shifts of up to ``config.shift`` px. All
        samples and shifts are scored in one matrix product instead of one
        ``cv2.matchTemplate`` call per sample (same values, no per-call overhead).
        """
        chars, matrix = self._matrix(font)
        windows = _shift_windows(canvas, self.config.shift)  # (shifts, CANVAS*CANVAS), zero-mean, unit norm
        scores = (windows @ matrix.T).max(axis=0) if len(chars) else np.zeros(0)
        best: dict[str, float] = {}
        for char, score in zip(chars, scores.tolist()):
            if char in charset and score > best.get(char, -1.0):
                best[char] = score
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True) + [("", 0.0)]
        (c1, s1), (c2, s2) = ranked[0], ranked[1]
        return c1, max(0.0, s1), c2, max(0.0, s2)

    def _matrix(self, font: str) -> tuple[list[str], np.ndarray]:
        """Template samples of ``font`` as rows (zero-mean, unit norm) and their characters, built once."""
        cached = self._matrices.get(font)
        if cached is None:
            chars, rows = [], []
            for char, samples in sorted(self.templates.fonts.get(font, {}).items()):
                for s in samples:
                    v = s.astype(np.float32).ravel()
                    v = v - v.mean()
                    norm = float(np.linalg.norm(v))
                    if norm > 0:
                        chars.append(char)
                        rows.append(v / norm)
            matrix = np.stack(rows) if rows else np.zeros((0, CANVAS * CANVAS), np.float32)
            cached = self._matrices.setdefault(font, (chars, matrix))
        return cached

    # internals -------------------------------------------------------------------------
    def _match(self, seg: _Segmentation, box, spec: KindSpec) -> GlyphMatch:
        char, score, runner, second = self.classify(normalize_glyph(seg.level, box), spec.font, spec.charset)
        return GlyphMatch(char, score, runner, score - second, box)

    def _accepted(self, g: GlyphMatch) -> bool:
        return g.score >= self.config.min_score and g.margin >= self.config.min_margin

    def _aspect_range(self, font: str) -> dict[str, tuple[float, float]]:
        """Per character: (narrowest, widest) width/height of its template samples, built once.

        A sample's ink spans the full canvas height, so its width/height is the share of
        canvas columns holding ink (normalisation keeps the glyph's aspect).
        """
        cached = self._aspects.get(font)
        if cached is None:
            ranges: dict[str, tuple[float, float]] = {}
            for char, samples in self.templates.fonts.get(font, {}).items():
                widths = [int((s.max(axis=0) > 0.25).sum()) / CANVAS for s in samples]
                if widths:
                    ranges[char] = (min(widths), max(widths))
            cached = self._aspects.setdefault(font, ranges)
        return cached

    def _plausible_part(self, g: GlyphMatch, font: str) -> bool:
        """A glyph cut out of a touching blob has the width/height of the character it matched."""
        lo, hi = self._aspect_range(font).get(g.char, (0.0, float("inf")))
        _, _, w, h = g.box
        slack = self.config.split_aspect_slack
        return lo * slack <= w / h <= hi / slack

    def _accepted_part(self, g: GlyphMatch, font: str) -> bool:
        return self._accepted(g) and self._plausible_part(g, font)

    def _read_blob(self, seg: _Segmentation, box, spec: KindSpec) -> list[GlyphMatch] | None:
        """One glyph, or several when the blob is touching glyphs; None if no confident reading."""
        x, y, w, h = box
        if w <= self.config.max_aspect * h:
            g = self._match(seg, box, spec)
            if self._accepted(g):
                return [g]
            # A narrow blob that does not match is unreadable (splitting it would only invent glyphs).
            return None
        return self._split(seg, box, spec, self.config.max_parts)

    def _split(self, seg: _Segmentation, box, spec: KindSpec, parts_left: int) -> list[GlyphMatch] | None:
        x, y, w, h = box
        if w <= self.config.max_aspect * h:
            g = self._match(seg, box, spec)
            return [g] if self._accepted_part(g, spec.font) else None
        if parts_left < 2:
            return None
        min_w = max(1, round(0.25 * h))
        ink = seg.mask[y : y + h, x : x + w].sum(axis=0)
        cuts = sorted(range(min_w, w - min_w + 1), key=lambda c: (ink[c] if c < w else 0, abs(c - w / 2)))[:4]
        best: list[GlyphMatch] | None = None
        for c in cuts:
            left = _trim(seg.mask, x, x + c, y, y + h)
            right = _trim(seg.mask, x + c, x + w, y, y + h)
            if left is None or right is None or left[2] > self.config.max_aspect * h:
                continue
            g = self._match(seg, left, spec)
            if not self._accepted_part(g, spec.font):
                continue
            rest = self._split(seg, right, spec, parts_left - 1)
            if rest is None:
                continue
            candidate = [g, *rest]
            if best is None or min(m.score for m in candidate) > min(m.score for m in best):
                best = candidate
        return best

    def _assemble(self, kind: str, glyphs: list[GlyphMatch]) -> DigitReading:
        text = "".join(g.char for g in glyphs)
        confidence = min((g.score for g in glyphs), default=0.0)
        glyph_t = tuple(glyphs)
        if not glyphs:
            return DigitReading.unknown(kind, "no glyphs")
        # Glyphs of one field share a height: a much shorter/taller one is not text of this font.
        digit_heights = [g.box[3] for g in glyphs if g.char.isdigit()]
        if digit_heights and max(digit_heights) > 1.3 * min(digit_heights):
            return DigitReading.unknown(kind, "inconsistent glyph heights", text=text, confidence=confidence, glyphs=glyph_t)

        if kind == "clock":
            text, problem = _clock_text(glyphs)
            if problem:
                return DigitReading.unknown(kind, problem, text=text, confidence=confidence, glyphs=glyph_t)
            h, m, s = (int(p) for p in text.split(":"))
            return DigitReading(Status.VALID, text, h * 3600 + m * 60 + s, confidence, "", glyph_t, kind)

        m = _FORMATS[kind].match(text)
        if not m:
            return DigitReading.unknown(kind, f"{text!r} is not a valid {kind} value", text=text, confidence=confidence, glyphs=glyph_t)
        value: Any = int(text) if kind == "digits" else (int(m.group(1)), int(m.group(2)))
        return DigitReading(Status.VALID, text, value, confidence, "", glyph_t, kind)


def _clock_text(glyphs: list[GlyphMatch]) -> tuple[str, str]:
    """``HH:MM:SS`` (6 digits) or ``MM:SS`` (4) from digit glyphs grouped in pairs by the gaps between them."""
    digits = "".join(g.char for g in glyphs)
    if len(glyphs) not in (4, 6):
        return digits, f"clock has {len(glyphs)} digits"
    # The clock is drawn on a fixed pitch (a narrow "1" is centred in a full-width cell), so
    # digit centres, not edges, show the pairs: a colon widens the step between two pairs.
    centres = [g.box[0] + g.box[2] / 2 for g in glyphs]
    steps = [b - a for a, b in zip(centres, centres[1:])]
    inner, outer = steps[0::2], steps[1::2]
    height = max(g.box[3] for g in glyphs)
    if min(outer) < max(inner) + max(2, 0.2 * height):
        return digits, "clock digits not grouped in pairs"
    pairs = [digits[i : i + 2] for i in range(0, len(digits), 2)]
    if len(pairs) == 2:
        pairs.insert(0, "00")
    if int(pairs[1]) >= 60 or int(pairs[2]) >= 60:
        return ":".join(pairs), "clock minutes/seconds out of range"
    return ":".join(pairs), ""


# --- OCR engine adapter --------------------------------------------------------------
@dataclass
class HybridStats:
    opencv: int = 0  # numeric reads answered by the recognizer
    fallback: int = 0  # numeric reads the recognizer rejected, sent to the fallback engine
    unsupported: int = 0  # reads the recognizer does not handle (text, dark panel text), sent to the fallback

    @property
    def fallback_rate(self) -> float:
        numeric = self.opencv + self.fallback
        return self.fallback / numeric if numeric else 0.0


class HybridOcr:
    """OCR engine: OpenCV digits first, ``fallback`` (Tesseract) for everything else.

    Numeric whitelists (see :data:`WHITELIST_KINDS`) on light text are read by
    the :class:`DigitRecognizer`. A VALID reading is returned as an ``OcrResult``
    with the reading's confidence; an UNKNOWN reading is not returned: the crop
    is read again by ``fallback`` exactly as without this engine. With
    ``fallback=None`` the UNKNOWN reading comes back as empty text.
    """

    def __init__(self, fallback: OcrEngine | None, recognizer: DigitRecognizer | None = None,
                 *, whitelist_kinds: dict[str | None, str] | None = None) -> None:
        self.fallback = fallback
        self.recognizer = recognizer or DigitRecognizer()
        self.whitelist_kinds = dict(WHITELIST_KINDS if whitelist_kinds is None else whitelist_kinds)
        self.stats = HybridStats()
        self._lock = threading.Lock()

    @property
    def config_key(self) -> tuple:
        fb = getattr(self.fallback, "config_key", None) if self.fallback is not None else None
        fb = fb if fb is not None else ("engine", id(self.fallback))
        return ("hybrid", self.recognizer.config_key, tuple(sorted(self.whitelist_kinds.items(), key=str)), fb)

    @property
    def launches(self) -> int | None:
        """Tesseract launches of the fallback (for ``PerceptionResult.ocr_launches``)."""
        n = getattr(self.fallback, "launches", None)
        return n if isinstance(n, int) else (0 if self.fallback is None else None)

    def _count(self, name: str) -> None:
        with self._lock:
            setattr(self.stats, name, getattr(self.stats, name) + 1)

    def read(self, crop: np.ndarray, *, whitelist: str | None = None, psm: int = 7, dark: bool = False) -> OcrResult:
        kind = None if dark else self.whitelist_kinds.get(whitelist)
        if kind is None or not self.recognizer.supports(kind):
            self._count("unsupported")
            return self._fallback(crop, whitelist, psm, dark, None)
        reading = self.recognizer.recognize(crop, kind)
        if reading.is_valid:
            self._count("opencv")
            return OcrResult(reading.text, reading.confidence, (f"opencv:{reading.text}",))
        self._count("fallback")
        return self._fallback(crop, whitelist, psm, dark, reading)

    def _fallback(self, crop, whitelist, psm, dark, reading: DigitReading | None) -> OcrResult:
        if self.fallback is not None:
            return self.fallback.read(crop, whitelist=whitelist, psm=psm, dark=dark)
        note = f"opencv:unknown ({reading.reason})" if reading is not None else "opencv:unsupported"
        return OcrResult("", reading.confidence if reading is not None else 0.0, (note,))


# --- template building ----------------------------------------------------------------
OCR_ENGINES = ("hybrid", "tesseract")


def make_ocr(name: str = "hybrid", *, memo: int = 0) -> OcrEngine:
    """The OCR engine called ``name``.

    ``hybrid`` (the controllers' default): OpenCV for the numeric HUD fields, Tesseract for
    everything else and for any number OpenCV is unsure of. ``tesseract``: Tesseract only.
    ``memo``: how many Tesseract pass results to remember by binarised image (0 = off,
    see ``TesseractOcr._read_once``).
    """
    from .ocr import TesseractOcr

    if name == "hybrid":
        return HybridOcr(TesseractOcr(memo=memo))
    if name == "tesseract":
        return TesseractOcr(memo=memo)
    raise ValueError(f"unknown OCR engine {name!r} (expected one of {', '.join(OCR_ENGINES)})")


def expected_text(kind: str, value: Any) -> str:
    """Characters a field of ``kind`` shows for a labelled ``value`` (clock: its six digits)."""
    if kind == "digits":
        return str(int(value))
    if kind == "fraction":
        return f"{int(value[0])}/{int(value[1])}"
    if kind == "clock":
        h, rest = divmod(int(value), 3600)
        return f"{h:02d}{rest // 60:02d}{rest % 60:02d}"
    raise ValueError(kind)


def harvest(recognizer_config: RecognizerConfig, templates: TemplateSet, crop: np.ndarray, kind: str, value: Any,
            kinds: dict[str, KindSpec] = DEFAULT_KINDS) -> int:
    """Add the glyphs of a labelled crop to ``templates``; only when segmentation finds exactly one glyph
    per expected character. Returns the number of new samples (0 when the crop was skipped)."""
    spec = kinds[kind]
    seg, _ = segment(channel(crop, spec.channel), recognizer_config, merge_pieces=spec.merge_pieces, strict=spec.strict)
    chars = expected_text(kind, value)
    if seg is None or len(seg.boxes) != len(chars):
        return 0
    if any(w > recognizer_config.max_aspect * h for _, _, w, h in seg.boxes):
        return 0
    return sum(templates.add(spec.font, ch, normalize_glyph(seg.level, box)) for ch, box in zip(chars, seg.boxes))


def labelled_crops(labels: dict, frames_dir: Path, regions=None) -> Iterable[tuple[str, str, str, np.ndarray, Any]]:
    """(frame, field, kind, crop, value) for every numeric value in ``labels.json``'s frames.

    Crops are what the detectors pass to the OCR engine (the idle count goes
    through ``white_only`` like in ``VillagerDetector``).
    """
    from .regions import RegionSet
    from .villager import white_only

    regions = regions or RegionSet.default()
    for name, label in sorted(labels.items()):
        frame = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        rs = regions.for_frame(frame)
        if label.get("hud", True) is False:
            continue
        for fname, kind in (("wood", "digits"), ("food", "digits"), ("gold", "digits"), ("stone", "digits"),
                            ("population", "fraction"), ("game_time", "clock")):
            if label.get(fname) is not None:
                yield name, fname, kind, rs[fname].crop(frame), label[fname]
        idle = label.get("_idle") or {}
        if idle.get("present") and idle.get("count"):
            yield name, "idle_count", "digits", white_only(rs["idle_count"].crop(frame)), idle["count"]


def extra_crops(path: str | Path) -> Iterable[tuple[str, str, str, np.ndarray, Any]]:
    """(file, field, kind, crop, value) for every crop listed in a ``crops.json``.

    ``crops.json`` holds ``{"crops": {file: {"field", "kind", "value", ...}}}`` next to the
    crop PNGs: raw region crops with checked values, for screens that are not worth a full
    labelled frame (``tests/fixtures/live_crops``). The idle count goes through ``white_only``
    like in ``labelled_crops``. A missing file yields nothing.
    """
    from .villager import white_only

    path = Path(path)
    if not path.is_file():
        return
    entries = json.loads(path.read_text(encoding="utf-8"))["crops"]
    for name, e in sorted(entries.items()):
        crop = cv2.imread(str(path.parent / name), cv2.IMREAD_COLOR)
        if crop is None:
            raise ValueError(f"missing crop {path.parent / name}")
        value = tuple(e["value"]) if e["kind"] == "fraction" else e["value"]
        yield name, e["field"], e["kind"], white_only(crop) if e["field"] == "idle_count" else crop, value


def build_templates(labels: dict, frames_dir: Path, *, config: RecognizerConfig = RecognizerConfig(),
                    exclude: Iterable[str] = (),
                    extra: Iterable[tuple[str, str, str, np.ndarray, Any]] = ()) -> TemplateSet:
    """Templates cut from every labelled numeric field, then from the ``extra`` crops
    (see :func:`extra_crops`). ``exclude``: frame or crop names to leave out."""
    excluded = set(exclude)
    templates = TemplateSet()
    for name, _, kind, crop, value in [*labelled_crops(labels, frames_dir), *extra]:
        if name not in excluded:
            harvest(config, templates, crop, kind, value)
    return templates


# --- CLI ------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    parser = argparse.ArgumentParser(description="OpenCV digit recognizer for the AoE2 HUD numbers.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="cut templates from labelled screenshots")
    b.add_argument("--labels", type=Path, default=fixtures / "labels.json")
    b.add_argument("--frames", type=Path, default=fixtures / "frames")
    b.add_argument("--extra-crops", type=Path, default=fixtures / "live_crops" / "crops.json",
                   help="crops.json of extra labelled crops (default: the live-game crops; a missing file is skipped)")
    b.add_argument("--out", type=Path, default=TEMPLATE_DIR)
    r = sub.add_parser("read", help="read the numeric HUD fields of a screenshot")
    r.add_argument("--image", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        labels = json.loads(args.labels.read_text(encoding="utf-8"))["frames"]
        templates = build_templates(labels, args.frames, extra=extra_crops(args.extra_crops))
        templates.save(args.out)
        for font, chars in sorted(templates.fonts.items()):
            print(f"{font}: " + "  ".join(f"{c!s}x{len(s)}" for c, s in sorted(chars.items())))
        print(f"wrote {args.out}")
    else:
        from .regions import RegionSet
        from .villager import white_only

        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"cannot read {args.image}")
        rs = RegionSet.default().for_frame(frame)
        recognizer = DigitRecognizer()
        for name, kind in (("wood", "digits"), ("food", "digits"), ("gold", "digits"), ("stone", "digits"),
                           ("population", "fraction"), ("game_time", "clock"), ("idle_count", "digits")):
            crop = rs[name].crop(frame)
            reading = recognizer.recognize(white_only(crop) if name == "idle_count" else crop, kind)
            print(f"{name:<11} {reading.status.value:<8} {reading.text!r:<12} value={reading.value!r:<12} "
                  f"conf={reading.confidence:.3f} {reading.reason}")


if __name__ == "__main__":
    main()
