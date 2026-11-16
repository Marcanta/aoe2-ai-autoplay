"""Detectors: region crop -> OCR -> parse/validate -> Detection.

Parsing is done by pure functions (``parse_*``, ``match_age``) so they can be
tested without Tesseract. Any read that fails parsing or validation, or whose
confidence is below ``min_confidence``, becomes ``UNKNOWN``: an unreadable
resource is never reported as 0.
"""

from __future__ import annotations

import re
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from .game_state import Age, Detection, Population, Selection
from .metrics import StageTimer
from .ocr import OcrEngine, OcrResult
from .regions import Region, RegionSet

MAX_RESOURCE = 999_999
MAX_POPULATION = 1000
DEFAULT_MIN_CONFIDENCE = 0.6

# Age names as shown in the HUD. Add other UI languages here.
AGE_NAMES: dict[Age, tuple[str, ...]] = {
    Age.DARK: ("Dark Age", "Âge sombre"),
    Age.FEUDAL: ("Feudal Age", "Âge féodal"),
    Age.CASTLE: ("Castle Age", "Âge des châteaux"),
    Age.IMPERIAL: ("Imperial Age", "Âge impérial"),
}

ANCHOR_DIR = Path(__file__).with_name("anchors")


# --- pure parsers ----------------------------------------------------------
_DIGITS = re.compile(r"^\d{1,6}$")
_FRACTION = re.compile(r"^(\d{1,4})/(\d{1,4})$")
_CLOCK = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})$")


def parse_resource(text: str) -> int | None:
    """``"1250"`` -> 1250. Empty, non-numeric, or leading-zero reads -> None (never 0 by default)."""
    text = text.replace(" ", "")
    if not _DIGITS.match(text) or (len(text) > 1 and text[0] == "0"):
        return None
    value = int(text)
    return value if value <= MAX_RESOURCE else None


def parse_fraction(text: str) -> tuple[int, int] | None:
    m = _FRACTION.match(text.replace(" ", ""))
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if any(len(s) > 1 and s[0] == "0" for s in (a, b)):
        return None
    return int(a), int(b)


def parse_population(text: str) -> Population | None:
    parsed = parse_fraction(text)
    if parsed is None:
        return None
    current, cap = parsed
    # Current can exceed cap (houses destroyed), but both stay bounded.
    if cap < 1 or cap > MAX_POPULATION or current > MAX_POPULATION:
        return None
    return Population(current, cap)


def parse_clock(text: str) -> int | None:
    """``"00:12:34"`` or ``"12:34"`` -> seconds."""
    m = _CLOCK.match(text.replace(" ", ""))
    if not m:
        return None
    hours = int(m.group(1) or 0)
    minutes, seconds = int(m.group(2)), int(m.group(3))
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())


def match_age(text: str, *, min_ratio: float = 0.75, min_margin: float = 0.1) -> tuple[Age, float] | None:
    """Fuzzy-match OCR text to an age name. Returns (age, similarity) or None if ambiguous."""
    norm = _normalize(text)
    if not norm:
        return None
    scores: dict[Age, float] = {}
    for age, names in AGE_NAMES.items():
        best = 0.0
        for name in names:
            n = _normalize(name)
            best = max(best, 1.0 if n in norm else SequenceMatcher(None, norm, n).ratio())
        scores[age] = best
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (age, top), (_, second) = ranked[0], ranked[1]
    if top < min_ratio or top - second < min_margin:
        return None
    return age, top


# --- detectors -------------------------------------------------------------
@dataclass(frozen=True)
class RegionReading:
    """What one region produced, for debug display."""

    region: Region
    ocr: OcrResult | None
    note: str = ""


@dataclass
class DetectorContext:
    frame: np.ndarray
    regions: RegionSet  # already scaled to the frame
    ocr: OcrEngine
    timestamp: float
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    timer: StageTimer | None = None

    def stage(self, name: str):
        """Time a block under ``name`` when a timer is attached."""
        return self.timer.stage(name) if self.timer is not None else nullcontext()


class Detector:
    """Base class: one GameState field, read from one or more regions."""

    field: str = ""

    def detect(self, ctx: DetectorContext) -> tuple[Detection, list[RegionReading]]:
        raise NotImplementedError

    @staticmethod
    def _read(ctx: DetectorContext, name: str, whitelist: str | None, psm: int = 7) -> tuple[RegionReading, str | None]:
        """OCR one region. Returns (reading, reason it can't be used or None)."""
        if name not in ctx.regions:
            return RegionReading(Region(name, 0, 0, 0, 0), None, "no region"), f"region '{name}' not configured"
        region = ctx.regions[name]
        if not region.calibrated:
            return RegionReading(region, None, "uncalibrated"), f"region '{name}' not calibrated"
        with ctx.stage("regions"):
            crop = region.crop(ctx.frame)
        if crop.size == 0:
            return RegionReading(region, None, "outside frame"), f"region '{name}' outside frame"
        result = ctx.ocr.read(crop, whitelist=whitelist, psm=psm, dark=region.dark_text)
        if result.empty:
            return RegionReading(region, result, "no text"), "no text read"
        if result.confidence < ctx.min_confidence:
            return RegionReading(region, result, "low confidence"), f"low confidence {result.confidence:.2f}"
        return RegionReading(region, result), None


class _ParsedDetector(Detector):
    """Single region, whitelist OCR, parsed by ``parse``."""

    region: str = ""
    whitelist: str | None = None

    def parse(self, text: str):  # -> value | None
        raise NotImplementedError

    def detect(self, ctx: DetectorContext) -> tuple[Detection, list[RegionReading]]:
        reading, problem = self._read(ctx, self.region, self.whitelist)
        raw = reading.ocr.text if reading.ocr else ""
        if problem:
            conf = reading.ocr.confidence if reading.ocr else 0.0
            return Detection.unknown(ctx.timestamp, raw or problem, conf), [reading]
        with ctx.stage("parse"):
            value = self.parse(raw)
        if value is None:
            return Detection.unknown(ctx.timestamp, raw, reading.ocr.confidence), [RegionReading(reading.region, reading.ocr, "parse failed")]
        return Detection.valid(value, reading.ocr.confidence, ctx.timestamp, raw), [reading]


class ResourceDetector(_ParsedDetector):
    whitelist = "0123456789"

    def __init__(self, resource: str) -> None:
        self.field = resource
        self.region = resource

    def parse(self, text: str) -> int | None:
        return parse_resource(text)


class PopulationDetector(_ParsedDetector):
    field = region = "population"
    whitelist = "0123456789/"

    def parse(self, text: str) -> Population | None:
        return parse_population(text)


class TimerDetector(_ParsedDetector):
    field = region = "game_time"
    whitelist = "0123456789:"

    def parse(self, text: str) -> int | None:
        return parse_clock(text)


AGE_ORDER = (Age.DARK, Age.FEUDAL, Age.CASTLE, Age.IMPERIAL)


def research_bar_fraction(crop: np.ndarray) -> float:
    """Share of saturated-red pixels: the fill of the age-up progress bar."""
    if crop.size == 0:
        return 0.0
    b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
    return float(((r > 140) & (g < 70) & (b < 70)).mean())


class AgeDetector(Detector):
    """Current age from the top-bar age name.

    While an age is being researched the bar shows the *target* age's name over a
    red progress fill, so the current age is the one before it.
    """

    field = "age"
    research_threshold = 0.03  # red share of the age_progress region

    def detect(self, ctx: DetectorContext) -> tuple[Detection, list[RegionReading]]:
        reading, problem = self._read(ctx, "age", None)
        raw = reading.ocr.text if reading.ocr else ""
        if problem:
            return Detection.unknown(ctx.timestamp, raw or problem), [reading]
        with ctx.stage("parse"):
            match = match_age(raw)
        if match is None:
            return Detection.unknown(ctx.timestamp, raw, reading.ocr.confidence), [RegionReading(reading.region, reading.ocr, "no age match")]
        age, ratio = match

        if "age_progress" in ctx.regions:
            with ctx.stage("regions"):
                fill = research_bar_fraction(ctx.regions["age_progress"].crop(ctx.frame))
            if fill >= self.research_threshold:
                index = AGE_ORDER.index(age)
                if index == 0:  # nothing researches *into* Dark Age: inconsistent read
                    return Detection.unknown(ctx.timestamp, f"{raw} (progress bar)", reading.ocr.confidence), [reading]
                age = AGE_ORDER[index - 1]
                raw = f"{raw} (researching)"
                reading = RegionReading(reading.region, reading.ocr, "age-up in progress")
        return Detection.valid(age, reading.ocr.confidence * ratio, ctx.timestamp, raw), [reading]


class SelectionDetector(Detector):
    """Name (and HP if readable) of the selected unit/building. Minimal for now."""

    field = "selection"

    def detect(self, ctx: DetectorContext) -> tuple[Detection, list[RegionReading]]:
        name_reading, problem = self._read(ctx, "selection_name", None)
        readings = [name_reading]
        raw = name_reading.ocr.text if name_reading.ocr else ""
        name = " ".join(raw.split())
        if problem or len(_normalize(name)) < 2:
            return Detection.unknown(ctx.timestamp, raw or (problem or "")), readings

        hp = hp_max = None
        hp_reading, hp_problem = self._read(ctx, "selection_hp", "0123456789/")
        readings.append(hp_reading)
        if not hp_problem and hp_reading.ocr:
            with ctx.stage("parse"):
                parsed = parse_fraction(hp_reading.ocr.text)
            if parsed and parsed[0] <= parsed[1]:
                hp, hp_max = parsed
        raw_all = raw + (f" | {hp_reading.ocr.text}" if hp_reading.ocr else "")
        return Detection.valid(Selection(name, hp, hp_max), name_reading.ocr.confidence, ctx.timestamp, raw_all), readings


def default_detectors() -> list[Detector]:
    """In implementation order: resources, population, timer, age, selection."""
    return [
        *(ResourceDetector(r) for r in ("wood", "food", "gold", "stone")),
        PopulationDetector(),
        TimerDetector(),
        AgeDetector(),
        SelectionDetector(),
    ]


# --- HUD presence -----------------------------------------------------------
class HudDetector:
    """Checks the in-game top bar is on screen by matching an icon template.

    Without this gate, OCR on a menu or loading screen could turn decorative
    text into numbers.
    """

    def __init__(self, template_path: Path = ANCHOR_DIR / "hud_anchor.png", threshold: float = 0.8, search_margin: int = 6) -> None:
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError(f"HUD anchor template missing: {template_path}")
        self.template = template
        self.threshold = threshold
        self.margin = search_margin

    def score(self, frame: np.ndarray, regions: RegionSet) -> float:
        """Best normalised correlation of the template around the anchor region (0..1)."""
        if "hud_anchor" not in regions:
            return 0.0
        region = regions["hud_anchor"]
        template = self.template
        if (region.w, region.h) != template.shape[1::-1]:
            template = cv2.resize(template, (region.w, region.h), interpolation=cv2.INTER_AREA)
        m = self.margin
        search = Region("search", region.x - m, region.y - m, region.w + 2 * m, region.h + 2 * m).crop(frame)
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            return 0.0
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        return float(max(0.0, result.max()))

    def visible(self, frame: np.ndarray, regions: RegionSet) -> tuple[bool, float]:
        s = self.score(frame, regions)
        return s >= self.threshold, s
