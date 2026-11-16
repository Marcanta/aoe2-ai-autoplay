"""OCR of small HUD crops with OpenCV preprocessing and Tesseract.

HUD text in AoE2 DE is small (about 9 px tall at 1680x1050), light on a dark
background. Each crop is upscaled, binarised and read several times with
different thresholds; the passes vote, and the confidence reflects both how
many passes agree and Tesseract's own confidence. A read that only one pass
produces scores low and is rejected by the detectors.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

_WINDOWS_DEFAULTS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
)


class OcrUnavailable(RuntimeError):
    """Raised when the Tesseract binary cannot be found."""


@dataclass(frozen=True)
class Pass:
    """One preprocessing variant.

    ``channel``: ``"min"`` keeps only near-white pixels (all of B, G, R high),
    ``"max"`` also keeps coloured text (e.g. red population when housed),
    ``"dark"`` keeps dark pixels (text on the parchment selection panel);
    its ``threshold`` applies to ``255 - max(B, G, R)``.
    """

    channel: str = "min"
    threshold: int = 170


# Chosen on the labelled fixtures (65 numeric HUD fields): each single pass misreads
# ~1 field, while this ordered vote gave 64 correct, 1 unknown, 0 wrong values.
# The two best single passes go first so the early stop usually ends after them.
DEFAULT_PASSES = (Pass("min", 170), Pass("max", 170), Pass("min", 130), Pass("min", 150), Pass("min", 190))
DARK_PASSES = (Pass("dark", 135), Pass("dark", 155), Pass("dark", 115))


@dataclass(frozen=True)
class OcrResult:
    text: str  # winning text, "" when nothing was read
    confidence: float  # 0..1
    votes: tuple[str, ...] = ()  # text read by each pass, for debugging
    image: np.ndarray | None = field(default=None, compare=False, repr=False)  # preprocessed crop of the first pass

    @property
    def empty(self) -> bool:
        return not self.text


class OcrEngine(Protocol):
    def read(self, crop: np.ndarray, *, whitelist: str | None = None, psm: int = 7, dark: bool = False) -> OcrResult: ...


def preprocess(crop: np.ndarray, p: Pass = Pass(), *, scale: int = 4, pad: int = 20) -> np.ndarray:
    """Black text on white, upscaled, with blobs touching the crop edge removed.

    Edge blobs are bits of icons or frame decoration that the ROI clips; real
    HUD text sits inside the ROI with a margin.
    """
    if crop.size == 0:
        return np.full((2 * pad, 2 * pad), 255, np.uint8)
    if crop.ndim == 2:
        gray = crop
    elif p.channel == "min":
        gray = crop.min(axis=2)
    elif p.channel == "max":
        gray = crop.max(axis=2)
    elif p.channel == "dark":
        gray = 255 - crop.max(axis=2)
    else:
        raise ValueError(f"unknown channel {p.channel!r}")
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    mask = (gray > p.threshold).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        touches_edge = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches_edge or area < scale * scale:  # clipped decoration or single-pixel noise
            mask[labels == i] = 0

    out = np.where(mask > 0, 0, 255).astype(np.uint8)
    return cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def has_ink(image: np.ndarray) -> bool:
    return bool((image < 128).any())


def find_tesseract(explicit: str | None = None) -> str | None:
    candidates = [explicit, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), *map(str, _WINDOWS_DEFAULTS)]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


class TesseractOcr:
    """Multi-pass Tesseract reader. Create once and reuse."""

    def __init__(
        self,
        cmd: str | None = None,
        *,
        lang: str = "eng",
        passes: tuple[Pass, ...] = DEFAULT_PASSES,
        dark_passes: tuple[Pass, ...] = DARK_PASSES,
        scale: int = 4,
        early_stop: int = 2,
    ) -> None:
        import pytesseract

        path = find_tesseract(cmd)
        if path is None:
            raise OcrUnavailable(
                "Tesseract not found. Install it (winget install UB-Mannheim.TesseractOCR) "
                "or set TESSERACT_CMD to the path of tesseract.exe."
            )
        pytesseract.pytesseract.tesseract_cmd = path
        self._tess = pytesseract
        self.lang = lang
        self.passes = passes
        self.dark_passes = dark_passes
        self.scale = scale
        self.early_stop = max(2, early_stop)

    def _read_once(self, image: np.ndarray, config: str) -> tuple[str, float]:
        data = self._tess.image_to_data(image, lang=self.lang, config=config, output_type=self._tess.Output.DICT)
        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            if text:
                words.append(text)
                confs.append(max(0.0, float(conf)) / 100.0)
        if not words:
            return "", 0.0
        return " ".join(words), sum(confs) / len(confs)

    def read(self, crop: np.ndarray, *, whitelist: str | None = None, psm: int = 7, dark: bool = False) -> OcrResult:
        """Read one line of text. ``dark=True`` for dark text on a light background."""
        config = f"--psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"

        reads: list[tuple[str, float]] = []
        first_image = None
        for p in self.dark_passes if dark else self.passes:
            image = preprocess(crop, p, scale=self.scale)
            if first_image is None:
                first_image = image
            reads.append(self._read_once(image, config) if has_ink(image) else ("", 0.0))
            # Each pass is a Tesseract process launch (~50-200 ms): stop once the first
            # `early_stop` passes agree unanimously on non-empty text.
            if len(reads) == self.early_stop and reads[0][0] and all(t == reads[0][0] for t, _ in reads):
                break
        return vote(reads, first_image)


def vote(reads: list[tuple[str, float]], image: np.ndarray | None = None) -> OcrResult:
    """Combine ``(text, tesseract_confidence)`` reads into one result.

    confidence = agreement * (0.5 + 0.5 * best Tesseract confidence among agreeing reads),
    where agreement is the share of passes run that produced the winning text.
    Tesseract sometimes reports 0 for a correct short read, hence the 0.5 floor:
    agreement between passes is the stronger signal.
    """
    votes = tuple(text for text, _ in reads)
    counts = Counter(t for t in votes if t)
    if not counts:
        return OcrResult("", 0.0, votes, image)
    text, n = counts.most_common(1)[0]
    best_conf = max(c for t, c in reads if t == text)
    confidence = (n / len(reads)) * (0.5 + 0.5 * best_conf)
    return OcrResult(text, confidence, votes, image)
