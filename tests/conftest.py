from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aoe2_ai.perception.ocr import OcrResult, find_tesseract
from aoe2_ai.perception.regions import RegionSet

FIXTURES = Path(__file__).parent / "fixtures"
FRAMES = FIXTURES / "frames"


def load_labels() -> dict:
    return json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))


requires_tesseract = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract binary not installed")


def tagged_frame(regions: RegionSet) -> np.ndarray:
    """Synthetic frame where each region is filled with its own grey level, so a fake OCR can tell crops apart."""
    w, h = regions.reference_size
    frame = np.zeros((h, w, 3), np.uint8)
    for i, region in enumerate(regions, start=1):
        if region.kind in ("icon", "bar"):  # not OCR'd; may overlap text regions
            continue
        frame[region.y : region.y + region.h, region.x : region.x + region.w] = i
    return frame


class FakeOcr:
    """OCR stand-in returning scripted text per region name (missing name -> nothing read)."""

    def __init__(self, regions: RegionSet, texts: dict[str, str], confidence: float = 0.95) -> None:
        self.names = {i: r.name for i, r in enumerate(regions, start=1)}
        self.texts = texts
        self.confidence = confidence
        self.calls: list[str] = []

    def read(self, crop, *, whitelist=None, psm=7, dark=False) -> OcrResult:
        name = self.names[int(crop[0, 0, 0])]
        self.calls.append(name)
        text = self.texts.get(name, "")
        return OcrResult(text, self.confidence if text else 0.0, (text,))
