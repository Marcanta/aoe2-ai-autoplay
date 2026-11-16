"""Debug views and dumps for perception results.

* ``overlay``  - the frame with every ROI outlined and its reading next to it
                 (green VALID, yellow STALE, red UNKNOWN, grey uncalibrated).
* ``crops_mosaic`` - each ROI enlarged next to the binarised image Tesseract read.
* ``save_debug`` - writes frame, overlay, mosaic and state JSON to a folder.
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from .game_state import Detection, Status
from .pipeline import PerceptionResult

_COLORS = {Status.VALID: (80, 200, 80), Status.STALE: (40, 200, 230), Status.UNKNOWN: (60, 60, 230)}
_GREY = (150, 150, 150)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ascii(text: str) -> str:
    """OpenCV's Hershey fonts only draw ASCII."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if ord(c) < 128)


def _short(det: Detection) -> str:
    value = det.value
    if value is None:
        shown = "?"
    elif hasattr(value, "current"):
        shown = f"{value.current}/{value.cap}"
    elif hasattr(value, "name") and hasattr(value, "hp"):
        shown = value.name + (f" {value.hp}/{value.hp_max}" if value.hp is not None else "")
    elif hasattr(value, "value"):
        shown = value.value
    else:
        shown = str(value)
    return _ascii(f"{shown} {det.confidence:.2f} {det.status.value}")


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.45) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 2))
    y = max(th + 2, min(y, img.shape[0] - 2))
    cv2.rectangle(img, (x - 1, y - th - 3), (x + tw + 1, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)


def overlay(frame: np.ndarray, result: PerceptionResult) -> np.ndarray:
    img = frame.copy()
    detections = result.state.detections()
    regions = result.regions
    if regions is None:
        return img
    for region in regions:
        field = "selection" if region.name.startswith("selection") else region.name
        det = detections.get(field)
        color = _GREY if not region.calibrated else (_COLORS[det.status] if det else (255, 200, 0))
        x, y, w, h = region.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        if det is not None and region.name != "selection_hp":
            text = f"{region.name}: {_short(det)}" if region.calibrated else f"{region.name}: uncalibrated"
            # Top-bar labels alternate between two lines so neighbours don't overlap.
            stagger = 16 if field in ("food", "stone", "age") else 0
            _label(img, text, (x, y + h + 14 + stagger), color)
    hud = f"HUD {'visible' if result.state.hud_visible else 'NOT visible'} (score {result.hud_score:.2f})"
    _label(img, hud, (10, img.shape[0] // 2), (255, 255, 255), 0.6)
    return img


def crops_mosaic(frame: np.ndarray, result: PerceptionResult, width: int = 900) -> np.ndarray:
    """One row per read region: enlarged original crop | binarised image | OCR votes."""
    rows = []
    for readings in result.readings.values():
        for reading in readings:
            if reading.ocr is None or reading.ocr.image is None:
                continue
            crop = reading.region.crop(frame)
            row_h = 60
            parts = []
            if crop.size:
                parts.append(cv2.resize(crop, (max(1, round(crop.shape[1] * row_h / crop.shape[0])), row_h), interpolation=cv2.INTER_NEAREST))
            binar = cv2.cvtColor(reading.ocr.image, cv2.COLOR_GRAY2BGR)
            parts.append(cv2.resize(binar, (max(1, round(binar.shape[1] * row_h / binar.shape[0])), row_h), interpolation=cv2.INTER_AREA))
            row = np.zeros((row_h + 24, width, 3), np.uint8)
            x = 0
            for p in parts:
                p = p[:, : max(0, width - x)]
                row[:row_h, x : x + p.shape[1]] = p
                x += p.shape[1] + 8
            votes = " | ".join(v or "-" for v in reading.ocr.votes)
            text = f"{reading.region.name}: '{reading.ocr.text}' conf {reading.ocr.confidence:.2f}  votes [{votes}] {reading.note}"
            cv2.putText(row, _ascii(text), (4, row_h + 17), _FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            rows.append(row)
    if not rows:
        blank = np.zeros((40, width, 3), np.uint8)
        cv2.putText(blank, "no OCR readings (HUD not visible?)", (8, 26), _FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return blank
    return np.vstack(rows)


def save_debug(out_dir: str | Path, frame: np.ndarray, result: PerceptionResult) -> Path:
    """Write ``<stamp>_frame.png``, ``_overlay.png``, ``_crops.png`` and ``_state.json``. Returns the state path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(result.state.timestamp)) + f"_{int(result.state.timestamp * 1000) % 1000:03d}"
    cv2.imwrite(str(out / f"{stamp}_frame.png"), frame)
    cv2.imwrite(str(out / f"{stamp}_overlay.png"), overlay(frame, result))
    cv2.imwrite(str(out / f"{stamp}_crops.png"), crops_mosaic(frame, result))
    state_path = out / f"{stamp}_state.json"
    payload = {"state": result.state.to_dict(), "observed": result.observed.to_dict(), "hud_score": result.hud_score}
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return state_path
