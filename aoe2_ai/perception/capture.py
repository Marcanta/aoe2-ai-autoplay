"""Frames for perception: live from the screen (existing MSS ``ScreenCapture``) or from PNG files.

Also a small fixture tool to collect test screenshots while playing:

    python -m aoe2_ai.perception.capture --name start              # one frame now
    python -m aoe2_ai.perception.capture --name eco --count 5 --interval 20 --delay 3

Each grab writes the full frame to ``captures/fixtures/`` and a HUD-only copy
(play area blanked, so it compresses to a few hundred KB) to
``tests/fixtures/frames/``, ready to be labelled in ``tests/fixtures/labels.json``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from aoe2_ai.capture import ScreenCapture

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "frames"
FULL_CAPTURE_DIR = REPO_ROOT / "captures" / "fixtures"

# Bands of a 1680x1050 frame kept in HUD-only fixtures (top bar + timer, bottom panel).
HUD_TOP_FRACTION = 120 / 1050
HUD_BOTTOM_FRACTION = 230 / 1050


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR uint8
    timestamp: float  # time.time() when grabbed (or loaded)
    source: str  # "screen" or a file path


class LiveSource:
    """Frames from the primary monitor. Reuses one ScreenCapture for speed."""

    def __init__(self, capture: ScreenCapture | None = None) -> None:
        self._owned = capture is None
        self.capture = capture or ScreenCapture()

    def grab(self) -> Frame:
        return Frame(self.capture.grab(), time.time(), "screen")

    def __iter__(self) -> Iterator[Frame]:
        while True:
            yield self.grab()

    def close(self) -> None:
        if self._owned:
            self.capture.close()

    def __enter__(self) -> "LiveSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_frame(path: str | Path, timestamp: float | None = None) -> Frame:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return Frame(image, time.time() if timestamp is None else timestamp, str(path))


def file_frames(paths: Iterable[str | Path]) -> Iterator[Frame]:
    for p in paths:
        yield load_frame(p)


def hud_only(image: np.ndarray) -> np.ndarray:
    """Copy of ``image`` with the play area blacked out; HUD bands keep their exact pixels."""
    out = image.copy()
    h = out.shape[0]
    top, bottom = round(h * HUD_TOP_FRACTION), h - round(h * HUD_BOTTOM_FRACTION)
    out[top:bottom] = 0
    return out


def save_fixture(image: np.ndarray, name: str) -> tuple[Path, Path]:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    FULL_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    full = FULL_CAPTURE_DIR / f"{stem}.png"
    hud = FIXTURE_DIR / f"{stem}.png"
    cv2.imwrite(str(full), image)
    cv2.imwrite(str(hud), hud_only(image), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return full, hud


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab test fixture frames from the screen.")
    parser.add_argument("--name", required=True, help="label prefix, e.g. 'feudal' or 'villager_selected'")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between grabs")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds before the first grab (to switch to the game)")
    args = parser.parse_args()

    print(f"First grab in {args.delay:.0f}s - switch to the game.")
    time.sleep(args.delay)
    with ScreenCapture() as cap:
        for i in range(args.count):
            if i:
                time.sleep(args.interval)
            full, hud = save_fixture(cap.grab(), args.name)
            print(f"[{i + 1}/{args.count}] {hud.relative_to(REPO_ROOT)}  (full frame: {full.relative_to(REPO_ROOT)})")


if __name__ == "__main__":
    main()
