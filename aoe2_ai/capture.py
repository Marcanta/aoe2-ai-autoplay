"""Screen capture of the primary monitor with MSS.

Usage as a script (saves one PNG and prints the capture speed):

    python -m aoe2_ai.capture
    python -m aoe2_ai.capture --show   # also display the capture in a debug window
    python -m aoe2_ai.capture --live   # live preview window (q / Esc to quit)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mss
import mss.tools
import numpy as np


class ScreenCapture:
    """Grabs frames from the primary monitor.

    Keep one instance alive for the whole agent loop: creating an
    ``mss.mss()`` object is much slower than grabbing a frame with it.
    MSS objects are not thread-safe, so use the instance from one thread.
    """

    def __init__(self) -> None:
        self._sct = mss.mss()
        # monitors[0] is the union of all screens; monitors[1] is the primary one.
        self.monitor: dict[str, int] = self._sct.monitors[1]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the primary monitor, in physical pixels."""
        return self.monitor["width"], self.monitor["height"]

    def grab_raw(self) -> mss.screenshot.ScreenShot:
        """Raw MSS screenshot (BGRA), useful to save with mss.tools.to_png."""
        return self._sct.grab(self.monitor)

    def grab(self) -> np.ndarray:
        """Primary monitor as a (height, width, 3) uint8 BGR array, ready for OpenCV."""
        bgra = np.asarray(self.grab_raw())
        return np.ascontiguousarray(bgra[:, :, :3])

    def save_png(self, path: str | Path) -> Path:
        """Capture the primary monitor straight to a PNG file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shot = self.grab_raw()
        mss.tools.to_png(shot.rgb, shot.size, output=str(path))
        return path

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def live_preview(cap: ScreenCapture) -> None:
    """Show captured frames continuously until q / Esc or the window is closed."""
    from aoe2_ai.debug import close_windows, show_image

    frames, start = 0, time.perf_counter()
    try:
        while show_image(cap.grab(), "aoe2-ai live", wait_ms=1):
            frames += 1
    finally:
        close_windows()
    elapsed = time.perf_counter() - start
    if frames:
        print(f"Live preview: {frames} frames, {frames / elapsed:.1f} FPS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the primary monitor.")
    parser.add_argument("--show", action="store_true", help="display the captured frame in an OpenCV window")
    parser.add_argument("--live", action="store_true", help="live preview in an OpenCV window (q / Esc to quit)")
    args = parser.parse_args()

    with ScreenCapture() as cap:
        if args.live:
            live_preview(cap)
            return

        width, height = cap.size
        print(f"Primary monitor: {width}x{height} at ({cap.monitor['left']}, {cap.monitor['top']})")

        out = cap.save_png(Path("captures") / f"primary_{time.strftime('%Y%m%d_%H%M%S')}.png")
        print(f"Saved {out}")

        n = 60
        start = time.perf_counter()
        for _ in range(n):
            frame = cap.grab()
        elapsed = time.perf_counter() - start
        print(f"Frame shape {frame.shape}, {n / elapsed:.1f} FPS over {n} grabs")

        if args.show:
            from aoe2_ai.debug import close_windows, show_image

            print("Showing capture; press any key (or close the window) to exit.")
            show_image(frame, "aoe2-ai capture")
            close_windows()


if __name__ == "__main__":
    main()
