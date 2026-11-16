"""OpenCV debug windows for inspecting frames during development."""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_WINDOW = "aoe2-ai debug"
_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q or Esc


def _fit(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Downscale ``frame`` to fit in max_width x max_height, keeping its aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def show_image(
    frame: np.ndarray,
    window: str = DEFAULT_WINDOW,
    *,
    wait_ms: int = 0,
    max_width: int = 1280,
    max_height: int = 720,
) -> bool:
    """Show a BGR (or grayscale) frame in a resizable OpenCV window.

    The frame is downscaled to fit ``max_width`` x ``max_height`` so a full-screen
    capture does not cover the screen it was taken from.

    ``wait_ms=0`` blocks until a key is pressed; a positive value waits that many
    milliseconds, which is what a live loop wants (e.g. 1).

    Returns ``False`` when the user pressed q / Esc or closed the window, so a
    loop can stop; ``True`` otherwise.
    """
    shown = _fit(frame, max_width, max_height)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, shown.shape[1], shown.shape[0])
    cv2.imshow(window, shown)

    if wait_ms > 0:
        key = cv2.waitKey(wait_ms) & 0xFF
    else:
        # Block until a key is pressed, but notice if the window is closed with the mouse.
        key = 255
        while key == 255:
            key = cv2.waitKey(100) & 0xFF
            if not _is_open(window):
                return False

    return key not in _QUIT_KEYS and _is_open(window)


def _is_open(window: str) -> bool:
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def close_windows() -> None:
    cv2.destroyAllWindows()
