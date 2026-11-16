"""Mouse and keyboard control with pyautogui.

Coordinates are physical screen pixels, the same space as ``ScreenCapture``
frames, so a point found in a captured frame can be clicked directly.

Safety: pyautogui's fail-safe stays on. Slam the mouse into any screen corner
to raise ``pyautogui.FailSafeException`` and stop the agent.
"""

from __future__ import annotations

import ctypes
import sys

# Match MSS: work in physical pixels even when Windows display scaling is > 100%.
# Must run before pyautogui is imported, since it sets a weaker DPI mode itself.
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()

import pyautogui  # noqa: E402

pyautogui.FAILSAFE = True


class InputController:
    """Thin wrapper around pyautogui with the actions the agent needs."""

    def __init__(self, *, move_duration: float = 0.15, pause: float = 0.05) -> None:
        self.move_duration = move_duration
        pyautogui.PAUSE = pause  # delay after every pyautogui call

    # --- mouse -----------------------------------------------------------
    @property
    def position(self) -> tuple[int, int]:
        p = pyautogui.position()
        return p.x, p.y

    def move_to(self, x: int, y: int, duration: float | None = None) -> None:
        pyautogui.moveTo(x, y, duration=self.move_duration if duration is None else duration)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self.move_to(x, y)
        pyautogui.click(x, y, clicks=clicks, interval=0.05, button=button)

    def right_click(self, x: int, y: int) -> None:
        self.click(x, y, button="right")

    def double_click(self, x: int, y: int) -> None:
        self.click(x, y, clicks=2)

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.3) -> None:
        """Left-button drag, e.g. a box selection in game."""
        self.move_to(x1, y1)
        pyautogui.mouseDown(button="left")
        try:
            pyautogui.moveTo(x2, y2, duration=duration)
        finally:
            pyautogui.mouseUp(button="left")

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> None:
        """Scroll ``clicks`` wheel notches (positive = up / zoom in)."""
        # On Windows pyautogui passes ``clicks`` as the raw wheel delta, but one
        # notch is WHEEL_DELTA = 120; smaller deltas are ignored by most apps.
        amount = clicks * 120 if sys.platform == "win32" else clicks
        pyautogui.scroll(amount, x=x, y=y)

    # --- keyboard --------------------------------------------------------
    def press(self, key: str, presses: int = 1) -> None:
        pyautogui.press(key, presses=presses, interval=0.05)

    def hotkey(self, *keys: str) -> None:
        pyautogui.hotkey(*keys)

    def type_text(self, text: str, interval: float = 0.02) -> None:
        pyautogui.write(text, interval=interval)
