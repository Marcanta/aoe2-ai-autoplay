"""Emergency stop: a latch that, once triggered, blocks all further input.

``EmergencyStop`` is the latch. ``StopKeyWatcher`` polls a global key (default
F12) on a background thread with ``GetAsyncKeyState``, so it works while the
game has focus, and triggers the latch when the key is pressed.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time

log = logging.getLogger("aoe2_ai.execution")

# Virtual-key codes for keys that make sense as a stop key.
VK_CODES: dict[str, int] = {
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
    "pause": 0x13,
    "scrolllock": 0x91,
    "end": 0x23,
    "insert": 0x2D,
    "home": 0x24,
    "esc": 0x1B,
    "escape": 0x1B,
}


def vk_code(key: str) -> int:
    key = key.lower()
    if key in VK_CODES:
        return VK_CODES[key]
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    raise ValueError(f"unsupported stop key {key!r}; use one of {sorted(VK_CODES)} or a letter/digit")


class EmergencyStop:
    """Thread-safe one-way latch. Once triggered it stays triggered for the session."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    @property
    def triggered(self) -> bool:
        return self._event.is_set()

    def trigger(self, reason: str = "emergency stop") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()
            log.warning("EMERGENCY STOP: %s", reason)

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds, returning early (True) if the stop is triggered."""
        return self._event.wait(timeout)


class StopKeyWatcher:
    """Background thread that triggers ``stop`` when ``key`` is pressed anywhere."""

    def __init__(self, stop: EmergencyStop, key: str = "f12", poll: float = 0.05) -> None:
        if sys.platform != "win32":
            raise RuntimeError("the stop-key watcher uses the Windows API")
        self.stop = stop
        self.key = key
        self.vk = vk_code(key)
        self.poll = poll
        self._thread = threading.Thread(target=self._run, name="stop-key-watcher", daemon=True)
        self._quit = threading.Event()

    def _run(self) -> None:
        get = ctypes.windll.user32.GetAsyncKeyState
        get(self.vk)  # clear the "pressed since last call" bit
        while not self._quit.is_set() and not self.stop.triggered:
            if get(self.vk) & 0x8000:
                self.stop.trigger(f"stop key {self.key.upper()} pressed")
                return
            time.sleep(self.poll)

    def start(self) -> "StopKeyWatcher":
        self._thread.start()
        return self

    def close(self) -> None:
        self._quit.set()

    def __enter__(self) -> "StopKeyWatcher":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()
