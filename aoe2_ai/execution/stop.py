"""Emergency stop: a latch that, once triggered, blocks all further input.

* ``EmergencyStop`` is the latch.
* ``StopKeyWatcher`` polls global keys (default F12 and Pause) on a background
  thread with ``GetAsyncKeyState``, so they work while the game has focus. If the
  watcher itself fails, it triggers the stop: a controller must never keep
  running without a working stop key.
* ``install_console_stop`` makes Ctrl+C / Ctrl+Break / closing the console
  trigger the latch from an OS thread, immediately, even while the main thread is
  busy (OCR, a wait).

Check which keys your keyboard really sends (laptop F-keys often need Fn):

    python -m aoe2_ai.execution --check
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import sys
import threading
import time
from typing import Callable, Iterable

log = logging.getLogger("aoe2_ai.execution")

DEFAULT_STOP_KEYS = ("f12", "pause")

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
VK_NAMES = {v: k for k, v in VK_CODES.items() if k != "escape"}


def vk_code(key: str) -> int:
    key = key.strip().lower()
    if key in VK_CODES:
        return VK_CODES[key]
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    raise ValueError(f"unsupported stop key {key!r}; use one of {sorted(VK_CODES)} or a letter/digit")


def parse_keys(keys: str | Iterable[str]) -> tuple[str, ...]:
    """``"f12,pause"`` or ``["f12", "pause"]`` -> validated, lower-case key names."""
    items = keys.split(",") if isinstance(keys, str) else list(keys)
    names = tuple(k.strip().lower() for k in items if k.strip())
    if not names:
        raise ValueError("at least one stop key is required")
    for name in names:
        vk_code(name)
    return names


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


def _async_key_state() -> Callable[[int], int]:
    if sys.platform != "win32":
        raise RuntimeError("the stop-key watcher uses the Windows API")
    get = ctypes.windll.user32.GetAsyncKeyState
    get.argtypes = [ctypes.c_int]
    get.restype = ctypes.c_short
    return get


class StopKeyWatcher:
    """Background thread that triggers ``stop`` when any of ``keys`` is pressed, in any window.

    ``key_state`` is ``GetAsyncKeyState`` (injectable for tests). Any error in the
    watcher triggers the stop rather than silently disabling it.
    """

    def __init__(
        self,
        stop: EmergencyStop,
        keys: str | Iterable[str] = DEFAULT_STOP_KEYS,
        poll: float = 0.03,
        key_state: Callable[[int], int] | None = None,
    ) -> None:
        self.stop = stop
        self.keys = parse_keys(keys)
        self.vks = {vk_code(k): k for k in self.keys}
        self.poll = poll
        self.key_state = key_state or _async_key_state()
        self._thread = threading.Thread(target=self._run, name="stop-key-watcher", daemon=True)
        self._quit = threading.Event()

    @property
    def label(self) -> str:
        return " or ".join(k.upper() for k in self.keys)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            for vk in self.vks:
                self.key_state(vk)  # clear the "pressed since last call" bit
            while not self._quit.is_set() and not self.stop.triggered:
                for vk, name in self.vks.items():
                    # Bit 15: down now. Bit 0: pressed since the last call (catches taps shorter than `poll`).
                    if self.key_state(vk) & 0x8001:
                        self.stop.trigger(f"stop key {name.upper()} pressed")
                        return
                self._quit.wait(self.poll)
        except Exception as exc:  # no working stop key -> stop
            self.stop.trigger(f"stop-key watcher failed: {exc!r}")

    def start(self) -> "StopKeyWatcher":
        self._thread.start()
        return self

    def close(self) -> None:
        self._quit.set()

    def __enter__(self) -> "StopKeyWatcher":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- console Ctrl+C -------------------------------------------------------------
CTRL_EVENTS = {0: "Ctrl+C", 1: "Ctrl+Break", 2: "console closed", 5: "logoff", 6: "shutdown"}
_installed_handlers: list = []  # keep ctypes callbacks alive


def console_handler(stop: EmergencyStop) -> Callable[[int], bool]:
    """The handler body: trigger the stop for any console control event."""

    def handle(event: int) -> bool:
        stop.trigger(CTRL_EVENTS.get(event, f"console event {event}"))
        return True  # handled: the loop exits on its own, no half-finished action

    return handle


def install_console_stop(stop: EmergencyStop) -> bool:
    """Make Ctrl+C / Ctrl+Break / closing the console trigger ``stop`` at once. Windows only."""
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCtrlHandler(None, False)  # undo an inherited "ignore Ctrl+C" flag
    routine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(lambda e: int(console_handler(stop)(e)))
    if not kernel32.SetConsoleCtrlHandler(routine, True):
        return False
    _installed_handlers.append(routine)
    return True


ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080


def disable_quick_edit() -> int | None:
    """Turn off QuickEdit for this console; returns the previous mode (for ``restore_console_mode``).

    With QuickEdit on, clicking into the console starts a text selection: Ctrl+C then
    copies instead of interrupting, and the program blocks on its next print until the
    selection is cleared. Returns None when there is no console or it can't be changed.
    """
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    mode = ctypes.c_uint32()
    if not handle or not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return None
    if kernel32.SetConsoleMode(handle, (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE):
        return mode.value
    return None


def restore_console_mode(mode: int | None) -> None:
    if mode is None or sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), mode | ENABLE_EXTENDED_FLAGS)


def start_exit_watchdog(stop: EmergencyStop, grace: float = 5.0, exit: Callable[[int], object] | None = None) -> threading.Thread:
    """Once ``stop`` triggers, give the program ``grace`` seconds to finish, then force-exit.

    No input can be sent after the stop anyway (the executor checks it first); this only
    guarantees the process does not hang around if the main thread is stuck.
    """
    import os

    hard_exit = exit or os._exit

    def watch() -> None:
        stop._event.wait()
        time.sleep(grace)
        log.error("still running %.0fs after the stop (%s): forcing exit", grace, stop.reason)
        logging.shutdown()
        hard_exit(2)

    thread = threading.Thread(target=watch, name="stop-exit-watchdog", daemon=True)
    thread.start()
    return thread


# --- diagnosis ---------------------------------------------------------------------
def check_keys(duration: float = 20.0, key_state: Callable[[int], int] | None = None, out=print) -> list[int]:
    """Print every key pressed for ``duration`` seconds, with its virtual-key code."""
    get = key_state or _async_key_state()
    for vk in range(1, 255):
        get(vk)
    seen: list[int] = []
    down: set[int] = set()
    end = time.monotonic() + duration
    while time.monotonic() < end:
        for vk in range(1, 255):
            pressed = bool(get(vk) & 0x8001)
            if pressed and vk not in down:
                down.add(vk)
                seen.append(vk)
                name = VK_NAMES.get(vk) or (chr(vk) if 0x30 <= vk <= 0x5A else "")
                usable = " <- usable as --stop-key " + name if name else ""
                out(f"key pressed: VK 0x{vk:02X} {name}{usable}")
            elif not pressed:
                down.discard(vk)
        time.sleep(0.02)
    return seen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emergency-stop helpers.")
    parser.add_argument("--check", action="store_true", help="print the keys you press (20 s), to pick a stop key")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return
    print(f"Press your stop key (e.g. F12) now; also try it with the game in front. Listening {args.seconds:.0f}s...")
    seen = check_keys(args.seconds)
    if VK_CODES["f12"] not in seen:
        print("F12 was not seen. If your F-keys need Fn, hold Fn, or pick another key with --stop-key.")


if __name__ == "__main__":
    main()
