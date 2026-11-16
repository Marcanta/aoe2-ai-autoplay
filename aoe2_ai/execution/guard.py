"""Game detection: is Age of Empires II the foreground window?

Keyboard input goes to whatever window has focus, so this is checked right
before every action. Anything other than a positive match blocks input.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Callable, Protocol

DEFAULT_PROCESS_NAMES = ("AoE2DE_s.exe", "AoE2DE.exe")
DEFAULT_TITLE = "Age of Empires II"


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str


class GameGuard(Protocol):
    def check(self) -> GuardResult: ...


class StaticGuard:
    """Always the same answer. For tests and offline runs."""

    def __init__(self, ok: bool, reason: str = "static") -> None:
        self.result = GuardResult(ok, reason)

    def check(self) -> GuardResult:
        return self.result


class CallableGuard:
    def __init__(self, fn: Callable[[], GuardResult]) -> None:
        self.fn = fn

    def check(self) -> GuardResult:
        try:
            return self.fn()
        except Exception as exc:  # a broken check must never let input through
            return GuardResult(False, f"guard error: {exc!r}")


@dataclass(frozen=True)
class ForegroundWindow:
    title: str
    process_path: str

    @property
    def process_name(self) -> str:
        return PureWindowsPath(self.process_path).name if self.process_path else ""


def foreground_window() -> ForegroundWindow | None:
    """Title and executable of the foreground window (Windows only; None elsewhere or on error)."""
    if sys.platform != "win32":
        return None
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)

    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    path = ""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if handle:
        try:
            size = ctypes.c_ulong(1024)
            pbuf = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(handle, 0, pbuf, ctypes.byref(size)):
                path = pbuf.value
        finally:
            kernel32.CloseHandle(handle)
    return ForegroundWindow(buf.value, path)


class ForegroundWindowGuard:
    """OK only when the foreground window belongs to the game.

    Matches the process executable name when it can be read; otherwise falls
    back to the window title. Configure ``process_names`` / ``title`` for other builds.
    """

    def __init__(
        self,
        process_names: tuple[str, ...] = DEFAULT_PROCESS_NAMES,
        title: str = DEFAULT_TITLE,
        probe: Callable[[], ForegroundWindow | None] = foreground_window,
    ) -> None:
        self.process_names = tuple(n.lower() for n in process_names)
        self.title = title.lower()
        self.probe = probe

    def check(self) -> GuardResult:
        try:
            win = self.probe()
        except Exception as exc:
            return GuardResult(False, f"cannot read foreground window: {exc!r}")
        if win is None:
            return GuardResult(False, "no foreground window")
        if win.process_name:
            if win.process_name.lower() in self.process_names:
                return GuardResult(True, f"foreground: {win.process_name}")
            return GuardResult(False, f"foreground is {win.process_name!r} ({win.title!r}), not the game")
        if self.title and self.title in win.title.lower():
            return GuardResult(True, f"foreground title: {win.title!r}")
        return GuardResult(False, f"foreground is {win.title!r}, not the game")
