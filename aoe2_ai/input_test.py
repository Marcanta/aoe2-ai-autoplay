"""Controlled test of mouse and keyboard input.

Opens a small always-on-top Tk window that records every event it receives,
drives the real mouse/keyboard into it with ``InputController``, then checks
that what arrived matches what was sent.

    python -m aoe2_ai.input_test

Keystrokes are only sent while the test window has keyboard focus, so nothing
is typed into other applications. Move the mouse into a screen corner to abort.
"""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
import tkinter as tk

from aoe2_ai.input import InputController  # sets DPI awareness before Tk starts

WIDTH, HEIGHT = 640, 420
TYPED_TEXT = "Hello AoE2 123"


class TestWindow:
    """Tk window that logs mouse and keyboard events into a thread-safe queue."""

    def __init__(self) -> None:
        self.events: queue.Queue[tuple] = queue.Queue()
        self.root = tk.Tk()
        self.root.title("aoe2-ai input test")
        self.root.geometry(f"{WIDTH}x{HEIGHT}+200+150")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT - 60, bg="#20262e", highlightthickness=0)
        self.canvas.pack()
        self.entry = tk.Entry(self.root, font=("Consolas", 14))
        self.entry.pack(fill="x", padx=10, pady=10)
        self.status = self.canvas.create_text(10, 10, anchor="nw", fill="#ddd", font=("Consolas", 11), text="")

        c = self.canvas
        c.bind("<Motion>", lambda e: self.events.put(("move", e.x_root, e.y_root)))
        c.bind("<ButtonPress-1>", lambda e: self._click("left", e))
        c.bind("<ButtonPress-3>", lambda e: self._click("right", e))
        c.bind("<Double-Button-1>", lambda e: self.events.put(("double", e.x_root, e.y_root)))
        c.bind("<ButtonRelease-1>", lambda e: self.events.put(("release", e.x_root, e.y_root)))
        # Tk 8.6 on Windows routes the wheel to the focused widget, so listen app-wide.
        self.root.bind_all("<MouseWheel>", lambda e: self.events.put(("wheel", e.delta)))
        self.entry.bind("<ButtonPress-1>", lambda e: self.events.put(("entry_click",)))
        self.root.bind_all("<KeyPress>", lambda e: self.events.put(("key", e.keysym, e.state)))

    def _click(self, button: str, e: tk.Event) -> None:
        self.events.put(("click", button, e.x_root, e.y_root))
        self.canvas.create_oval(e.x - 4, e.y - 4, e.x + 4, e.y + 4, fill="#e8a33d" if button == "left" else "#4fa3e8", width=0)

    def set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.canvas.itemconfigure(self.status, text=text))

    def canvas_origin(self) -> tuple[int, int]:
        return self.canvas.winfo_rootx(), self.canvas.winfo_rooty()

    def entry_center(self) -> tuple[int, int]:
        e = self.entry
        return e.winfo_rootx() + e.winfo_width() // 2, e.winfo_rooty() + e.winfo_height() // 2

    def hwnd(self) -> int:
        return ctypes.windll.user32.GetParent(self.root.winfo_id())


def drain(q: queue.Queue, settle: float = 0.3) -> list[tuple]:
    time.sleep(settle)
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def run_checks(win: TestWindow, results: list[tuple[str, bool, str]]) -> None:
    ctl = InputController(move_duration=0.2)
    ox, oy = win.canvas_origin()
    tol = 2  # pixels

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        win.set_status("\n".join(f"{'PASS' if r[1] else 'FAIL'}  {r[0]}" for r in results))

    def near(ax: int, ay: int, bx: int, by: int) -> bool:
        return abs(ax - bx) <= tol and abs(ay - by) <= tol

    time.sleep(0.5)
    drain(win.events)

    # 1. Move: cursor lands where asked, and the window sees motion.
    targets = [(100, 100), (500, 80), (320, 250), (60, 300)]
    for tx, ty in targets:
        sx, sy = ox + tx, oy + ty
        ctl.move_to(sx, sy)
        px, py = ctl.position
        check(f"move to canvas ({tx},{ty})", near(px, py, sx, sy), f"cursor at ({px},{py}), wanted ({sx},{sy})")
    moves = [e for e in drain(win.events) if e[0] == "move"]
    check("window received motion events", len(moves) > 0, f"{len(moves)} motion events")

    # 2. Left / right / double click at precise points.
    sx, sy = ox + 150, oy + 180
    ctl.click(sx, sy)
    ev = [e for e in drain(win.events) if e[0] == "click"]
    check("left click position", bool(ev) and ev[-1][1] == "left" and near(ev[-1][2], ev[-1][3], sx, sy), str(ev[-1:]))

    sx, sy = ox + 450, oy + 180
    ctl.right_click(sx, sy)
    ev = [e for e in drain(win.events) if e[0] == "click"]
    check("right click position", bool(ev) and ev[-1][1] == "right" and near(ev[-1][2], ev[-1][3], sx, sy), str(ev[-1:]))

    ctl.double_click(ox + 300, oy + 300)
    ev = drain(win.events)
    check("double click", any(e[0] == "double" for e in ev), f"{sum(e[0] == 'click' for e in ev)} clicks")

    # 3. Drag (box select): press at start, release at end.
    x1, y1, x2, y2 = ox + 80, oy + 60, ox + 260, oy + 200
    ctl.drag(x1, y1, x2, y2)
    ev = drain(win.events)
    press = [e for e in ev if e[0] == "click"]
    release = [e for e in ev if e[0] == "release"]
    ok = bool(press and release) and near(press[0][2], press[0][3], x1, y1) and near(release[-1][1], release[-1][2], x2, y2)
    check("drag start/end", ok, f"press={press[:1]} release={release[-1:]}")

    # 4. Keyboard: focus the entry by clicking it, and refuse to type unless we own focus.
    ctl.click(*win.entry_center())
    drain(win.events)
    fg = ctypes.windll.user32.GetForegroundWindow()
    has_focus = fg == win.hwnd()
    check("test window has keyboard focus", has_focus, f"foreground hwnd {fg}, ours {win.hwnd()}")
    if not has_focus:
        check("keyboard tests skipped (no focus)", False, "not typing into another window")
        return

    # 5. Scroll wheel (needs a focused Tk widget to be delivered on Windows).
    ctl.scroll(3, ox + 320, oy + 200)
    ctl.scroll(-3, ox + 320, oy + 200)
    wheel = [e[1] for e in drain(win.events) if e[0] == "wheel"]
    check("scroll up and down", any(d > 0 for d in wheel) and any(d < 0 for d in wheel), f"deltas {wheel}")

    ctl.type_text(TYPED_TEXT)
    time.sleep(0.3)
    got = win.entry.get()
    check("typed text arrives intact", got == TYPED_TEXT, f"entry={got!r}")
    drain(win.events)

    for key, keysym in [("h", "h"), ("esc", "Escape"), ("f1", "F1"), ("delete", "Delete"), ("left", "Left")]:
        ctl.press(key)
        keys = [e[1] for e in drain(win.events, 0.15) if e[0] == "key"]
        check(f"press {key}", keysym in keys, f"got {keys}")

    ctl.hotkey("ctrl", "a")
    ev = [e for e in drain(win.events) if e[0] == "key"]
    ctrl_a = any(e[1].lower() == "a" and e[2] & 0x4 for e in ev)  # 0x4 = Control
    check("hotkey ctrl+a (modifier held)", ctrl_a, str(ev))

    ctl.hotkey("shift", "h")
    ev = [e for e in drain(win.events) if e[0] == "key"]
    check("hotkey shift+h", any(e[1] == "H" for e in ev), str(ev))


def main() -> int:
    if sys.platform != "win32":
        print("This test targets Windows.")
        return 2
    win = TestWindow()
    results: list[tuple[str, bool, str]] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            run_checks(win, results)
        except BaseException as exc:  # includes pyautogui.FailSafeException
            error.append(exc)
        finally:
            win.root.after(1500, win.root.destroy)

    win.root.update()
    win.root.focus_force()
    threading.Thread(target=worker, daemon=True).start()
    win.root.mainloop()

    width = max((len(r[0]) for r in results), default=10)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}".rstrip())
    if error:
        print(f"ABORTED: {error[0]!r}")
        return 1
    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
