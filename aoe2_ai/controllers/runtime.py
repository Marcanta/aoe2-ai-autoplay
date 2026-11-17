"""Live-run plumbing shared by the controller CLIs: emergency stop, guard, executor, observer.

Every controller runs behind the same safety net, so it lives in one place:
stop key watcher (F12 / Pause), Ctrl+C console handler, exit watchdog,
foreground-window guard, and a single :class:`ActionExecutor` for all input.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Callable, Protocol

from aoe2_ai.execution import ActionExecutor, EmergencyStop
from aoe2_ai.perception.digit_recognizer import OCR_ENGINES
from aoe2_ai.perception.profile import FULL, PerceptionProfile

from .observation import LiveObserver


class RunnableController(Protocol):
    stats: Any
    failures: list

    def run(self, max_cycles: int | None = None) -> Any: ...


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="observe and decide, but never send input")
    parser.add_argument("--stop-key", default="f12,pause", help="emergency stop key(s), comma-separated (default f12,pause); "
                        "check yours with: python -m aoe2_ai.execution --check")
    parser.add_argument("--max-cycles", type=int, help="stop after this many cycles")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds before starting (to switch to the game)")
    parser.add_argument("--metrics", metavar="DIR", help="record observation latency, cycle outcomes and before/after frames of every action to DIR")
    parser.add_argument("--ocr", choices=OCR_ENGINES, default="hybrid",
                        help="OCR engine: hybrid (default; OpenCV for numbers, Tesseract for text and as fallback) or tesseract (Tesseract only)")
    parser.add_argument("-v", "--verbose", action="store_true")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("pytesseract").setLevel(logging.INFO)  # one debug line per OCR call drowns -v output


def run_live(
    args: argparse.Namespace,
    make_controller: Callable[[LiveObserver, ActionExecutor], RunnableController],
    debug_dir: str,
    profile: PerceptionProfile = FULL,
) -> None:
    """Build the safety net, run the controller until stop / halt / ``--max-cycles``, clean up.

    ``profile``: the fields the controller reads; the observer computes only those.
    """
    # Imported here: aoe2_ai.input sets process DPI awareness and loads pyautogui.
    from aoe2_ai.execution import ForegroundWindowGuard, StopKeyWatcher
    from aoe2_ai.execution.stop import disable_quick_edit, install_console_stop, restore_console_mode, start_exit_watchdog
    from aoe2_ai.input import InputController

    stop = EmergencyStop()
    watcher = StopKeyWatcher(stop, args.stop_key)  # validates the keys before anything starts
    console = install_console_stop(stop)  # Ctrl+C stops at once, even mid-OCR
    guard = ForegroundWindowGuard()
    executor = ActionExecutor(None if args.dry_run else InputController(), guard, dry_run=args.dry_run, stop=stop)
    observer = LiveObserver(guard, profile=profile, ocr=getattr(args, "ocr", "hybrid"))
    controller = make_controller(observer, executor)
    metrics = None
    if getattr(args, "metrics", None):
        from .metrics import LiveMetrics

        metrics = LiveMetrics(args.metrics)
        metrics.attach(observer, controller)

    print(
        f"Starting in {args.delay:.0f}s - switch to the game.\n"
        f"STOP: press {watcher.label} (works with the game in front; on laptops F-keys may need Fn),\n"
        f"      or Ctrl+C in this window{'' if console else ' (handler not installed)'}, or slam the mouse into a screen corner."
    )
    quick_edit = disable_quick_edit()  # a click in the console would otherwise freeze the program
    start_exit_watchdog(stop)  # a stop always ends the process, even if the main thread is stuck
    with watcher:
        if stop.wait(args.delay):
            print(f"Stopped before starting: {stop.reason}")
            return
        try:
            stats = controller.run(args.max_cycles)
        except KeyboardInterrupt:
            stop.trigger("Ctrl+C")
            stats = controller.stats
        finally:
            observer.close()
            restore_console_mode(quick_edit)
            if metrics is not None:
                summary = metrics.close()
                print(f"Metrics ({args.metrics}): {json.dumps(summary)}")
    print(f"Done: {stats}. Failures: {len(controller.failures)}" + (f" (see {debug_dir})" if controller.failures else ""))
