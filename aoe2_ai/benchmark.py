"""Perception latency benchmark on the labelled fixture screenshots.

    python -m aoe2_ai.benchmark                       # full observation, every labelled frame, cold
    python -m aoe2_ai.benchmark --repeat 5 --json out.json

Each observation runs exactly what a controller's ``LiveObserver.observe`` runs
(HUD perception, Town Center panel, idle-villager button, villager selection),
with the screen grab replaced by loading the screenshot from disk (timed as
``capture``). Scenarios:

* ``cold``     - every frame on a fresh tracker: what one observation costs.
* ``sequence`` - the labelled in-game frames in capture order, one tracker: a
                 game in progress where every frame differs from the previous one.

Correctness is checked against ``tests/fixtures/labels.json`` with the same
rules as the tests: a labelled value must be VALID and equal (or, for fields in
``_ocr_misses``, may be not VALID); a field labelled ``null`` must not be VALID.
A VALID wrong value is always counted as wrong.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from aoe2_ai.controllers.observation import LiveObserver, Observation
from aoe2_ai.execution import StaticGuard
from aoe2_ai.perception import Perception, load_frame
from aoe2_ai.perception.detectors import _normalize
from aoe2_ai.perception.game_state import Status

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
STATE_FIELDS = ("wood", "food", "gold", "stone", "population", "game_time", "age", "selection")
STAGES = ("capture", "hud", "regions", "ocr", "parse", "panels", "track")


# --- correctness -------------------------------------------------------------
def _plain(value: Any) -> Any:
    if hasattr(value, "current"):
        return [value.current, value.cap]
    if hasattr(value, "value"):  # enum
        return value.value
    return value


def _verdict(det, want, may_miss: bool, matches: Callable[[Any], bool]) -> str:
    valid = det.status is Status.VALID
    if want is None:
        return "wrong" if valid else "ok"
    if not valid:
        return "missed" if may_miss else "unknown"
    return "ok" if matches(det.value) else "wrong"


def check(obs: Observation, label: dict, fields: tuple[str, ...] | None = None) -> dict[str, str]:
    """Per labelled field: ``ok``, ``missed`` (allowed miss), ``unknown`` (not allowed) or ``wrong``."""
    out: dict[str, str] = {}
    misses = set(label.get("_ocr_misses", ()))
    wanted = lambda name: fields is None or name in fields  # noqa: E731

    if wanted("hud"):
        out["hud"] = "ok" if obs.state.hud_visible == label.get("hud", True) else "wrong"
    for name in STATE_FIELDS:
        if name not in label or not wanted(name):
            continue
        want = label[name]
        det = getattr(obs.state, name)
        if name == "selection" and want is not None:
            def matches(v, want=want):
                if SequenceMatcher(None, _normalize(v.name), _normalize(want["name"])).ratio() < 0.8:
                    return False
                return "hp" not in want or [v.hp, v.hp_max] == want["hp"]
        else:
            matches = lambda v, want=want: _plain(v) == want  # noqa: E731
        out[name] = _verdict(det, want, name in misses, matches)
    if "_idle" in label and wanted("idle"):
        out["idle_present"] = _verdict(obs.idle.present, label["_idle"]["present"], False, lambda v: v == label["_idle"]["present"])
        out["idle_count"] = _verdict(obs.idle.count, label["_idle"]["count"], "idle_count" in misses, lambda v: v == label["_idle"]["count"])
    if "_villager" in label and wanted("villager"):
        out["villager"] = _verdict(obs.villager.selected, label["_villager"], False, lambda v: v == label["_villager"])
    return out


# --- runs --------------------------------------------------------------------
class FileSource:
    def __init__(self) -> None:
        self.path: Path | None = None

    def grab(self):
        return load_frame(self.path)

    def close(self) -> None:
        pass


@dataclass
class Run:
    name: str
    totals: list[float] = field(default_factory=list)  # seconds per observation
    stages: list[dict[str, float]] = field(default_factory=list)
    launches: list[int] = field(default_factory=list)
    verdicts: list[dict[str, str]] = field(default_factory=list)

    def add(self, obs: Observation, verdicts: dict[str, str]) -> None:
        self.totals.append(obs.timings["total"])
        self.stages.append(obs.timings)
        self.launches.append(obs.ocr_launches or 0)
        self.verdicts.append(verdicts)

    def summary(self) -> dict[str, Any]:
        ms = sorted(t * 1000 for t in self.totals)
        p95 = ms[min(len(ms) - 1, max(0, round(0.95 * len(ms)) - 1))] if ms else 0.0
        counts: dict[str, int] = {}
        for v in self.verdicts:
            for outcome in v.values():
                counts[outcome] = counts.get(outcome, 0) + 1
        return {
            "observations": len(ms),
            "mean_ms": statistics.fmean(ms) if ms else 0.0,
            "median_ms": statistics.median(ms) if ms else 0.0,
            "p95_ms": p95,
            "max_ms": ms[-1] if ms else 0.0,
            "launches_mean": statistics.fmean(self.launches) if self.launches else 0.0,
            "launches_max": max(self.launches, default=0),
            "stage_mean_ms": {s: statistics.fmean(st.get(s, 0.0) for st in self.stages) * 1000 for s in STAGES} if self.stages else {},
            "correctness": counts,
        }


def make_observer(perception: Perception, source: FileSource) -> LiveObserver:
    return LiveObserver(StaticGuard(True, "benchmark"), perception=perception, source=source)


def observe_file(observer: LiveObserver, source: FileSource, path: Path, timestamp: float) -> Observation:
    source.path = path
    started = time.perf_counter()
    frame = load_frame(path, timestamp=timestamp)
    capture = time.perf_counter() - started
    return observer.observe_frame(frame, started=started, capture=capture)


def run_cold(labels: dict, frames_dir: Path, repeat: int, perception: Perception) -> Run:
    run = Run("cold")
    source = FileSource()
    observer = make_observer(perception, source)
    for _ in range(repeat):
        for name in sorted(labels):
            perception.reset()
            obs = observe_file(observer, source, frames_dir / name, 0.0)
            run.add(obs, check(obs, labels[name]))
    return run


def run_sequence(labels: dict, sequence: list[str], frames_dir: Path, repeat: int, perception: Perception) -> Run:
    run = Run("sequence")
    source = FileSource()
    observer = make_observer(perception, source)
    for _ in range(repeat):
        perception.reset()
        for t, name in enumerate(sequence):
            obs = observe_file(observer, source, frames_dir / name, t * 60.0)
            run.add(obs, check(obs, labels[name]))
    return run


def format_summary(name: str, s: dict[str, Any]) -> str:
    c = s["correctness"]
    stages = "  ".join(f"{k} {v:.0f}" for k, v in s["stage_mean_ms"].items())
    return (
        f"{name:<28} n={s['observations']:<3} mean {s['mean_ms']:7.0f} ms  median {s['median_ms']:7.0f}  p95 {s['p95_ms']:7.0f}  "
        f"launches {s['launches_mean']:5.1f} (max {s['launches_max']})  "
        f"ok {c.get('ok', 0)} missed {c.get('missed', 0)} unknown {c.get('unknown', 0)} wrong {c.get('wrong', 0)}\n"
        f"{'':<28} stage means (ms, summed over threads): {stages}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark perception latency on the labelled screenshots.")
    parser.add_argument("--labels", type=Path, default=FIXTURES / "labels.json")
    parser.add_argument("--frames", type=Path, default=FIXTURES / "frames")
    parser.add_argument("--repeat", type=int, default=3, help="passes over the frame set (default 3)")
    parser.add_argument("--scenario", choices=("cold", "sequence", "all"), default="all")
    parser.add_argument("--json", type=Path, help="write the summaries to this file")
    args = parser.parse_args()

    data = json.loads(args.labels.read_text(encoding="utf-8"))
    labels, sequence = data["frames"], data.get("sequence", [])
    print(f"{len(labels)} labelled frames, sequence of {len(sequence)}, repeat {args.repeat}")

    perception = Perception()
    results: dict[str, Any] = {}
    try:
        # Warm-up: first Tesseract launches and thread start-up are slower.
        run_cold({n: labels[n] for n in sorted(labels)[:1]}, args.frames, 1, perception)
        if args.scenario in ("cold", "all"):
            results["cold"] = run_cold(labels, args.frames, args.repeat, perception).summary()
        if args.scenario in ("sequence", "all") and sequence:
            results["sequence"] = run_sequence(labels, sequence, args.frames, args.repeat, perception).summary()
    finally:
        perception.close()

    for name, summary in results.items():
        print(format_summary(name, summary))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
