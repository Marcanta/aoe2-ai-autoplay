"""Perception latency benchmark on the labelled fixture screenshots.

    python -m aoe2_ai.benchmark                       # full observation, every labelled frame, no cache
    python -m aoe2_ai.benchmark --profile town_center --cache
    python -m aoe2_ai.benchmark --compare --json docs/benchmarks/after.json   # every profile, cache off and on

Each observation runs exactly what a controller's ``LiveObserver.observe`` runs
(HUD perception, Town Center panel, idle-villager button, villager selection,
limited to the profile's fields), with the screen grab replaced by loading the
screenshot from disk (timed as ``capture``). Scenarios:

* ``cold``     - every frame on a fresh tracker and empty cache: what one
                 observation of a new screen costs.
* ``sequence`` - the labelled in-game frames in capture order, one tracker: a
                 game in progress where every frame differs from the previous one.
* ``steady``   - each frame observed twice in a row, the second one measured: a
                 screen that did not change between two controller cycles (the
                 region cache's best case; without the cache it equals ``cold``).

Correctness is checked against ``tests/fixtures/labels.json`` with the same
rules as the tests, for the fields the profile computes: a labelled value must
be VALID and equal (or, for fields in ``_ocr_misses``, may be not VALID); a
field labelled ``null`` must not be VALID. A VALID wrong value is always wrong.
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
from aoe2_ai.perception.profile import FULL, PROFILES, PerceptionProfile

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
STATE_FIELDS = ("wood", "food", "gold", "stone", "population", "game_time", "age", "selection")
STAGES = ("capture", "hud", "regions", "cache", "ocr", "parse", "panels", "track")


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


def _selection_matches(want: dict) -> Callable[[Any], bool]:
    def matches(v) -> bool:
        if SequenceMatcher(None, _normalize(v.name), _normalize(want["name"])).ratio() < 0.8:
            return False
        return "hp" not in want or [v.hp, v.hp_max] == want["hp"]
    return matches


def check(obs: Observation, label: dict, profile: PerceptionProfile = FULL) -> dict[str, str]:
    """Per labelled field the profile computes: ``ok``, ``missed`` (allowed miss), ``unknown`` (not allowed) or ``wrong``."""
    out: dict[str, str] = {"hud": "ok" if obs.state.hud_visible == label.get("hud", True) else "wrong"}
    misses = set(label.get("_ocr_misses", ()))
    for name in STATE_FIELDS:
        if name not in label or name not in profile:
            continue
        want = label[name]
        det = getattr(obs.state, name)
        if name == "selection" and want is not None:
            matches = _selection_matches(want)
        else:
            matches = lambda v, want=want: _plain(v) == want  # noqa: E731
        out[name] = _verdict(det, want, name in misses, matches)
    if "_idle" in label and "idle" in profile:
        present, count = label["_idle"]["present"], label["_idle"]["count"]
        out["idle_present"] = _verdict(obs.idle.present, present, False, lambda v: v == present)
        out["idle_count"] = _verdict(obs.idle.count, count, "idle_count" in misses, lambda v: v == count)
    if "_villager" in label and "villager" in profile:
        out["villager"] = _verdict(obs.villager.selected, label["_villager"], False, lambda v: v == label["_villager"])
    return out


# --- runs --------------------------------------------------------------------
class FileSource:
    """Placeholder screen source: the benchmark loads frames itself (``observe_file``)."""

    def grab(self):
        raise RuntimeError("the benchmark feeds frames through observe_frame")

    def close(self) -> None:
        pass


@dataclass
class Run:
    name: str
    profile: PerceptionProfile = FULL
    totals: list[float] = field(default_factory=list)  # seconds per observation
    stages: list[dict[str, float]] = field(default_factory=list)
    launches: list[int] = field(default_factory=list)
    verdicts: list[dict[str, str]] = field(default_factory=list)

    def observe(self, observer: LiveObserver, path: Path, label: dict, timestamp: float) -> Observation:
        obs = observe_file(observer, path, timestamp)
        self.totals.append(obs.timings["total"])
        self.stages.append(obs.timings)
        self.launches.append(obs.ocr_launches or 0)
        self.verdicts.append(check(obs, label, self.profile))
        return obs

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


def make_observer(perception: Perception, profile: PerceptionProfile) -> LiveObserver:
    return LiveObserver(StaticGuard(True, "benchmark"), perception=perception, source=FileSource(), profile=profile)


def observe_file(observer: LiveObserver, path: Path, timestamp: float) -> Observation:
    started = time.perf_counter()
    frame = load_frame(path, timestamp=timestamp)
    capture = time.perf_counter() - started
    return observer.observe_frame(frame, started=started, capture=capture)


def run_cold(labels: dict, frames_dir: Path, repeat: int, perception: Perception, profile: PerceptionProfile = FULL) -> Run:
    """Fresh tracker and empty cache for every frame (``reset`` clears both)."""
    run, observer = Run("cold", profile), make_observer(perception, profile)
    for _ in range(repeat):
        for name in sorted(labels):
            perception.reset()
            run.observe(observer, frames_dir / name, labels[name], 0.0)
    return run


def run_sequence(labels: dict, sequence: list[str], frames_dir: Path, repeat: int, perception: Perception,
                 profile: PerceptionProfile = FULL) -> Run:
    run, observer = Run("sequence", profile), make_observer(perception, profile)
    for _ in range(repeat):
        perception.reset()
        for t, name in enumerate(sequence):
            run.observe(observer, frames_dir / name, labels[name], t * 60.0)
    return run


def run_steady(labels: dict, frames_dir: Path, repeat: int, perception: Perception, profile: PerceptionProfile = FULL) -> Run:
    """Each frame twice in a row on a fresh tracker/cache; only the second observation is measured."""
    run, observer = Run("steady", profile), make_observer(perception, profile)
    for _ in range(repeat):
        for name in sorted(labels):
            perception.reset()
            observe_file(observer, frames_dir / name, 0.0)
            run.observe(observer, frames_dir / name, labels[name], 1.0)
    return run


def benchmark(labels: dict, sequence: list[str], frames_dir: Path, repeat: int, profile: PerceptionProfile,
              cache: bool, scenarios: tuple[str, ...]) -> dict[str, Any]:
    perception = Perception(cache=cache)
    results: dict[str, Any] = {}
    try:
        # Warm-up: the first Tesseract launches and thread start-up are slower.
        run_cold({n: labels[n] for n in sorted(labels)[:1]}, frames_dir, 1, perception, profile)
        if "cold" in scenarios:
            results["cold"] = run_cold(labels, frames_dir, repeat, perception, profile).summary()
        if "sequence" in scenarios and sequence:
            results["sequence"] = run_sequence(labels, sequence, frames_dir, repeat, perception, profile).summary()
        if "steady" in scenarios:
            results["steady"] = run_steady(labels, frames_dir, repeat, perception, profile).summary()
    finally:
        perception.close()
    return results


def format_summary(name: str, s: dict[str, Any]) -> str:
    c = s["correctness"]
    stages = "  ".join(f"{k} {v:.0f}" for k, v in s["stage_mean_ms"].items())
    return (
        f"{name:<34} n={s['observations']:<3} mean {s['mean_ms']:7.0f} ms  median {s['median_ms']:7.0f}  p95 {s['p95_ms']:7.0f}  "
        f"launches {s['launches_mean']:5.1f} (max {s['launches_max']})  "
        f"ok {c.get('ok', 0)} missed {c.get('missed', 0)} unknown {c.get('unknown', 0)} wrong {c.get('wrong', 0)}\n"
        f"{'':<34} stage means (ms, summed over threads): {stages}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark perception latency on the labelled screenshots.")
    parser.add_argument("--labels", type=Path, default=FIXTURES / "labels.json")
    parser.add_argument("--frames", type=Path, default=FIXTURES / "frames")
    parser.add_argument("--repeat", type=int, default=3, help="passes over the frame set (default 3)")
    parser.add_argument("--scenario", choices=("cold", "sequence", "steady", "all"), default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--cache", action="store_true", help="turn the region cache on")
    parser.add_argument("--compare", action="store_true", help="every profile, cache off and on (ignores --profile/--cache)")
    parser.add_argument("--json", type=Path, help="write the summaries to this file")
    args = parser.parse_args()

    data = json.loads(args.labels.read_text(encoding="utf-8"))
    labels, sequence = data["frames"], data.get("sequence", [])
    scenarios = ("cold", "sequence", "steady") if args.scenario == "all" else (args.scenario,)
    print(f"{len(labels)} labelled frames, sequence of {len(sequence)}, repeat {args.repeat}", flush=True)

    configs = [(p, c) for p in PROFILES for c in (False, True)] if args.compare else [(args.profile, args.cache)]
    results: dict[str, Any] = {}
    for profile_name, cache in configs:
        config = f"{profile_name}{' +cache' if cache else ''}"
        runs = benchmark(labels, sequence, args.frames, args.repeat, PerceptionProfile.named(profile_name), cache, scenarios)
        for scenario, summary in runs.items():
            results[f"{config} / {scenario}"] = summary
            print(format_summary(f"{config} / {scenario}", summary), flush=True)
        if args.json:  # after every configuration, so an interrupted run keeps what it measured
            args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.json:
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
