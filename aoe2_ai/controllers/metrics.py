"""Live-run metrics for the controller CLIs (``--metrics DIR``).

Records, without changing what the controller does:

* every observation: wall latency, Tesseract launches, per-stage timings, game detected,
  the numeric fields it holds (value / status / raw text) and, with ``HybridOcr``, how many
  numbers OpenCV read and how many went to the Tesseract fallback;
* every ``frame_every`` seconds in game, the screenshot (``frame_NNNN.png``), so the
  recorded readings can be checked against the screen afterwards;
* every cycle: outcome, decision and detail;
* for every cycle that pressed a key: the first and last frame of the cycle
  (``cycle_NNNN_<outcome>_before.png`` / ``_after.png``) and the observations as
  JSON, so each action can be checked by eye afterwards.

``summary.json`` holds latency mean / median / p95 (all observations and in-game
only), launches per observation, OpenCV / fallback counts, UNKNOWN field readings
and outcome counts; ``observations.jsonl`` and
``cycles.jsonl`` the raw records.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2

from aoe2_ai.perception.profile import NOT_REQUESTED

from .observation import Observation

# Outcomes where no key was pressed.
PASSIVE = {"waited", "blocked", "dry_run", "halted", "stopped"}
# Numeric HUD fields recorded per observation (the ones HybridOcr can read).
NUMERIC_FIELDS = ("wood", "food", "gold", "stone", "population", "game_time")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def latency_summary(ms: list[float]) -> dict[str, float]:
    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms) if ms else 0.0,
        "median_ms": statistics.median(ms) if ms else 0.0,
        "p95_ms": percentile(ms, 0.95),
        "max_ms": max(ms, default=0.0),
    }


class LiveMetrics:
    """Wraps an observer and a controller's ``cycle`` to record what happens."""

    def __init__(self, directory: str | Path, frame_every: float | None = 15.0) -> None:
        self.directory = Path(directory)
        self.frame_every = frame_every
        self._last_frame_t: float | None = None
        self._frames = 0
        self._ocr_stats = None  # HybridOcr.stats of the observer, if any
        self.directory.mkdir(parents=True, exist_ok=True)
        self.observations: list[dict[str, Any]] = []
        self.cycles: list[dict[str, Any]] = []
        self._cycle_obs: list[Observation] = []
        self._obs_file = (self.directory / "observations.jsonl").open("a", encoding="utf-8")
        self._cycle_file = (self.directory / "cycles.jsonl").open("a", encoding="utf-8")

    def attach(self, observer, controller) -> None:
        observe, cycle = observer.observe, controller.cycle
        ocr = getattr(getattr(observer, "perception", None), "ocr", None)
        self._ocr_stats = getattr(ocr, "stats", None)

        def recorded_observe() -> Observation:
            before = self._ocr_counts()
            obs = observe()
            after = self._ocr_counts()
            digits = None if before is None else {k: after[k] - before[k] for k in after}
            self._record_observation(obs, digits)
            return obs

        def recorded_cycle():
            self._cycle_obs = []
            report = cycle()
            self._record_cycle(report)
            return report

        observer.observe = recorded_observe
        # The controller calls self.observer.observe(); both point at the same observer object.
        controller.cycle = recorded_cycle

    def _ocr_counts(self) -> dict[str, int] | None:
        st = self._ocr_stats
        if st is None:
            return None
        return {"opencv": st.opencv, "fallback": st.fallback, "unsupported": st.unsupported}

    def _record_observation(self, obs: Observation, digits: dict[str, int] | None = None) -> None:
        self._cycle_obs.append(obs)
        fields = {}
        for name in NUMERIC_FIELDS:
            det = getattr(obs.state, name, None)
            if det is None or det.raw_text == NOT_REQUESTED:
                continue
            fields[name] = {"value": det.to_dict()["value"], "status": det.status.value, "raw": det.raw_text,
                            "conf": round(det.confidence, 3)}
        idle = obs.idle
        if idle.present.raw_text != NOT_REQUESTED and idle.present.is_valid:
            fields["idle_present"] = {"value": idle.present.value, "status": idle.present.status.value}
            if idle.present.value:  # the count is only read on the yellow button
                fields["idle_count"] = {"value": idle.count.value, "status": idle.count.status.value,
                                        "raw": idle.count.raw_text, "conf": round(idle.count.confidence, 3)}
        rec = {
            "t": obs.timestamp,
            "total_ms": obs.timings.get("total", 0.0) * 1000,
            "launches": obs.ocr_launches,
            "game_detected": obs.game_detected,
            "stages_ms": {k: round(v * 1000, 2) for k, v in obs.timings.items() if k != "total"},
            "digits": digits,
            "fields": fields,
        }
        if (self.frame_every is not None and obs.game_detected and obs.frame is not None
                and (self._last_frame_t is None or obs.timestamp - self._last_frame_t >= self.frame_every)):
            self._frames += 1
            name = f"frame_{self._frames:04d}.png"
            cv2.imwrite(str(self.directory / name), obs.frame.image)
            self._last_frame_t = obs.timestamp
            rec["frame"] = name
        self.observations.append(rec)
        self._obs_file.write(json.dumps(rec) + "\n")
        self._obs_file.flush()

    def _record_cycle(self, report) -> None:
        outcome = getattr(report.outcome, "value", str(report.outcome))
        rec: dict[str, Any] = {
            "cycle": len(self.cycles) + 1,
            "t": time.time(),
            "outcome": outcome,
            "decision": report.decision.reason if report.decision else "",
            "detail": report.detail,
            "observations": len(self._cycle_obs),
        }
        if outcome not in PASSIVE and self._cycle_obs:
            stem = f"cycle_{rec['cycle']:04d}_{outcome}"
            first, last = self._cycle_obs[0], self._cycle_obs[-1]
            for tag, obs in (("before", first), ("after", last)):
                if obs.frame is not None:
                    cv2.imwrite(str(self.directory / f"{stem}_{tag}.png"), obs.frame.image)
            (self.directory / f"{stem}.json").write_text(
                json.dumps({"cycle": rec, "before": first.to_dict(), "after": last.to_dict()}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            rec["frames"] = stem
        self.cycles.append(rec)
        self._cycle_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._cycle_file.flush()

    def summary(self) -> dict[str, Any]:
        all_ms = [o["total_ms"] for o in self.observations]
        game_ms = [o["total_ms"] for o in self.observations if o["game_detected"]]
        launches = [o["launches"] for o in self.observations if o["launches"] is not None]
        outcomes: dict[str, int] = {}
        for c in self.cycles:
            outcomes[c["outcome"]] = outcomes.get(c["outcome"], 0) + 1
        digits = [o["digits"] for o in self.observations if o.get("digits") is not None]
        opencv = sum(d["opencv"] for d in digits)
        fallback = sum(d["fallback"] for d in digits)
        unknown: dict[str, int] = {}
        readings = 0
        for o in self.observations:
            if not o["game_detected"]:
                continue
            for name, f in o.get("fields", {}).items():
                if "raw" not in f:
                    continue
                readings += 1
                if f["status"] != "valid":
                    unknown[name] = unknown.get(name, 0) + 1
        return {
            "observations": latency_summary(all_ms),
            "observations_in_game": latency_summary(game_ms),
            "launches_mean": statistics.fmean(launches) if launches else 0.0,
            "launches_total": sum(launches),
            "digit_reads": None if not digits else {
                "opencv": opencv, "fallback": fallback,
                "fallback_rate": fallback / (opencv + fallback) if opencv + fallback else 0.0,
            },
            "numeric_readings_in_game": readings,
            "numeric_not_valid_in_game": unknown,
            "frames_saved": self._frames,
            "cycles": len(self.cycles),
            "outcomes": outcomes,
        }

    def close(self) -> dict[str, Any]:
        summary = self.summary()
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._obs_file.close()
        self._cycle_file.close()
        return summary
