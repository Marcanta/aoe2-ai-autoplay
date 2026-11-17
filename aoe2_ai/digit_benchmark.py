"""Region-level benchmark: OpenCV digit recognizer vs Tesseract on the numeric HUD fields.

    python -m aoe2_ai.digit_benchmark                                  # labelled fixtures, 3 passes
    python -m aoe2_ai.digit_benchmark --extra captures --json docs/benchmarks/digits.json

Every labelled numeric field of ``tests/fixtures/labels.json`` (wood, food,
gold, stone, population, game clock, idle-villager count) is cropped exactly as
the detectors crop it and read by three engines with the detectors' whitelist:

* ``opencv``    - :class:`DigitRecognizer` alone (no Tesseract).
* ``tesseract`` - the existing multi-pass :class:`TesseractOcr`.
* ``hybrid``    - :class:`HybridOcr`: OpenCV, Tesseract when OpenCV says UNKNOWN.

A read counts as *exact* when it is accepted (confidence >= the detectors'
0.6 and parsed by the detectors' parser) and equal to the label, *wrong* when it
is accepted and differs, *unknown* otherwise. *Negatives* are crops that must
not produce a value (the main menu, fields labelled ``null``).

``opencv (leave-one-out)`` rebuilds the templates without the frame being read,
so no glyph is matched against a template cut from its own screenshot.
``--extra DIR`` adds an agreement check on unlabelled screenshots (e.g. the
live-run captures): fields where both engines give a value must agree.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from aoe2_ai.perception.detectors import DEFAULT_MIN_CONFIDENCE, parse_clock, parse_population, parse_resource
from aoe2_ai.perception.digit_recognizer import (
    DigitRecognizer,
    HybridOcr,
    build_templates,
    labelled_crops,
)
from aoe2_ai.perception.ocr import OcrEngine, TesseractOcr
from aoe2_ai.perception.regions import RegionSet
from aoe2_ai.perception.villager import white_only

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
WHITELIST = {"digits": "0123456789", "fraction": "0123456789/", "clock": "0123456789:"}
FIELDS = {  # field -> kind
    "wood": "digits", "food": "digits", "gold": "digits", "stone": "digits",
    "population": "fraction", "game_time": "clock", "idle_count": "digits",
}


def _parse(kind: str, text: str) -> Any:
    if kind == "digits":
        return parse_resource(text)
    if kind == "fraction":
        p = parse_population(text)
        return None if p is None else (p.current, p.cap)
    return parse_clock(text)


def _plain(kind: str, value: Any) -> Any:
    return tuple(value) if kind == "fraction" else value


def accepted_value(engine: OcrEngine, crop: np.ndarray, kind: str) -> Any:
    """What a detector would accept from ``engine`` for this crop: a parsed value or None."""
    r = engine.read(crop, whitelist=WHITELIST[kind])
    if r.empty or r.confidence < DEFAULT_MIN_CONFIDENCE:
        return None
    return _parse(kind, r.text)


class OpenCvOnly:
    """The recognizer as an OCR engine without fallback (UNKNOWN -> empty text)."""

    def __init__(self, recognizer: DigitRecognizer) -> None:
        self.engine = HybridOcr(None, recognizer)

    def read(self, crop, **kw):
        return self.engine.read(crop, **kw)


@dataclass
class Tally:
    exact: int = 0
    wrong: int = 0
    unknown: int = 0
    negatives_ok: int = 0
    negatives_wrong: int = 0
    ms: list[float] = field(default_factory=list)
    wrong_cases: list[str] = field(default_factory=list)

    def add(self, got: Any, want: Any, ms: float | None, case: str) -> None:
        if ms is not None:
            self.ms.append(ms)
        if want is None:
            if got is None:
                self.negatives_ok += 1
            else:
                self.negatives_wrong += 1
                self.wrong_cases.append(f"{case}: {got!r} where nothing may be read")
        elif got is None:
            self.unknown += 1
        elif got == want:
            self.exact += 1
        else:
            self.wrong += 1
            self.wrong_cases.append(f"{case}: {got!r} != {want!r}")

    def summary(self) -> dict[str, Any]:
        total = self.exact + self.wrong + self.unknown
        ms = sorted(self.ms)
        p95 = ms[min(len(ms) - 1, max(0, round(0.95 * len(ms)) - 1))] if ms else 0.0
        return {
            "fields": total,
            "exact": self.exact,
            "wrong": self.wrong,
            "unknown": self.unknown,
            "exact_rate": self.exact / total if total else 0.0,
            "unknown_rate": self.unknown / total if total else 0.0,
            "negatives_ok": self.negatives_ok,
            "negatives_wrong": self.negatives_wrong,
            "mean_ms": statistics.fmean(ms) if ms else 0.0,
            "median_ms": statistics.median(ms) if ms else 0.0,
            "p95_ms": p95,
            "wrong_cases": self.wrong_cases,
        }


def negative_crops(labels: dict, frames_dir: Path) -> list[tuple[str, str, str, np.ndarray]]:
    """(frame, field, kind, crop) that must not be read as a value: HUD absent or field labelled null."""
    regions = RegionSet.default()
    out = []
    for name, label in sorted(labels.items()):
        frame = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        rs = regions.for_frame(frame)
        for fname, kind in FIELDS.items():
            if fname == "idle_count":
                null = label.get("hud", True) is False or (label.get("_idle") or {}).get("count", 0) is None
            else:
                null = label.get("hud", True) is False or (fname in label and label[fname] is None)
            if null:
                crop = rs[fname].crop(frame)
                out.append((name, fname, kind, white_only(crop) if fname == "idle_count" else crop))
    return out


def run(labels: dict, frames_dir: Path, repeat: int, extra: Path | None) -> dict[str, Any]:
    positives = [(n, f, k, c, _plain(k, v)) for n, f, k, c, v in labelled_crops(labels, frames_dir)]
    negatives = negative_crops(labels, frames_dir)
    print(f"{len(positives)} labelled numeric fields, {len(negatives)} negative crops, repeat {repeat}", flush=True)

    tess = TesseractOcr()
    recognizer = DigitRecognizer()
    hybrid = HybridOcr(tess, recognizer)
    engines: dict[str, OcrEngine] = {"opencv": OpenCvOnly(recognizer), "tesseract": tess, "hybrid": hybrid}
    for engine in engines.values():  # warm-up (first Tesseract launch, numpy/cv2 first calls)
        accepted_value(engine, positives[0][3], positives[0][2])
    _plus((0, 0), hybrid.stats)  # reset the warm-up counts

    tallies: dict[str, dict[str, Tally]] = {e: {"all": Tally()} for e in engines}
    launches = {e: 0 for e in engines}  # Tesseract launches on the labelled fields
    hybrid_counts = {"labelled": (0, 0), "negatives": (0, 0)}  # (answered by OpenCV, sent to Tesseract)
    for engine_name, engine in engines.items():
        for _ in range(repeat):
            for name, fname, kind, crop, want in positives:
                t0, launches0 = time.perf_counter(), tess.launches
                got = accepted_value(engine, crop, kind)
                ms = (time.perf_counter() - t0) * 1000
                launches[engine_name] += tess.launches - launches0
                for key in ("all", fname):
                    tallies[engine_name].setdefault(key, Tally()).add(got, want, ms, f"{name} {fname}")
            if engine_name == "hybrid":
                hybrid_counts["labelled"] = _plus(hybrid_counts["labelled"], hybrid.stats)
            for name, fname, kind, crop in negatives:
                got = accepted_value(engine, crop, kind)
                tallies[engine_name]["all"].add(got, None, None, f"{name} {fname}")
            if engine_name == "hybrid":
                hybrid_counts["negatives"] = _plus(hybrid_counts["negatives"], hybrid.stats)

    # Leave one frame out: templates without the frame being read.
    loo = Tally()
    for frame_name in sorted({p[0] for p in positives}):
        held_out = DigitRecognizer(build_templates(labels, frames_dir, exclude=[frame_name]))
        engine = OpenCvOnly(held_out)
        for name, fname, kind, crop, want in positives:
            if name == frame_name:
                loo.add(accepted_value(engine, crop, kind), want, None, f"{name} {fname}")

    reads = repeat * len(positives)
    results: dict[str, Any] = {
        "engines": {e: {k: t.summary() for k, t in by.items()} for e, by in tallies.items()},
        "opencv_leave_one_out": loo.summary(),
        "tesseract_launches_per_field": {e: n / max(1, reads) for e, n in launches.items()},
        "hybrid_fallback": {
            part: {"opencv": a, "fallback": b, "fallback_rate": b / (a + b) if a + b else 0.0}
            for part, (a, b) in hybrid_counts.items()
        },
    }
    if extra is not None:
        results["agreement"] = agreement(extra, recognizer, tess)
    return results


def _plus(acc: tuple[int, int], stats) -> tuple[int, int]:
    """Add the HybridOcr counters accumulated since the last call to ``acc`` and reset them."""
    out = (acc[0] + stats.opencv, acc[1] + stats.fallback)
    stats.opencv = stats.fallback = stats.unsupported = 0
    return out


def agreement(directory: Path, recognizer: DigitRecognizer, tess: TesseractOcr) -> dict[str, Any]:
    """Unlabelled screenshots: count fields where OpenCV and Tesseract agree / disagree / only one reads."""
    regions = RegionSet.default()
    opencv = OpenCvOnly(recognizer)
    out = {"frames": 0, "both_agree": 0, "disagree": 0, "opencv_only": 0, "tesseract_only": 0, "neither": 0, "disagreements": []}
    for path in sorted(directory.rglob("*.png")):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (regions.reference_size[1], regions.reference_size[0]):
            continue
        out["frames"] += 1
        rs = regions.for_frame(frame)
        for fname, kind in FIELDS.items():
            crop = rs[fname].crop(frame)
            crop = white_only(crop) if fname == "idle_count" else crop
            a, b = accepted_value(opencv, crop, kind), accepted_value(tess, crop, kind)
            if a is not None and b is not None:
                if a == b:
                    out["both_agree"] += 1
                else:
                    out["disagree"] += 1
                    out["disagreements"].append(f"{path.name} {fname}: opencv {a!r} tesseract {b!r}")
            elif a is not None:
                out["opencv_only"] += 1
            elif b is not None:
                out["tesseract_only"] += 1
            else:
                out["neither"] += 1
    return out


def format_results(r: dict[str, Any]) -> str:
    lines = [f"{'engine':<26} {'fields':>6} {'exact':>6} {'wrong':>6} {'unknown':>8} {'neg wrong':>9} {'mean ms':>8} {'median':>7} {'p95':>7}"]

    def row(name: str, s: dict[str, Any]) -> str:
        return (f"{name:<26} {s['fields']:>6} {s['exact']:>6} {s['wrong']:>6} {s['unknown']:>8} {s['negatives_wrong']:>9} "
                f"{s['mean_ms']:>8.2f} {s['median_ms']:>7.2f} {s['p95_ms']:>7.2f}")

    for engine, by in r["engines"].items():
        lines.append(row(engine, by["all"]))
    lines.append(row("opencv (leave-one-out)", r["opencv_leave_one_out"]))
    for part, hf in r["hybrid_fallback"].items():
        lines.append(f"hybrid fallback to Tesseract, {part}: {hf['fallback']}/{hf['opencv'] + hf['fallback']} reads ({hf['fallback_rate']:.1%})")
    lines.append("Tesseract launches per labelled field: " + ", ".join(f"{e} {n:.2f}" for e, n in r["tesseract_launches_per_field"].items()))
    lines.append("")
    lines.append("per field (exact / unknown / wrong, median ms):")
    for fname in FIELDS:
        cells = []
        for engine, by in r["engines"].items():
            s = by.get(fname)
            if s:
                cells.append(f"{engine} {s['exact']}/{s['unknown']}/{s['wrong']} {s['median_ms']:.2f} ms")
        lines.append(f"  {fname:<11} " + "   ".join(cells))
    for engine, by in r["engines"].items():
        for case in by["all"]["wrong_cases"]:
            lines.append(f"WRONG [{engine}] {case}")
    if "agreement" in r:
        a = r["agreement"]
        lines.append(f"agreement on {a['frames']} unlabelled frames: agree {a['both_agree']}, disagree {a['disagree']}, "
                     f"opencv only {a['opencv_only']}, tesseract only {a['tesseract_only']}, neither {a['neither']}")
        lines.extend(f"DISAGREE {d}" for d in a["disagreements"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the OpenCV digit recognizer with Tesseract per HUD field.")
    parser.add_argument("--labels", type=Path, default=FIXTURES / "labels.json")
    parser.add_argument("--frames", type=Path, default=FIXTURES / "frames")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--extra", type=Path, help="directory of unlabelled screenshots for an agreement check")
    parser.add_argument("--json", type=Path, help="write the results to this file")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["frames"]
    results = run(labels, args.frames, args.repeat, args.extra)
    print(format_results(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
