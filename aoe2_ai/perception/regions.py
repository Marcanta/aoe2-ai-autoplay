"""Configurable screen regions (ROIs) of the AoE2 DE interface.

Regions are stored in JSON in the pixel space of a reference resolution and
scaled proportionally to the actual frame size. Edit the JSON (see
``python -m aoe2_ai.perception --calibrate``) to adjust them.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

DEFAULT_REGIONS_PATH = Path(__file__).with_name("regions_1680x1050.json")


@dataclass(frozen=True)
class Region:
    """A rectangle ``(x, y, w, h)`` plus how to read it.

    ``kind`` tells the detectors how to interpret the text: ``digits``,
    ``fraction`` (``4/5``), ``clock`` (``00:12:34``), ``text``, or ``icon``.
    ``calibrated`` is False for placeholder rectangles that were never checked
    against a real frame; detectors report those fields as UNKNOWN.
    ``dark_text`` is True where text is dark on a light background (the
    parchment selection panel) instead of light on dark (the top bar).
    """

    name: str
    x: int
    y: int
    w: int
    h: int
    kind: str = "text"
    calibrated: bool = True
    dark_text: bool = False

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def scaled(self, sx: float, sy: float) -> "Region":
        return replace(self, x=round(self.x * sx), y=round(self.y * sy), w=max(1, round(self.w * sx)), h=max(1, round(self.h * sy)))

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Pixels of this region, clipped to the frame (possibly empty)."""
        height, width = frame.shape[:2]
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(width, self.x + self.w), min(height, self.y + self.h)
        if x1 <= x0 or y1 <= y0:
            return frame[0:0, 0:0]
        return frame[y0:y1, x0:x1]


@dataclass(frozen=True)
class RegionSet:
    reference_size: tuple[int, int]  # (width, height) the coordinates are written for
    regions: dict[str, Region]

    def __getitem__(self, name: str) -> Region:
        return self.regions[name]

    def __contains__(self, name: str) -> bool:
        return name in self.regions

    def __iter__(self):
        return iter(self.regions.values())

    def for_size(self, width: int, height: int) -> "RegionSet":
        """Regions scaled to a frame of ``width`` x ``height``."""
        ref_w, ref_h = self.reference_size
        if (width, height) == (ref_w, ref_h):
            return self
        sx, sy = width / ref_w, height / ref_h
        if abs(sx - sy) / max(sx, sy) > 0.02:
            warnings.warn(
                f"frame {width}x{height} has a different aspect ratio than the region reference "
                f"{ref_w}x{ref_h}; HUD regions may be off, recalibrate them",
                stacklevel=2,
            )
        return RegionSet((width, height), {n: r.scaled(sx, sy) for n, r in self.regions.items()})

    def for_frame(self, frame: np.ndarray) -> "RegionSet":
        height, width = frame.shape[:2]
        return self.for_size(width, height)

    # --- persistence ----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_REGIONS_PATH) -> "RegionSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ref = tuple(data["reference_size"])
        regions = {name: Region(name=name, **spec) for name, spec in data["regions"].items()}
        return cls((int(ref[0]), int(ref[1])), regions)

    @classmethod
    def default(cls) -> "RegionSet":
        return cls.load(DEFAULT_REGIONS_PATH)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        regions = {}
        for name, region in self.regions.items():
            spec = asdict(region)
            del spec["name"]
            regions[name] = spec
        data = {"reference_size": list(self.reference_size), "regions": regions}
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path
