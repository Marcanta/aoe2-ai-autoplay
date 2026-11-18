"""Visual detection of the player's own sheep (herd animals) in the game world. OpenCV only.

How a frame is searched (all in screen pixels; see ``docs/villager_assignment.md``):

1. **World area only.** The top bar, the bottom panel, the minimap / score panel and a
   margin along the screen edges (the mouse there scrolls the camera) are masked out
   (:data:`HUD_EXCLUSIONS`). Nothing outside the world area is ever reported as a target.
2. **Candidates from the player colour.** A herd animal the player owns wears a collar in
   the player's colour (blue for player 1). Small, compact blobs of that colour in the
   world area are the candidates. This makes ownership structural: an enemy sheep (other
   collar colour) or a neutral one is never a candidate, so never a target.
3. **Template verification.** Around each candidate, masked normalised cross-correlation
   (``TM_CCOEFF_NORMED`` with the animal's silhouette as mask, so the grass behind the
   template does not count) against every template, mirrored, at several zoom scales.
   The best match must contain the collar blob.
4. **Classification.** A match is ACCEPTED only when its score is at least ``accept``,
   no carcass template scores close to it, its whole box lies in the world area, and it
   does not overlap another accepted animal. Everything else is reported as rejected,
   with the reason, and is never clicked.

Templates are PNG files with the silhouette in the alpha channel, in
``herd_templates/living/`` (and ``herd_templates/carcass/`` for dead animals, when
available). Make one from a screenshot with::

    python -m aoe2_ai.perception.object_detector --extract --image shot.png --box X,Y,W,H --name sheep_a

and check what the detector sees with::

    python -m aoe2_ai.perception.object_detector --image shot.png --save captures/sheep

This module only describes what is on screen; whether to click is the controller's job.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .game_state import Status

REFERENCE_SIZE = (1680, 1050)  # (width, height) the rectangles below are measured at
TEMPLATE_DIR = Path(__file__).with_name("herd_templates")

# Interface areas (x, y, w, h) at REFERENCE_SIZE where nothing in the world can be clicked.
# Measured on the 1680x1050 fixture frames; scaled to the frame size like the HUD regions.
HUD_EXCLUSIONS: dict[str, tuple[int, int, int, int]] = {
    "top_bar": (0, 0, 1680, 50),
    "top_right_menu": (1350, 0, 330, 80),  # menu buttons and their hanging ornaments
    "top_left_ornament": (0, 0, 45, 95),  # corner ornament + idle-unit portrait under it
    "bottom_panel": (0, 880, 1680, 170),  # command card, selection panel and their frame
    "minimap_and_scores": (1330, 820, 350, 230),
}
EDGE_MARGIN = 12  # px at the reference size; the mouse on a screen edge scrolls the camera


# --- world area ------------------------------------------------------------------
def world_mask(height: int, width: int, exclusions: dict[str, tuple[int, int, int, int]] = HUD_EXCLUSIONS,
               edge: int = EDGE_MARGIN) -> np.ndarray:
    """Boolean mask of the clickable game world for a ``width`` x ``height`` frame."""
    sx, sy = width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]
    mask = np.zeros((height, width), bool)
    ex, ey = max(1, round(edge * sx)), max(1, round(edge * sy))
    mask[ey:height - ey, ex:width - ex] = True
    for x, y, w, h in exclusions.values():
        x0, y0 = round(x * sx), round(y * sy)
        mask[max(0, y0):max(0, y0 + round(h * sy)), max(0, x0):max(0, x0 + round(w * sx))] = False
    return mask


def box_in_world(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """True when every pixel of ``box`` (x, y, w, h) is inside the world area."""
    x, y, w, h = box
    height, width = mask.shape
    if x < 0 or y < 0 or x + w > width or y + h > height or w <= 0 or h <= 0:
        return False
    return bool(mask[y:y + h, x:x + w].all())


def point_in_world(mask: np.ndarray, x: int, y: int) -> bool:
    height, width = mask.shape
    return 0 <= x < width and 0 <= y < height and bool(mask[y, x])


# --- templates -------------------------------------------------------------------
@dataclass(frozen=True)
class HerdTemplate:
    """One view of an animal: BGR pixels + silhouette mask (255 = animal)."""

    name: str
    kind: str  # "living" or "carcass"
    image: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    @property
    def anchor(self) -> tuple[float, float]:
        """Centroid of the silhouette, as fractions of (w, h): where to click."""
        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return 0.5, 0.5
        return float(xs.mean() + 0.5) / self.mask.shape[1], float(ys.mean() + 0.5) / self.mask.shape[0]

    def flipped(self) -> "HerdTemplate":
        return HerdTemplate(f"{self.name}~flip", self.kind, np.ascontiguousarray(self.image[:, ::-1]),
                            np.ascontiguousarray(self.mask[:, ::-1]))


def owner_mask(bgr: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8)) > 0


def extract_template(crop: np.ndarray, owner_low=(100, 100, 80), owner_high=(135, 255, 255),
                     max_saturation: int = 115) -> tuple[np.ndarray, np.ndarray] | None:
    """Silhouette of the animal in ``crop`` (on grass): (tight BGR image, mask), or None.

    Herd animals (grey/brown/white) are far less saturated than grass and dirt; the collar
    is the player colour. The largest such component is the animal.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    animal = (hsv[..., 1] < max_saturation) | owner_mask(crop, owner_low, owner_high)
    m = cv2.morphologyEx(animal.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[k]
    mask = np.where(labels[y:y + h, x:x + w] == k, 255, 0).astype(np.uint8)
    return np.ascontiguousarray(crop[y:y + h, x:x + w]), mask


def save_template(path: str | Path, image: np.ndarray, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.dstack([image, mask]))
    return path


def load_templates(directory: str | Path = TEMPLATE_DIR) -> list[HerdTemplate]:
    """``directory/living/*.png`` and ``directory/carcass/*.png`` (BGRA, alpha = silhouette)."""
    out: list[HerdTemplate] = []
    for kind in ("living", "carcass"):
        for path in sorted((Path(directory) / kind).glob("*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                raise ValueError(f"herd template {path} must be a BGRA PNG (alpha = silhouette)")
            out.append(HerdTemplate(path.stem, kind, np.ascontiguousarray(img[..., :3]),
                                    np.where(img[..., 3] > 127, 255, 0).astype(np.uint8)))
    return out


# --- results ---------------------------------------------------------------------
class Verdict(str, Enum):
    ACCEPTED = "accepted"
    LOW_CONFIDENCE = "low_confidence"  # looks somewhat like a sheep, not enough to click
    AMBIGUOUS = "ambiguous"  # living and carcass templates score too close
    CARCASS = "carcass"
    OVERLAP = "overlap"  # overlaps another detected animal: a click could hit either
    OUTSIDE_WORLD = "outside_world"  # touches the HUD, minimap or a screen edge


@dataclass(frozen=True)
class SheepCandidate:
    x: int  # click point (frame pixels)
    y: int
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float  # best living-template score
    carcass_score: float | None  # best carcass-template score, None without carcass templates
    template: str
    scale: float  # zoom scale of the best template
    verdict: Verdict
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict is Verdict.ACCEPTED

    @property
    def point(self) -> tuple[int, int]:
        return self.x, self.y

    def distance_to(self, x: float, y: float) -> float:
        return float(np.hypot(self.x - x, self.y - y))

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "box": list(self.box), "score": round(self.score, 3),
            "carcass_score": None if self.carcass_score is None else round(self.carcass_score, 3),
            "template": self.template, "scale": self.scale, "verdict": self.verdict.value, "reason": self.reason,
        }


@dataclass(frozen=True)
class SheepScan:
    """Every sheep-like thing in one frame.

    ``status`` VALID means the scan ran on a usable frame: an empty ``sheep`` then means
    "no sheep *visible now*", nothing more. UNKNOWN means the frame could not be scanned
    (no frame, wrong format, no templates): it says nothing about sheep at all.
    """

    timestamp: float
    status: Status
    reason: str
    sheep: tuple[SheepCandidate, ...] = ()  # accepted only, best score first
    rejected: tuple[SheepCandidate, ...] = ()
    candidates: int = 0  # player-colour blobs examined
    elapsed: float = 0.0

    @classmethod
    def unknown(cls, timestamp: float, reason: str) -> "SheepScan":
        return cls(timestamp, Status.UNKNOWN, reason)

    @property
    def is_valid(self) -> bool:
        return self.status is Status.VALID

    def near(self, x: float, y: float, tolerance: float) -> SheepCandidate | None:
        """The accepted sheep closest to (x, y) within ``tolerance`` px, or None."""
        best = min(self.sheep, key=lambda s: s.distance_to(x, y), default=None)
        return best if best is not None and best.distance_to(x, y) <= tolerance else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp, "status": self.status.value, "reason": self.reason,
            "sheep": [s.to_dict() for s in self.sheep], "rejected": [s.to_dict() for s in self.rejected],
            "candidates": self.candidates, "elapsed_ms": round(self.elapsed * 1000, 1),
        }


# --- detector ----------------------------------------------------------------------
@dataclass(frozen=True)
class SheepDetectorConfig:
    """Thresholds. Sizes are pixels at REFERENCE_SIZE and the default zoom (scale 1.0)."""

    owner_hsv_low: tuple[int, int, int] = (100, 100, 80)  # player 1 blue (OpenCV HSV, H in 0..179)
    owner_hsv_high: tuple[int, int, int] = (135, 255, 255)
    collar_min_area: int = 6
    collar_max_area: int = 250
    collar_max_side: int = 30
    scales: tuple[float, ...] = (0.7, 0.85, 1.0, 1.2, 1.4)  # camera zoom levels searched
    work_scale: float = 0.5  # matching runs on a downscaled copy (4x fewer pixels)
    accept: float = 0.72  # living-template score to accept a sheep
    report_low: float = 0.55  # below this a candidate is not reported at all (not sheep-like)
    probe_min: float = 0.45  # one-zoom probe score under which the full zoom sweep is skipped
    carcass_margin: float = 0.05  # carcass within this of the living score -> ambiguous
    max_saturation: int = 115  # "animal" pixels for the body pre-filter
    min_body_fraction: float = 0.35  # body touching the collar: at least this x the smallest silhouette
    max_body_factor: float = 2.0  # ... and at most this x the largest one (two animals touching still pass)
    size_slack: float = 1.3  # a template may be up to this much larger than the body blob it is matched on
    merge_distance: int = 14  # collar blobs closer than this are one candidate
    overlap_iou: float = 0.15  # accepted animals overlapping more than this are both rejected
    duplicate_iou: float = 0.5  # matches overlapping more than this are one animal
    max_candidates: int = 80


@dataclass
class _Prepared:
    template: HerdTemplate
    scale: float
    image: np.ndarray  # at work scale
    mask: np.ndarray
    full_size: tuple[int, int]  # (w, h) in frame pixels


class SheepDetector:
    """Finds the player's own living sheep in the world area of a frame."""

    def __init__(self, config: SheepDetectorConfig = SheepDetectorConfig(),
                 templates: list[HerdTemplate] | None = None) -> None:
        self.config = config
        base = load_templates() if templates is None else list(templates)
        self.templates = base + [t.flipped() for t in base]
        self._prepared: dict[tuple[int, int], list[_Prepared]] = {}
        self._worlds: dict[tuple[int, int], np.ndarray] = {}

    @property
    def has_living(self) -> bool:
        return any(t.kind == "living" for t in self.templates)

    # preparation, per frame size ---------------------------------------------------
    def _prepare(self, width: int, height: int) -> list[_Prepared]:
        key = (width, height)
        if key not in self._prepared:
            ui = width / REFERENCE_SIZE[0]
            ws = self.config.work_scale
            out = []
            for t in self.templates:
                for s in self.config.scales:
                    f = ui * s
                    fw, fh = max(1, round(t.size[0] * f)), max(1, round(t.size[1] * f))
                    ww, wh = max(4, round(fw * ws)), max(4, round(fh * ws))
                    img = cv2.resize(t.image, (ww, wh), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(t.mask, (ww, wh), interpolation=cv2.INTER_NEAREST)
                    if (mask > 0).sum() < 12:
                        continue
                    out.append(_Prepared(t, s, img, mask, (fw, fh)))
            self._prepared[key] = out
        return self._prepared[key]

    # main entry -------------------------------------------------------------------------
    def detect(self, frame: np.ndarray | None, timestamp: float | None = None) -> SheepScan:
        ts = time.time() if timestamp is None else timestamp
        started = time.perf_counter()
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            return SheepScan.unknown(ts, "no usable frame")
        if not self.has_living:
            return SheepScan.unknown(ts, "no living sheep templates loaded")
        cfg = self.config
        height, width = frame.shape[:2]
        world = self._world(height, width)
        prepared = self._prepare(width, height)
        if not prepared:
            return SheepScan.unknown(ts, "templates too small at this frame size")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        centres = self._candidates(hsv, world)
        truncated = len(centres) > cfg.max_candidates
        centres = centres[: cfg.max_candidates]

        found: list[SheepCandidate] = []
        for cx, cy, bw, bh in centres:
            cand = self._verify(frame, world, prepared, cx, cy, bw, bh)
            if cand is not None:
                found.append(cand)
        sheep, rejected = self._classify(found)
        reason = f"{len(sheep)} sheep, {len(rejected)} rejected, {len(centres)} candidates"
        if truncated:
            reason += f" (candidate limit {cfg.max_candidates} reached: some sheep may be missed)"
        return SheepScan(ts, Status.VALID, reason, tuple(sheep), tuple(rejected), len(centres), time.perf_counter() - started)

    def _world(self, height: int, width: int) -> np.ndarray:
        key = (width, height)
        if key not in self._worlds:
            self._worlds[key] = world_mask(height, width)
        return self._worlds[key]

    def _candidates(self, hsv: np.ndarray, world: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Centres of compact player-colour blobs in the world area that touch an animal-sized
        low-saturation body, merged when close.

        The body test is what keeps a base cheap to scan: villagers (small white/blue figures)
        and buildings (huge low-saturation areas) carry the player colour too, but not next
        to a body the size of a herd animal.
        """
        cfg = self.config
        ui = hsv.shape[1] / REFERENCE_SIZE[0]
        lo, hi = min(cfg.scales), max(cfg.scales)
        min_area = cfg.collar_min_area * (ui * lo) ** 2
        max_area = cfg.collar_max_area * (ui * hi) ** 2
        max_side = cfg.collar_max_side * ui * hi
        living = [t for t in self.templates if t.kind == "living"]
        sil = [int((t.mask > 0).sum()) for t in living]
        body_min = cfg.min_body_fraction * min(sil) * (ui * lo) ** 2
        body_max = cfg.max_body_factor * max(sil) * (ui * hi) ** 2
        body_side = cfg.max_body_factor * max(max(t.size) for t in living) * ui * hi

        owner = cv2.inRange(hsv, np.array(cfg.owner_hsv_low, np.uint8), np.array(cfg.owner_hsv_high, np.uint8))
        owner[~world] = 0
        n, _, stats, centroids = cv2.connectedComponentsWithStats(owner)
        body = ((hsv[..., 1] < cfg.max_saturation) & world & (owner == 0)).astype(np.uint8)
        _, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(body)
        height, width = owner.shape
        reach = max(2, round(3 * ui))
        blobs = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if not (min_area <= area <= max_area and w <= max_side and h <= max_side):
                continue
            ring = body_labels[max(0, y - reach):min(height, y + h + reach), max(0, x - reach):min(width, x + w + reach)]
            ids, counts = np.unique(ring[ring > 0], return_counts=True)
            if ids.size == 0:
                continue
            b = body_stats[int(ids[np.argmax(counts)])]
            if body_min <= b[cv2.CC_STAT_AREA] <= body_max and max(b[cv2.CC_STAT_WIDTH], b[cv2.CC_STAT_HEIGHT]) <= body_side:
                blobs.append((int(area), int(round(centroids[k][0])), int(round(centroids[k][1])),
                              int(b[cv2.CC_STAT_WIDTH]), int(b[cv2.CC_STAT_HEIGHT])))
        blobs.sort(reverse=True)
        merged: list[tuple[int, int, int, int]] = []
        dist = cfg.merge_distance * ui
        for _, x, y, bw, bh in blobs:
            if all(np.hypot(x - m[0], y - m[1]) > dist for m in merged):
                merged.append((x, y, bw, bh))
        return merged

    def _verify(self, frame, world, prepared: list[_Prepared], cx: int, cy: int, bw: int, bh: int) -> SheepCandidate | None:
        """Best template match around one collar. ``bw`` x ``bh`` is the body blob touching it:
        it bounds the zoom, so only templates of a plausible size are tried."""
        cfg = self.config
        height, width = frame.shape[:2]
        slack = cfg.size_slack
        fit = [p for p in prepared
               if p.full_size[0] <= slack * bw and p.full_size[1] <= slack * bh  # not bigger than the body
               and p.full_size[0] * p.full_size[1] * slack * slack >= bw * bh / cfg.max_body_factor]  # nor far smaller
        if not fit:
            return None
        max_w = max(p.full_size[0] for p in fit)
        max_h = max(p.full_size[1] for p in fit)
        x0, y0 = max(0, cx - max_w), max(0, cy - max_h)
        x1, y1 = min(width, cx + max_w), min(height, cy + max_h)
        ws = cfg.work_scale
        win = cv2.resize(frame[y0:y1, x0:x1], (max(1, round((x1 - x0) * ws)), max(1, round((y1 - y0) * ws))),
                         interpolation=cv2.INTER_AREA)
        best: dict[str, tuple[float, tuple[int, int, int, int], _Prepared]] = {}
        lx, ly = (cx - x0) * ws, (cy - y0) * ws  # collar, in window work pixels

        def match(p: _Prepared) -> None:
            th, tw = p.image.shape[:2]
            if win.shape[0] < th or win.shape[1] < tw:
                return
            res = cv2.matchTemplate(win, p.image, cv2.TM_CCOEFF_NORMED, mask=p.mask)
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            # Only placements whose box contains the collar blob.
            xs0, xs1 = max(0, int(np.floor(lx - tw)) + 1), min(res.shape[1], int(np.ceil(lx)) + 1)
            ys0, ys1 = max(0, int(np.floor(ly - th)) + 1), min(res.shape[0], int(np.ceil(ly)) + 1)
            if xs1 <= xs0 or ys1 <= ys0:
                return
            sub = res[ys0:ys1, xs0:xs1]
            j = int(np.argmax(sub))
            score = float(sub.flat[j])
            my, mx = divmod(j, sub.shape[1])
            box = (x0 + round((xs0 + mx) / ws), y0 + round((ys0 + my) / ws), *p.full_size)
            kind = p.template.kind
            if kind not in best or score > best[kind][0]:
                best[kind] = (score, box, p)

        living = [p for p in fit if p.template.kind == "living"]
        if not living:
            return None
        # Probe: every living view at the two zooms closest to the body size (two, because
        # touching animals make one bigger body). Most non-animals (villager crowds, house
        # fronts) stop here, before the full sweep.
        sizes = {p.scale: p.full_size[0] * p.full_size[1] for p in living}
        probe = sorted(sizes, key=lambda s: abs(np.log(max(1, bw * bh) / max(1, sizes[s]))))[:2]
        for p in living:
            if p.scale in probe:
                match(p)
        if best.get("living", (-1.0,))[0] < cfg.probe_min:
            return None
        for p in living:
            if p.scale not in probe:
                match(p)
        if best["living"][0] >= cfg.report_low:
            for p in fit:
                if p.template.kind != "living":
                    match(p)
        if "living" not in best:
            return None
        score, box, p = best["living"]
        if score < cfg.report_low:
            return None
        carcass = best["carcass"][0] if "carcass" in best else None
        ax, ay = p.template.anchor
        px, py = int(round(box[0] + ax * box[2])), int(round(box[1] + ay * box[3]))
        verdict, reason = Verdict.ACCEPTED, f"score {score:.2f}"
        if carcass is not None and carcass > score:
            verdict, reason = Verdict.CARCASS, f"carcass {carcass:.2f} > living {score:.2f}"
        elif carcass is not None and carcass >= score - cfg.carcass_margin:
            verdict, reason = Verdict.AMBIGUOUS, f"living {score:.2f} vs carcass {carcass:.2f}"
        elif score < cfg.accept:
            verdict, reason = Verdict.LOW_CONFIDENCE, f"score {score:.2f} < {cfg.accept:.2f}"
        elif not box_in_world(world, box) or not point_in_world(world, px, py):
            verdict, reason = Verdict.OUTSIDE_WORLD, "box touches the HUD, minimap or a screen edge"
        return SheepCandidate(px, py, box, score, carcass, p.template.name, p.scale, verdict, reason)

    def _classify(self, found: list[SheepCandidate]) -> tuple[list[SheepCandidate], list[SheepCandidate]]:
        cfg = self.config
        found = sorted(found, key=lambda c: c.score, reverse=True)
        # One animal matched from two collar blobs: keep the best.
        unique: list[SheepCandidate] = []
        for c in found:
            if all(iou(c.box, u.box) <= cfg.duplicate_iou for u in unique):
                unique.append(c)
        accepted = [c for c in unique if c.accepted]
        rejected = [c for c in unique if not c.accepted]
        clear = []
        for c in accepted:
            others = [o for o in unique if o is not c and o.verdict is not Verdict.LOW_CONFIDENCE]
            hit = next((o for o in others if iou(c.box, o.box) > cfg.overlap_iou), None)
            if hit is not None:
                rejected.append(SheepCandidate(c.x, c.y, c.box, c.score, c.carcass_score, c.template, c.scale,
                                               Verdict.OVERLAP, f"overlaps another animal at ({hit.x}, {hit.y})"))
            else:
                clear.append(c)
        return clear, rejected


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# --- debug ------------------------------------------------------------------------------
COLORS = {Verdict.ACCEPTED: (0, 255, 0), Verdict.LOW_CONFIDENCE: (0, 200, 255), Verdict.AMBIGUOUS: (0, 140, 255),
          Verdict.CARCASS: (0, 0, 255), Verdict.OVERLAP: (255, 0, 255), Verdict.OUTSIDE_WORLD: (128, 128, 128)}


def draw_scan(frame: np.ndarray, scan: SheepScan) -> np.ndarray:
    out = frame.copy()
    world = world_mask(*frame.shape[:2])
    out[~world] = (out[~world] * 0.5).astype(np.uint8)
    for c in (*scan.sheep, *scan.rejected):
        x, y, w, h = c.box
        color = COLORS[c.verdict]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.drawMarker(out, (c.x, c.y), color, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(out, f"{c.verdict.value} {c.score:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def main(argv: list[str] | None = None) -> None:
    import json

    parser = argparse.ArgumentParser(description="Find the player's own sheep in a screenshot, or make a template.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--save", type=Path, help="write the overlay and scan JSON to this directory")
    parser.add_argument("--extract", action="store_true", help="cut a template from --box and save it as --name")
    parser.add_argument("--box", help="X,Y,W,H of one animal (with --extract)")
    parser.add_argument("--name", help="template name (with --extract)")
    parser.add_argument("--kind", choices=("living", "carcass"), default="living")
    args = parser.parse_args(argv)

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")
    if args.extract:
        if not args.box or not args.name:
            raise SystemExit("--extract needs --box and --name")
        x, y, w, h = (int(v) for v in args.box.split(","))
        result = extract_template(frame[y:y + h, x:x + w])
        if result is None:
            raise SystemExit("no animal silhouette found in that box")
        path = save_template(TEMPLATE_DIR / args.kind / f"{args.name}.png", *result)
        print(f"saved {path} ({result[0].shape[1]}x{result[0].shape[0]}, {int((result[1] > 0).sum())} silhouette px)")
        return

    scan = SheepDetector().detect(frame)
    print(json.dumps(scan.to_dict(), indent=2))
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save / f"{args.image.stem}_sheep.png"), draw_scan(frame, scan))
        (args.save / f"{args.image.stem}_sheep.json").write_text(json.dumps(scan.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
