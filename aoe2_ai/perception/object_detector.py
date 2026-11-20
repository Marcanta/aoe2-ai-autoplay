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
        """Where to click, as fractions of (w, h): the point of the body farthest from both the
        silhouette's edge and the collar. On an animal facing the camera the collar sits at the
        centroid; clicking the body away from it keeps the click-point check (no player colour
        under the click) meaningful."""
        body = (self.mask > 0).astype(np.uint8)
        if not body.any():
            return 0.5, 0.5
        collar = owner_mask(self.image, (100, 100, 80), (135, 255, 255)) & (body > 0)
        if collar.any():
            near = cv2.dilate(collar.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            away = body.copy()
            away[near] = 0
            if away.sum() >= 0.2 * body.sum():
                body = away
        dist = cv2.distanceTransform(np.pad(body, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return (x + 0.5) / self.mask.shape[1], (y + 0.5) / self.mask.shape[0]

    def flipped(self) -> "HerdTemplate":
        return HerdTemplate(f"{self.name}~flip", self.kind, np.ascontiguousarray(self.image[:, ::-1]),
                            np.ascontiguousarray(self.mask[:, ::-1]))


def owner_mask(bgr: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8)) > 0


def animal_pixels(hsv: np.ndarray, ground_s, ground_v, margin: int = 30, dark_ratio: float = 0.7,
                  dark_margin: int = 10) -> np.ndarray:
    """Pixels that may belong to a herd animal, judged against the ground under them.

    An animal pixel is clearly less saturated than the ground (``margin`` below it: white
    sheep, pale pigs, grey geese), or clearly darker and somewhat less saturated (``dark_ratio``
    of its brightness, ``dark_margin`` below its saturation: buffalo, a goose's grey wing).
    ``ground_s`` / ``ground_v`` are scalars or per-pixel arrays (:func:`ground_hsv`).

    Every fixed threshold tried failed on some map live: S < 115 swallowed desert sand
    (S ~115), fixed S/V cuts missed a goose on savanna (ground S ~160), and S < 85 swallowed
    frosty winter ground (S ~75) around pigs. Shadows on grass and sand are *more* saturated
    than the ground, so they do not count.
    """
    sat, val = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    gs = np.asarray(ground_s, np.int16)
    gv = np.asarray(ground_v, np.float32)
    return (sat < gs - margin) | ((val < dark_ratio * gv) & (sat < gs - dark_margin))


def ground_hsv(hsv: np.ndarray, block: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel estimate of the ground's (saturation, value): the median of each ``block`` px tile.

    Animals are a small part of any tile, so the median is the terrain.
    """
    h, w = hsv.shape[:2]
    th, tw = -(-h // block), -(-w // block)
    out = []
    for ch in (1, 2):
        padded = np.pad(hsv[..., ch], ((0, th * block - h), (0, tw * block - w)), mode="edge")
        tiles = padded.reshape(th, block, tw, block).transpose(0, 2, 1, 3).reshape(th, tw, -1)
        med = np.median(tiles, axis=2).astype(np.float32)
        out.append(cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR))
    return out[0].astype(np.int16), out[1]


def extract_template(crop: np.ndarray, owner_low=(100, 100, 80), owner_high=(135, 255, 255),
                     margin: int = 30) -> tuple[np.ndarray, np.ndarray] | None:
    """Silhouette of the one animal in ``crop``: (tight BGR image, mask), or None.

    Only for cutting templates, where the crop shows one animal on its ground. The ground is
    the median of the crop border; animal pixels are :func:`animal_pixels` against it, plus the
    collar. Thin lines (tile grid, sand ripples, frost) are opened away, and the component
    holding the collar (player colour) is the animal; without a collar, the largest.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    edge = np.concatenate([hsv[0], hsv[-1], hsv[:, 0], hsv[:, -1]])
    ground_s, ground_v = (int(v) for v in np.median(edge[:, 1:], axis=0))
    collar = owner_mask(crop, owner_low, owner_high)
    animal = animal_pixels(hsv, ground_s, ground_v, margin) | collar
    m = cv2.morphologyEx(animal.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if collar.any():
        near = cv2.dilate(collar.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ids, counts = np.unique(labels[near & (labels > 0)], return_counts=True)
        if ids.size:
            k = int(ids[np.argmax(counts)])
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
    COLLAR_MISMATCH = "collar_mismatch"  # far more or less player colour than a collar: a rider, a villager...


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
    collar_min_fill: float = 0.25  # blob area / bounding box: collars are compact bands, trims are lines
    scales: tuple[float, ...] = (0.7, 0.85, 1.0, 1.2, 1.4)  # camera zoom levels searched
    work_scale: float = 0.5  # matching runs on a downscaled copy (4x fewer pixels)
    accept: float = 0.75  # living-template score to accept a sheep
    report_low: float = 0.55  # below this a candidate is not reported at all (not sheep-like)
    probe_min: float = 0.45  # probe score under which the full zoom sweep is skipped
    probe_scales: int = 2  # zooms closest to the body size tried in the probe
    carcass_margin: float = 0.05  # carcass within this of the living score -> ambiguous
    # Player colour inside the matched silhouette, relative to the template's own collar share.
    # Real animals: 0.88-1.17x; a scout's rider 1.9x; a villager matched by shape 0.08x.
    collar_share_band: tuple[float, float] = (0.5, 1.6)
    refine_radius: int = 3  # px: full-resolution re-scoring around the half-resolution match
    # Body pre-filter, against the local ground (animal_pixels, ground_hsv).
    ground_margin: int = 30  # animal: this much less saturated than the ground...
    dark_ratio: float = 0.7  # ... or darker than this x the ground's brightness
    dark_margin: int = 10  # ... and this much less saturated
    ground_block: int = 48  # px tiles over which the ground is the median
    min_body_fraction: float = 0.35  # body touching the collar: at least this x the smallest silhouette
    max_body_factor: float = 2.0  # ... and at most this x the largest one (two animals touching still pass)
    size_slack: float = 1.6  # a template may be up to this much larger than the body blob (legs and pale parts may not count as body)
    min_size_slack: float = 1.3  # ... and at most this much smaller (1.6 here let a pig template onto a house roof)
    merge_distance: int = 14  # collar blobs closer than this are one candidate
    overlap_iou: float = 0.15  # overlap (IoU) above which two detections count as overlapping
    allow_herd_overlap: bool = True  # overlapping accepted animals stay clickable (herd under the TC)
    click_clearance: int = 5  # px radius around the click point...
    max_owner_under_click: float = 0.2  # ... where player colour above this share means "not clear"
    # Collar isolation: bluish pixels (hue, min saturation) around a collar blob, within
    # collar_surround x its size, must stay under max_blue_surround (fog of war, water, frost).
    bluish_hue: tuple[int, int] = (95, 140)
    bluish_min_saturation: int = 60
    collar_surround: float = 1.5
    max_blue_surround: float = 0.15
    duplicate_iou: float = 0.5  # matches overlapping more than this are one animal
    max_candidates: int = 80


@dataclass
class _Prepared:
    template: HerdTemplate
    scale: float
    image: np.ndarray  # at work scale
    mask: np.ndarray
    full_size: tuple[int, int]  # (w, h) in frame pixels
    full: dict = field(default_factory=dict)  # (w, h) -> template image, mask at full resolution


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
        sheep, rejected = self._classify(found, hsv)
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
        ground_s, ground_v = ground_hsv(hsv, cfg.ground_block)
        animal = animal_pixels(hsv, ground_s, ground_v, cfg.ground_margin, cfg.dark_ratio, cfg.dark_margin)
        body = (animal & world & (owner == 0)).astype(np.uint8)
        _, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(body)
        height, width = owner.shape
        reach = max(2, round(3 * ui))
        # A collar is a small blue patch on a non-blue animal and ground. Blue fog of war (winter
        # maps), water and frost are blue all around: their blobs are dropped here. Integral
        # image of "bluish" pixels, so each check is O(1).
        hue, sat = hsv[..., 0], hsv[..., 1]
        bluish = ((hue >= cfg.bluish_hue[0]) & (hue <= cfg.bluish_hue[1]) & (sat > cfg.bluish_min_saturation)).astype(np.uint8)
        integral = cv2.integral(bluish, sdepth=cv2.CV_32S)
        blobs = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if not (min_area <= area <= max_area and w <= max_side and h <= max_side):
                continue
            if area < cfg.collar_min_fill * w * h:
                continue  # thin or diagonal line (a roof's blue trim fills 14% of its box; collars 33-71%)
            pad = max(reach, round(cfg.collar_surround * max(w, h)))
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
            blue_around = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) - area
            if blue_around > cfg.max_blue_surround * max(1, (x1 - x0) * (y1 - y0) - area):
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
               and p.full_size[0] * p.full_size[1] * cfg.min_size_slack ** 2 >= bw * bh / cfg.max_body_factor]  # nor far smaller
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
        probe = sorted(sizes, key=lambda s: abs(np.log(max(1, bw * bh) / max(1, sizes[s]))))[:cfg.probe_scales]
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
        # The half-resolution search finds where; the full-resolution score decides. At half
        # resolution a sheep template at the smallest zoom is ~4x8 px, and a villager's white
        # apron over a blue dress scored 0.72-0.78 there, 0.60-0.70 at full resolution.
        score, box = self._refine(frame, box, p)
        carcass = None
        if "carcass" in best:
            carcass = self._refine(frame, best["carcass"][1], best["carcass"][2])[0]
        ax, ay = p.template.anchor
        px, py = int(round(box[0] + ax * box[2])), int(round(box[1] + ay * box[3]))
        verdict, reason = Verdict.ACCEPTED, f"score {score:.2f}"
        if carcass is not None and carcass > score:
            verdict, reason = Verdict.CARCASS, f"carcass {carcass:.2f} > living {score:.2f}"
        elif carcass is not None and carcass >= score - cfg.carcass_margin:
            verdict, reason = Verdict.AMBIGUOUS, f"living {score:.2f} vs carcass {carcass:.2f}"
        elif score < cfg.accept:
            verdict, reason = Verdict.LOW_CONFIDENCE, f"score {score:.2f} < {cfg.accept:.2f}"
        elif (share := self._collar_share(frame, box, p)) is not None and not (
                cfg.collar_share_band[0] <= share <= cfg.collar_share_band[1]):
            verdict, reason = Verdict.COLLAR_MISMATCH, f"score {score:.2f} but {share:.2f}x the template's player colour"
        elif not box_in_world(world, box) or not point_in_world(world, px, py):
            verdict, reason = Verdict.OUTSIDE_WORLD, "box touches the HUD, minimap or a screen edge"
        return SheepCandidate(px, py, box, score, carcass, p.template.name, p.scale, verdict, reason)

    def _refine(self, frame: np.ndarray, box: tuple[int, int, int, int], p: _Prepared) -> tuple[float, tuple[int, int, int, int]]:
        """Masked score of ``p`` at full resolution, searched ``refine_radius`` px around ``box``."""
        x, y, w, h = box
        height, width = frame.shape[:2]
        r = max(1, round(self.config.refine_radius * width / REFERENCE_SIZE[0]))
        key = (w, h)
        if key not in p.full:
            p.full[key] = (cv2.resize(p.template.image, key, interpolation=cv2.INTER_AREA),
                           cv2.resize(p.template.mask, key, interpolation=cv2.INTER_NEAREST))
        img, mask = p.full[key]
        x0, y0 = max(0, x - r), max(0, y - r)
        win = frame[y0:min(height, y + h + r), x0:min(width, x + w + r)]
        if win.shape[0] < h or win.shape[1] < w or (mask > 0).sum() < 12:
            return -1.0, box
        res = np.nan_to_num(cv2.matchTemplate(win, img, cv2.TM_CCOEFF_NORMED, mask=mask), nan=-1.0, posinf=-1.0, neginf=-1.0)
        j = int(np.argmax(res))
        my, mx = divmod(j, res.shape[1])
        return float(res.flat[j]), (x0 + mx, y0 + my, w, h)

    def _collar_share(self, frame: np.ndarray, box: tuple[int, int, int, int], p: _Prepared) -> float | None:
        """Player-colour share inside the matched silhouette / the template's own share."""
        t = p.template
        own_t = owner_mask(t.image, self.config.owner_hsv_low, self.config.owner_hsv_high) & (t.mask > 0)
        ref = own_t.sum() / max(1, (t.mask > 0).sum())
        if ref <= 0:
            return None  # template without a visible collar: nothing to compare
        x, y, w, h = box
        height, width = frame.shape[:2]
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return None
        mask = cv2.resize(t.mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        own = owner_mask(frame[y:y + h, x:x + w], self.config.owner_hsv_low, self.config.owner_hsv_high) & mask
        return float(own.sum() / max(1, mask.sum()) / ref)

    def _classify(self, found: list[SheepCandidate], hsv: np.ndarray | None = None) -> tuple[list[SheepCandidate], list[SheepCandidate]]:
        """Split into clickable sheep and rejected ones.

        Overlapping *accepted* animals stay clickable (``allow_herd_overlap``): a herd pushed
        under the Town Center overlaps, and a click on either one sends the villager to a
        player's animal. An accepted animal overlapping something rejected that still looked
        animal-like (a rider, a carcass, an ambiguous match) is not clicked, nor one with the
        player colour right under its click point (a villager in front of it).
        """
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
            others = [o for o in unique if o is not c and o.verdict is not Verdict.LOW_CONFIDENCE
                      and not (cfg.allow_herd_overlap and o.accepted)]
            hit = next((o for o in others if iou(c.box, o.box) > cfg.overlap_iou), None)
            why = f"overlaps {'another animal' if hit is not None and hit.accepted else 'a rejected look-alike'}"                   f" at ({hit.x}, {hit.y})" if hit is not None else ""
            if not why and hsv is not None and self._colour_under(hsv, c.x, c.y) > cfg.max_owner_under_click:
                why = "player colour under the click point (a villager in front?)"
            if why:
                rejected.append(SheepCandidate(c.x, c.y, c.box, c.score, c.carcass_score, c.template, c.scale,
                                               Verdict.OVERLAP, why))
            else:
                clear.append(c)
        return clear, rejected

    def _colour_under(self, hsv: np.ndarray, x: int, y: int) -> float:
        """Share of player-colour pixels in a small disc around (x, y)."""
        r = max(2, round(self.config.click_clearance * hsv.shape[1] / REFERENCE_SIZE[0]))
        h, w = hsv.shape[:2]
        patch = hsv[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
        if patch.size == 0:
            return 1.0
        yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
        disc = (yy - (y - max(0, y - r))) ** 2 + (xx - (x - max(0, x - r))) ** 2 <= r * r
        own = cv2.inRange(patch, np.array(self.config.owner_hsv_low, np.uint8), np.array(self.config.owner_hsv_high, np.uint8)) > 0
        return float((own & disc).sum() / max(1, disc.sum()))


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
          Verdict.CARCASS: (0, 0, 255), Verdict.OVERLAP: (255, 0, 255), Verdict.OUTSIDE_WORLD: (128, 128, 128),
          Verdict.COLLAR_MISMATCH: (255, 128, 0)}


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
