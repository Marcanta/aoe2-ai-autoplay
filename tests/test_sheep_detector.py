"""Sheep detection on real game crops (tests/fixtures/sheep), pasted into a real HUD frame.

The crops come from 1680x1050 captures; on that map the herd animals are water buffalo
(the sheep equivalent, player-blue collar). Where a test checks that an animal is *found*,
the detector is built without the template cut from that very animal (leave-one-out), so
the test measures generalisation, not a template matching itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoe2_ai.perception.game_state import Status
from aoe2_ai.perception.object_detector import (
    HUD_EXCLUSIONS,
    HerdTemplate,
    SheepCandidate,
    SheepDetector,
    SheepScan,
    Verdict,
    box_in_world,
    extract_template,
    load_templates,
    world_mask,
)

from conftest import FRAMES

SHEEP = Path(__file__).parent / "fixtures" / "sheep"
LABELS = json.loads((SHEEP / "sheep_labels.json").read_text(encoding="utf-8"))["items"]
HUD_FRAME = FRAMES / "start_20261116_184105.png"  # real HUD, play area blacked out
TEMPLATES = load_templates()
OWN_TEMPLATE = {"a": "water_buffalo_a", "b": "water_buffalo_b", "c": "water_buffalo_c", "d": "water_buffalo_d_walking"}


def crop(name: str) -> np.ndarray:
    img = cv2.imread(str(SHEEP / LABELS[name]["file"]), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def hud_frame() -> np.ndarray:
    img = cv2.imread(str(HUD_FRAME), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == (1050, 1680)
    return img


def grass_frame() -> np.ndarray:
    """Real HUD with the world area tiled with real grass."""
    frame = hud_frame()
    g = crop("grass")
    h, w = frame.shape[:2]
    tiled = np.tile(g, (h // g.shape[0] + 1, w // g.shape[1] + 1, 1))[:h, :w]
    world = world_mask(h, w, edge=0)
    frame[world] = tiled[world]
    return frame


def paste(frame: np.ndarray, img: np.ndarray, x: int, y: int) -> np.ndarray:
    out = frame.copy()
    h, w = img.shape[:2]
    out[y:y + h, x:x + w] = img
    return out


def detector(exclude: str | None = None, **templates_filter) -> SheepDetector:
    skip = OWN_TEMPLATE.get(exclude) if exclude else None
    return SheepDetector(templates=[t for t in TEMPLATES if t.name != skip])


def boxes(name: str, offset=(0, 0), scale=1.0, kind="sheep_living") -> dict[str, tuple[int, int, int, int]]:
    """Label boxes of ``name`` in a frame where its crop was pasted at its source position + offset, scaled."""
    cx, cy = LABELS[name]["crop"][:2]
    out = {}
    for a in LABELS[name]["animals"]:
        if a["label"] != kind:
            continue
        x, y, w, h = a["box"]
        out[a["id"]] = (round(cx + (x - cx) * scale + offset[0]), round(cy + (y - cy) * scale + offset[1]),
                        round(w * scale), round(h * scale))
    return out


def inside(s: SheepCandidate, box, margin=6) -> bool:
    x, y, w, h = box
    return x - margin <= s.x <= x + w + margin and y - margin <= s.y <= y + h + margin


def place(name: str, offset=(0, 0), scale=1.0, base: np.ndarray | None = None) -> np.ndarray:
    img = crop(name)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x, y = LABELS[name]["crop"][:2]
    return paste(grass_frame() if base is None else base, img, x + offset[0], y + offset[1])


def single(animal: str, pad: int = 14) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop of one open-field animal (from the source capture region) and its box in that crop."""
    item = LABELS["herd_open"]
    a = next(a for a in item["animals"] if a["id"] == animal)
    x, y, w, h = a["box"]
    cx, cy = item["crop"][:2]
    img = crop("herd_open")[y - cy - pad:y - cy + h + pad, x - cx - pad:x - cx + w + pad]
    return img.copy(), (pad, pad, w, h)


# --- scan status ---------------------------------------------------------------
def test_unusable_frame_is_unknown_not_empty():
    scan = SheepDetector().detect(None)
    assert scan.status is Status.UNKNOWN and not scan.is_valid and scan.sheep == ()


def test_no_templates_is_unknown():
    scan = SheepDetector(templates=[]).detect(grass_frame())
    assert scan.status is Status.UNKNOWN and "templates" in scan.reason


def test_templates_load_with_silhouettes():
    kinds = {t.kind for t in TEMPLATES}
    assert kinds == {"living", "carcass"}
    for t in TEMPLATES:
        assert t.image.shape[:2] == t.mask.shape and 0.3 < (t.mask > 0).mean() < 0.95


# --- one / several / none -------------------------------------------------------------
@pytest.mark.parametrize("animal", [
    "a",
    pytest.param("b", marks=pytest.mark.xfail(strict=True, reason=(
        "known miss: buffalo b is the only one facing left; without its own template it scores 0.74, "
        "under the 0.75 bar, which sits just above a villager false accept (0.73)"))),
    "c",
])
def test_one_visible_sheep_found_without_its_own_template(animal):
    img, (bx, by, bw, bh) = single(animal)
    at = (700, 400)
    scan = detector(exclude=animal).detect(paste(grass_frame(), img, *at))
    assert scan.is_valid
    assert len(scan.sheep) == 1, scan.to_dict()
    s = scan.sheep[0]
    assert inside(s, (at[0] + bx, at[1] + by, bw, bh)) and s.score >= 0.72


def test_multiple_visible_sheep():
    scan = SheepDetector().detect(place("herd_open"))
    truth = boxes("herd_open")
    assert len(scan.sheep) == 3, scan.to_dict()
    hits = {k for s in scan.sheep for k, b in truth.items() if inside(s, b)}
    assert hits == set(truth)


@pytest.mark.parametrize("name", ["crowd_base", "grass"])
def test_no_sheep_visible_is_valid_and_empty(name):
    scan = SheepDetector().detect(place(name))
    assert scan.is_valid and scan.sheep == (), scan.to_dict()


def test_real_frames_without_herd_animals_give_no_sheep():
    # Full HUD fixtures (play area blacked out): the HUD is full of colour, nothing may come out of it.
    for path in sorted(FRAMES.glob("*.png")):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert SheepDetector().detect(frame).sheep == (), path.name


# --- obscured / ambiguous / carcass ------------------------------------------------------
@pytest.mark.parametrize("cover", [0.25, 0.4])
def test_partially_obscured_sheep_never_misplaced(cover):
    img, (bx, by, bw, bh) = single("c")
    grass = crop("grass")
    covered = img.copy()
    cut = bx + round(bw * (1 - cover))  # hide the rear of the animal (the collar is at the front)
    covered[:, :cut] = grass[:covered.shape[0], :cut]
    at = (700, 400)
    scan = detector(exclude="c").detect(paste(grass_frame(), covered, *at))
    for s in scan.sheep:  # found or not, an accepted detection must be on the animal
        assert inside(s, (at[0] + bx, at[1] + by, bw, bh))


def test_collar_hidden_gives_no_target():
    img, (bx, by, bw, bh) = single("a")
    grass = crop("grass")
    hidden = img.copy()
    hidden[:, bx + bw // 2:] = grass[:hidden.shape[0], bx + bw // 2:hidden.shape[1]]  # front half (collar) covered
    scan = SheepDetector().detect(paste(grass_frame(), hidden, 700, 400))
    assert scan.sheep == ()


def test_overlapping_herd_at_town_center_no_misplaced_click():
    # Real frame: three buffalo herded under the Town Center, overlapping each other.
    scan = SheepDetector().detect(place("herd_tc"))
    truth = boxes("herd_tc")
    for s in scan.sheep:
        assert any(inside(s, b) for b in truth.values())
    assert all(r.verdict is not Verdict.ACCEPTED for r in scan.rejected)


def test_overlapping_own_animals_stay_clickable():
    # A herd under the Town Center overlaps: a click on either one is a click on a player's animal.
    det = SheepDetector()
    a = SheepCandidate(100, 100, (80, 80, 50, 40), 0.9, None, "t", 1.0, Verdict.ACCEPTED)
    b = SheepCandidate(120, 110, (95, 90, 50, 40), 0.85, None, "t", 1.0, Verdict.ACCEPTED)
    far = SheepCandidate(400, 400, (380, 380, 50, 40), 0.8, None, "t", 1.0, Verdict.ACCEPTED)
    sheep, rejected = det._classify([a, b, far])
    assert sheep == [a, b, far] and rejected == []


def test_overlap_refusal_can_be_turned_back_on():
    from aoe2_ai.perception.object_detector import SheepDetectorConfig

    det = SheepDetector(SheepDetectorConfig(allow_herd_overlap=False))
    a = SheepCandidate(100, 100, (80, 80, 50, 40), 0.9, None, "t", 1.0, Verdict.ACCEPTED)
    b = SheepCandidate(120, 110, (95, 90, 50, 40), 0.85, None, "t", 1.0, Verdict.ACCEPTED)
    sheep, rejected = det._classify([a, b])
    assert sheep == [] and {r.verdict for r in rejected} == {Verdict.OVERLAP}


def test_animal_overlapping_a_rejected_look_alike_is_not_clicked():
    det = SheepDetector()
    a = SheepCandidate(100, 100, (80, 80, 50, 40), 0.9, None, "t", 1.0, Verdict.ACCEPTED)
    rider = SheepCandidate(115, 105, (92, 88, 50, 40), 0.8, None, "t", 1.0, Verdict.COLLAR_MISMATCH)
    sheep, rejected = det._classify([a, rider])
    assert sheep == [] and any(r.verdict is Verdict.OVERLAP and "look-alike" in r.reason for r in rejected)


def test_player_colour_under_the_click_point_is_not_clicked():
    # Something blue (a villager) right where the click would land: no click.
    img, _ = single("c")
    frame = paste(grass_frame(), img, 700, 400)
    clear = SheepDetector().detect(frame)
    assert len(clear.sheep) == 1
    s = clear.sheep[0]
    cv2.circle(frame, (s.x, s.y), 4, (200, 40, 30), -1)  # BGR player blue
    scan = SheepDetector().detect(frame)
    assert scan.sheep == ()  # refused: player colour under the click, or far too much of it on the animal
    assert any(r.verdict in (Verdict.OVERLAP, Verdict.COLLAR_MISMATCH) for r in scan.rejected)


def test_same_animal_matched_twice_is_one():
    det = SheepDetector()
    a = SheepCandidate(100, 100, (80, 80, 50, 40), 0.9, None, "t", 1.0, Verdict.ACCEPTED)
    a2 = SheepCandidate(101, 101, (81, 81, 50, 40), 0.8, None, "t", 1.0, Verdict.ACCEPTED)
    sheep, rejected = det._classify([a, a2])
    assert sheep == [a] and rejected == []


def test_carcass_being_gathered_is_not_a_target():
    scan = SheepDetector().detect(place("carcass_gathered"))
    carcass = boxes("carcass_gathered", kind="sheep_carcass")["h"]
    assert not any(inside(s, carcass, margin=0) for s in scan.sheep), scan.to_dict()


def _as(t: HerdTemplate, kind: str) -> HerdTemplate:
    return HerdTemplate(t.name, kind, t.image, t.mask)


def test_carcass_template_scoring_higher_is_rejected_as_carcass():
    by_name = {t.name: t for t in TEMPLATES}
    # The animal's own picture as a "carcass", another buffalo as "living": the carcass wins.
    det = SheepDetector(templates=[by_name["water_buffalo_b"], _as(by_name["water_buffalo_a"], "carcass")])
    img, _ = single("a")
    scan = det.detect(paste(grass_frame(), img, 700, 400))
    assert scan.sheep == ()
    assert [r.verdict for r in scan.rejected] == [Verdict.CARCASS]


def test_living_and_carcass_scoring_alike_is_ambiguous():
    a = {t.name: t for t in TEMPLATES}["water_buffalo_a"]
    det = SheepDetector(templates=[a, _as(a, "carcass")])
    img, _ = single("a")
    scan = det.detect(paste(grass_frame(), img, 700, 400))
    assert scan.sheep == ()
    assert [r.verdict for r in scan.rejected] == [Verdict.AMBIGUOUS]


def test_enemy_coloured_sheep_is_never_a_target():
    img, _ = single("a")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array((100, 100, 80), np.uint8), np.array((135, 255, 255), np.uint8)) > 0
    assert blue.sum() > 10
    red = img.copy()
    red[blue] = (30, 30, 200)  # same animal, red (enemy) collar
    scan = SheepDetector().detect(paste(grass_frame(), red, 700, 400))
    assert scan.sheep == () and scan.candidates == 0


# --- interface areas -------------------------------------------------------------------
@pytest.mark.parametrize("where", ["bottom_panel", "minimap_and_scores", "top_bar"])
def test_sheep_drawn_on_the_hud_is_ignored(where):
    img, _ = single("b")
    x, y, w, h = HUD_EXCLUSIONS[where]
    at = (x + min(40, max(0, w - img.shape[1])), y + max(0, min(h - img.shape[0], 30)))
    scan = SheepDetector().detect(paste(hud_frame(), img, *at))
    assert scan.sheep == (), scan.to_dict()


def test_sheep_straddling_the_bottom_panel_is_not_clicked():
    img, (bx, by, bw, bh) = single("b", pad=4)
    top = HUD_EXCLUSIONS["bottom_panel"][1]
    frame = paste(grass_frame(), img, 700, top - by - bh // 2)  # lower half under the panel edge
    scan = SheepDetector().detect(frame)
    assert all(s.y < top for s in scan.sheep)
    world = world_mask(1050, 1680)
    assert all(box_in_world(world, s.box) for s in scan.sheep)


def test_world_mask_excludes_hud_and_edges_at_other_resolutions():
    for w, h in [(1680, 1050), (1920, 1200), (1280, 800)]:
        m = world_mask(h, w)
        assert m[h // 2, w // 2]
        assert not m[h - 5, w // 2] and not m[5, w // 2]  # bottom panel, top bar
        assert not m[h // 2, 1] and not m[h // 2, w - 2]  # edges (camera scroll)
        assert not m[round(h * 0.95), round(w * 0.9)]  # minimap


# --- zoom and camera position ----------------------------------------------------------
@pytest.mark.parametrize("scale", [0.8, 1.25])
def test_other_zoom_levels(scale):
    frame = place("herd_open", offset=(-200, -150), scale=scale)
    scan = SheepDetector().detect(frame)
    truth = boxes("herd_open", offset=(-200, -150), scale=scale)
    hits = {k for s in scan.sheep for k, b in truth.items() if inside(s, b)}
    assert len(hits) >= 2, scan.to_dict()  # zoom costs at most one animal...
    assert all(any(inside(s, b) for b in truth.values()) for s in scan.sheep)  # ...and never a wrong click


@pytest.mark.parametrize("offset", [(-700, -150), (-300, 100)])
def test_camera_position_moves_detections_with_it(offset):
    base = SheepDetector().detect(place("herd_open"))
    moved = SheepDetector().detect(place("herd_open", offset=offset))
    assert len(moved.sheep) == len(base.sheep) == 3
    a = sorted((s.x, s.y) for s in base.sheep)
    b = sorted((s.x - offset[0], s.y - offset[1]) for s in moved.sheep)
    assert all(abs(p[0] - q[0]) <= 2 and abs(p[1] - q[1]) <= 2 for p, q in zip(a, b))


def test_higher_resolution_frame():
    frame = cv2.resize(place("herd_open"), (1920, 1200), interpolation=cv2.INTER_LINEAR)
    scan = SheepDetector().detect(frame)
    assert len(scan.sheep) == 3


# --- helpers -----------------------------------------------------------------------------
def test_near_finds_the_closest_accepted_sheep_within_tolerance():
    s1 = SheepCandidate(100, 100, (80, 80, 50, 40), 0.9, None, "t", 1.0, Verdict.ACCEPTED)
    s2 = SheepCandidate(300, 100, (280, 80, 50, 40), 0.9, None, "t", 1.0, Verdict.ACCEPTED)
    scan = SheepScan(0.0, Status.VALID, "", (s1, s2))
    assert scan.near(105, 98, 12) is s1
    assert scan.near(200, 100, 12) is None


def test_extract_template_finds_the_animal_silhouette():
    img, (bx, by, bw, bh) = single("a")
    result = extract_template(img)
    assert result is not None
    t, m = result
    assert abs(t.shape[1] - bw) <= 8 and abs(t.shape[0] - bh) <= 8


# --- white sheep on a desert map (second live capture) ------------------------------------
HELD_OUT = ("s1", "s4", "s5")  # sheep of desert_sheep that no template was cut from


@pytest.mark.parametrize("name", ["desert_sheep", "desert_sheep_2"])
def test_white_sheep_on_sand_all_found(name):
    scan = SheepDetector().detect(place(name))
    truth = boxes(name)
    hits = {k for s in scan.sheep for k, b in truth.items() if inside(s, b, margin=3)}
    assert hits == set(truth), scan.to_dict()
    assert len(scan.sheep) == len(truth)


def test_held_out_white_sheep_are_found():
    scan = SheepDetector().detect(place("desert_sheep"))
    truth = boxes("desert_sheep")
    for k in HELD_OUT:
        assert any(inside(s, truth[k], margin=3) for s in scan.sheep), k


def test_sand_is_not_an_animal_body():
    # Sand is moderately desaturated (S ~115) but it is the ground: judged against itself, it
    # must not read as an animal (a fixed S < 115 rule once turned the whole desert into one body).
    from aoe2_ai.perception.object_detector import animal_pixels, ground_hsv

    hsv = cv2.cvtColor(crop("desert_scout"), cv2.COLOR_BGR2HSV)
    gs, gv = ground_hsv(hsv)
    assert animal_pixels(hsv, gs, gv)[:30, :30].mean() < 0.1


def test_own_scout_on_sand_is_not_a_sheep():
    scan = SheepDetector().detect(place("desert_scout"))
    assert scan.sheep == (), scan.to_dict()


def test_too_much_player_colour_is_a_collar_mismatch():
    # The scout's rider (blue tunic) under a sheep template's silhouette: far more player colour
    # than a collar, whatever the shape score.
    det = SheepDetector()
    frame = place("desert_scout")
    sheep_front = next(p for p in det._prepare(1680, 1050) if p.template.name == "sheep_quarter_d" and p.scale == 1.0)
    w, h = sheep_front.full_size
    share = det._collar_share(frame, (407 - w // 2, 742 - h // 2, w, h), sheep_front)
    assert share is not None and share > det.config.collar_share_band[1]


# --- goose (savanna map): one template per species ------------------------------------------
@pytest.mark.parametrize("name", ["goose_savanna_a", "goose_savanna_b"])
def test_goose_on_savanna_is_found(name):
    # goose_savanna_b: another frame, another place on the map; no template was cut from it.
    scan = SheepDetector().detect(place(name))
    truth = boxes(name)
    assert len(scan.sheep) == 1, scan.to_dict()
    assert inside(scan.sheep[0], next(iter(truth.values())), margin=3)
    assert scan.sheep[0].template.startswith("goose")


@pytest.mark.parametrize("name", ["goose_savanna_b", "pig_winter_10", "grass"])
def test_ground_is_not_an_animal_body(name):
    # Savanna (S ~160), frosty winter ground (S ~75) and grass: the ground never reads as animal.
    from aoe2_ai.perception.object_detector import animal_pixels, ground_hsv

    hsv = cv2.cvtColor(crop(name), cv2.COLOR_BGR2HSV)
    gs, gv = ground_hsv(hsv)
    assert animal_pixels(hsv, gs, gv)[:30, :30].mean() < 0.1


def test_no_species_template_means_no_target():
    # Without the goose template, a goose is not a target (safe miss, not a guess).
    det = SheepDetector(templates=[t for t in TEMPLATES if not t.name.startswith("goose")])
    assert det.detect(place("goose_savanna_b")).sheep == ()


# --- pigs on a winter Arabia: species, fog of war, frost ---------------------------------------
def test_pigs_on_winter_arabia():
    scan = SheepDetector().detect(place("pig_winter_04"))
    truth = boxes("pig_winter_04")
    found = {k for s in scan.sheep for k, b in truth.items() if inside(s, b, margin=3)}
    assert {"p04a", "p04b", "p04c"} <= found, scan.to_dict()  # p04c: held out
    assert all(any(inside(s, b, margin=3) for b in truth.values()) for s in scan.sheep)


def test_held_out_pigs_later_in_the_game():
    # No template was cut from this frame; two of the six pigs overlap.
    scan = SheepDetector().detect(place("pig_winter_10"))
    truth = boxes("pig_winter_10")
    found = {k for s in scan.sheep for k, b in truth.items() if inside(s, b, margin=3)}
    assert len(found) >= 4, scan.to_dict()
    assert all(any(inside(s, b, margin=3) for b in truth.values()) for s in scan.sheep)  # never a wrong click


def test_click_points_are_on_the_body_not_the_collar():
    scan = SheepDetector().detect(place("pig_winter_10"))
    assert scan.sheep
    for s in scan.sheep:
        assert SheepDetector()._colour_under(cv2.cvtColor(place("pig_winter_10"), cv2.COLOR_BGR2HSV), s.x, s.y) <= 0.2


@pytest.mark.parametrize("name", ["fog_berries_winter", "frost_winter"])
def test_blue_fog_and_frost_are_not_collars(name):
    # The winter fog of war and frost read as player blue; a collar must be blue on non-blue.
    scan = SheepDetector().detect(place(name))
    assert scan.sheep == (), scan.to_dict()
