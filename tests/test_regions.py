import numpy as np
import pytest

from aoe2_ai.perception.regions import Region, RegionSet

REQUIRED = {"wood", "food", "gold", "stone", "population", "game_time", "age", "selection_name", "selection_hp", "hud_anchor"}


def test_default_regions_cover_required_fields():
    regions = RegionSet.default()
    assert regions.reference_size == (1680, 1050)
    assert REQUIRED <= set(regions.regions)


def test_default_regions_inside_reference_frame():
    regions = RegionSet.default()
    w, h = regions.reference_size
    for r in regions:
        assert 0 <= r.x and 0 <= r.y and r.x + r.w <= w and r.y + r.h <= h, r


def test_same_size_is_identity():
    regions = RegionSet.default()
    assert regions.for_size(1680, 1050) is regions


def test_scaling_proportional():
    regions = RegionSet.default().for_size(3360, 2100)
    wood = regions["wood"]
    assert (wood.x, wood.y, wood.w, wood.h) == (84, 36, 104, 32)


def test_aspect_ratio_mismatch_warns():
    with pytest.warns(UserWarning):
        RegionSet.default().for_size(1920, 1080)


def test_crop_clips_to_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert Region("a", 190, 90, 50, 50).crop(frame).shape == (10, 10, 3)
    assert Region("b", 300, 300, 10, 10).crop(frame).size == 0


def test_json_round_trip(tmp_path):
    regions = RegionSet.default()
    path = regions.save(tmp_path / "r.json")
    assert RegionSet.load(path) == regions
