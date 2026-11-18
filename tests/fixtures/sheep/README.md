# Sheep detection dataset

Crops of real 1680x1050 Age of Empires II DE captures (French UI, player 1 blue). On the map
of every capture so far the herd animals are **water buffalo**, the sheep equivalent.
`sheep_labels.json` gives, per crop, the source frame, the crop rectangle in that frame, and
every animal's box (source-frame pixels) with a label and a note.

| File | Content | Labels |
|---|---|---|
| `herd_open.png` | three buffalo grazing in the open (start of game) | 3 `sheep_living` |
| `herd_tc.png` | three buffalo herded under the Town Center, overlapping | 3 `sheep_living` |
| `carcass_gathered.png` | a slain buffalo being gathered, two standing behind villagers | 1 `sheep_carcass`, 1 `sheep_living_occluded` (two animals) |
| `crowd_base.png` | dense base: villager crowds and houses in the player colour | none (negatives) |
| `grass.png` | grass, bushes, dirt | none (background) |

The source frames themselves are in `captures/` and `runs/` (not committed). Four of the
templates in `aoe2_ai/perception/herd_templates/` were cut from the same frames, so tests that
check an animal is *found* exclude that animal's own template (leave-one-out).

To add a scene: crop it from a capture, add the file and its entry (source, crop, boxes)
to `sheep_labels.json`, and test it in `tests/test_sheep_detector.py`.
