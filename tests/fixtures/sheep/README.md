# Sheep detection dataset

Crops of real 1680x1050 Age of Empires II DE captures (French UI, player 1 blue): water
buffalo (the sheep equivalent) on a grass map, white sheep on a desert map, a goose on savanna, pigs on a winter Arabia.
`sheep_labels.json` gives, per crop, the source frame, the crop rectangle in that frame, and
every animal's box (source-frame pixels) with a label and a note.

| File | Content | Labels |
|---|---|---|
| `herd_open.png` | three buffalo grazing in the open (start of game) | 3 `sheep_living` |
| `herd_tc.png` | three buffalo herded under the Town Center, overlapping | 3 `sheep_living` |
| `carcass_gathered.png` | a slain buffalo being gathered, two standing behind villagers | 1 `sheep_carcass`, 1 `sheep_living_occluded` (two animals) |
| `crowd_base.png` | dense base: villager crowds and houses in the player colour | none (negatives) |
| `grass.png` | grass, bushes, dirt | none (background) |
| `desert_sheep.png` | six white sheep on sand, a villager | 6 `sheep_living` (3 templates cut from here, 3 held out) |
| `desert_sheep_2.png` | the same sheep in an earlier capture, other poses | 6 `sheep_living` |
| `desert_scout.png` | the player's scout cavalry on sand | none (negative) |
| `goose_savanna_a.png` | a goose on savanna (template `goose_a` cut from this frame) | 1 `sheep_living` |
| `goose_savanna_b.png` | the same goose 31 s earlier, elsewhere, other light | 1 `sheep_living` (held out) |
| `pig_winter_04.png` | four pigs on a winter Arabia (templates `pig_a`, `pig_b` from two) | 4 `sheep_living` |
| `pig_winter_10.png` | six pigs ~1.5 min later, two overlapping | 6 `sheep_living` (all held out) |
| `fog_berries_winter.png` | berry bushes under blue fog of war | none (negative) |
| `frost_winter.png` | a frost patch | none (negative) |

The source frames themselves are in `captures/` and `runs/` (not committed). Ten of the
templates in `aoe2_ai/perception/herd_templates/` were cut from the same frames, so tests that
check an animal is *found* either exclude that animal's own template (leave-one-out) or use
animals no template was cut from (desert sheep s1, s4, s5, every sheep of `desert_sheep_2`, `goose_savanna_b`, and pigs p04c, p04d and all of `pig_winter_10`).
Labels are drawn from the frames by eye, not from the detector's output.

To add a scene: crop it from a capture, add the file and its entry (source, crop, boxes)
to `sheep_labels.json`, and test it in `tests/test_sheep_detector.py`.
