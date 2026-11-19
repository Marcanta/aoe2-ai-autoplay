# Villager assignment: idle villager -> sheep (MVP 5)

Sends idle villagers, one at a time, to gather food from a visible sheep, and checks on
screen that each one actually started. Deterministic, no LLM, no memory reading.

```
aoe2_ai/perception/object_detector.py        SheepDetector: own sheep in the world area (OpenCV only)
aoe2_ai/perception/herd_templates/           template PNGs, alpha = silhouette (living/, carcass/)
aoe2_ai/controllers/villager_assignment.py   decide() + VillagerAssignmentController + CLI
tests/fixtures/sheep/                        labelled crops of real captures (see its README)
tests/test_sheep_detector.py                 detection on real crops pasted into a real HUD frame
tests/test_villager_assignment_controller.py closed loop with scripted observations / detector / input
```

It reuses, unchanged: `LiveObserver` / `Observation`, the idle-villager perception
(`VillagerDetector`, `VillagerFingerprint`), the `.` "next idle villager" hotkey action of the
idle-villager controller, `ActionExecutor` (foreground guard, emergency stop, dedupe /
pending verification, dry run), `DebugRecorder`, and the `runtime.run_live` safety net
(F12 / Pause / Ctrl+C / mouse in a corner). Town Center logic, OCR and the existing
perception are not modified.

## Running it

1. Single-player skirmish, Dark Age, at least one idle villager, **player 1 blue** (the
   collar colour; see Limitations). The "Go to next idle villager" hotkey must be `.`.
2. From the activated venv:

```powershell
python -m aoe2_ai.controllers assign-sheep --dry-run -v                  # observe + detect, log "would right-click", send nothing
python -m aoe2_ai.controllers assign-sheep                               # Phase A: stops after 1 verified assignment
python -m aoe2_ai.controllers assign-sheep --max-assignments 0 --metrics runs/sheep --assignments-log runs/sheep/assignments.json   # Phase B
python -m aoe2_ai.perception.object_detector --image shot.png --save captures/sheep   # what the detector sees on a screenshot
```

Do not run it together with the idle-villager or Town Center controller: all of them change
the selection. Failures (and every "no target" / "target lost") are saved to
`captures/villager_assignment/` (frame + observation JSON). Settings are
`VillagerAssignmentConfig` fields, overridable with `--config file.json`.

## One cycle

| Step | What happens | Never |
|---|---|---|
| Observe | idle button (yellow / grey / count) + selection panel, one frame | act on an observation older than `max_observation_age` (3 s) |
| Evaluate (`decide`) | ASSIGN only if game detected, button **seen** yellow, not holding | treat UNKNOWN as 0, or as "some idle" |
| Select | press `.` via the executor; poll until a villager is read selected (name, this frame) | go on without a verified villager |
| Repeat check | the selected villager matches one commanded in the last `processed_ttl` (180 s)? -> SKIPPED, hold | command a villager twice |
| Target | `SheepDetector` on the frame where the villager was verified (the hotkey centred the camera on it) | use a sheep seen in an earlier frame |
| Choose | ACCEPTED sheep only; a sheep not already given to another villager first, then the nearest to the reference point | click a rejected / ambiguous detection |
| Re-check | fresh frame: game detected, same villager selected, the sheep detected again within `target_tolerance` (12 px) | click if any of these fails (TARGET_LOST) |
| Act | right-click through the executor (guard checks the foreground window right before sending); logged with villager fingerprint + target | send a second command while one is unverified |
| Verify | poll the selection panel + idle count for `command_verify_timeout` (6 s) | count the click itself as success |

Verification results:

* **SUCCESS**: the same villager (same HP) now reads the gatherer role, "Shepherd" /
  "Berger", in 2 frames, and the idle count did not go up. The role is a direct signal about
  *that* villager.
* **FAILURE**: during the whole window the same villager kept its idle role and a readable
  idle count never dropped. -> back-off.
* **UNKNOWN**: everything else. In particular an idle-count drop alone (another villager may
  have been tasked), a villager that already was a shepherd (its role cannot change), a lost
  or unreadable selection, the game leaving the screen, a stop. -> back-off, and the villager
  is remembered so it is **not** commanded again.

Movement toward the sheep is not used as evidence: the camera does not follow the villager
and the selected unit's position is not tracked, so it could not be observed reliably.

After `max_consecutive_failures` (3) failures/UNKNOWNs in a row the controller halts.

**Phase B (several villagers)** is the same cycle repeated, one villager per cycle, with a
fresh observation every time (`--max-assignments 0`). Sheep already given to a villager are
remembered by a 96 px grey thumbnail of their surroundings (`view_signature`, the same
signature the idle-villager fingerprint uses); a later candidate whose surroundings correlate
>= 0.85 counts as taken. When every visible sheep is taken (or cannot be told apart), the
controller does not click and holds for `no_target_hold` (30 s) or until the idle set changes;
`allow_shared_targets=true` lets it reuse a taken sheep instead. No resource balancing.

**Preference "near the Town Center".** The Town Center is not located on screen yet, so the
reference point is the centre of the world area, where the hotkey put the selected villager:
the closest sheep is the shortest walk. A `tc_locator(frame) -> (x, y)` can be passed to the
controller to prefer sheep near the Town Center instead.

## Sheep detection

All in screen pixels, OpenCV only (`SheepDetector.detect(frame)` -> `SheepScan`):

1. **World area.** Masked out: top bar (y < 50), menu buttons (top right), the corner
   ornament + idle portrait (top left), bottom panel (y >= 880), minimap and score panel
   (bottom right), and 12 px along every screen edge (the mouse there scrolls the camera).
   Rectangles are measured at 1680x1050 and scaled like the HUD regions.
2. **Candidates = the player's collar.** A herd animal the player owns wears a collar in the
   player colour. Compact blobs of that colour (HSV 100-135 hue, S >= 100, V >= 80 for blue),
   6-250 px at the default zoom, that touch an animal-coloured blob of animal size (0.35x to
   2x a template silhouette) are the candidates. Animal-coloured = very desaturated (S < 85:
   white wool, grey), or moderately desaturated *and* dark (S < 115, V < 170: buffalo), or at
   least 40 below the **local ground's** saturation (`ground_saturation`: median S of each
   48 px tile). Grass is saturated; sand is moderately desaturated (S ~105-125) but bright;
   savanna ground is S ~160, against which a goose's breast (S ~115) and grey wing (S ~130)
   stand out. Each fixed-threshold version failed on one map live: a plain S < 115 turned
   the whole desert into one "body", and fixed S/V cuts missed the goose on savanna. Enemy sheep (other collar colour) and
   neutral ones never become candidates. Villagers and buildings also carry the colour; the
   body-size test removes most of them cheaply.
3. **Template verification.** Masked `TM_CCOEFF_NORMED` (the silhouette is the mask, so the
   grass behind a template does not count) at half resolution, every template and its
   mirror, at the zooms 0.7-1.4 whose size fits the body blob. A probe at the two zooms
   closest to the body size runs first; under 0.45 the candidate is dropped. Only
   placements whose box contains the collar are considered.
4. **Full-resolution re-score.** The best half-resolution match is scored again at full
   resolution, +-3 px around it; that score is the one compared with the thresholds. At
   half resolution a sheep template at the smallest zoom is ~4x8 px, and female villagers
   (white apron, blue dress) scored 0.72-0.78 there; at full resolution 0.60-0.70, while
   real animals score 0.76-1.00.
5. **Collar share.** The player colour inside the matched silhouette must be 0.5x-1.6x the
   template's own collar share. Real animals: 0.88-1.17x; the scout's rider: 1.9-2.7x.
6. **Classification** (`Verdict`):

| Verdict | Rule | Clicked? |
|---|---|---|
| ACCEPTED | living score >= **0.72**, inside the world, no conflict | yes |
| LOW_CONFIDENCE | 0.55 <= score < 0.72 | no |
| CARCASS | a carcass template beats every living one | no |
| AMBIGUOUS | a carcass template within 0.05 of the living score | no |
| COLLAR_MISMATCH | player colour share outside 0.5x-1.6x of the template's | no |
| OUTSIDE_WORLD | box touches the HUD, minimap or a screen edge | no |
| OVERLAP | box overlaps another detected animal (IoU > 0.15) | no |

Below 0.55 a candidate is not reported. A scan on a missing / unusable frame, or with no
templates, is `UNKNOWN`: "could not look", never "no sheep".

### Templates and dataset

**One template set per species.** AoE2 has many herdables (sheep, goose, turkey, llama, pig,
cow, goat, water buffalo...), each map picks some, and they share only the player collar. The
detector recognises a species only once it has a template of it: without one the animal is
not a target (a safe miss, logged as "no sheep visible"). Captured so far: **water buffalo**
on grass, **white sheep** on a desert map, and a **goose** on savanna. Templates:

| Template | From | Pose |
|---|---|---|
| `living/water_buffalo_a` | `start_20261116_184105`, open field | grazing, facing right |
| `living/water_buffalo_b` | same | grazing, facing left |
| `living/water_buffalo_c` | same | grazing, facing right |
| `living/water_buffalo_d_walking` | `eco_20261116_184221`, herded at the TC | walking (legs slightly cut) |
| `living/sheep_front_a` | `sheep_20261118_230725`, desert | facing the camera |
| `living/sheep_side_c` | same | side view, head up |
| `living/sheep_quarter_d` | same | three-quarter view |
| `living/goose_a` | `goose_savanna_a` (live run 2026-11-19, saved "no target" frame) | walking, facing right |
| `carcass/water_buffalo_carcass_a` | `eco_20261116_184302`, being gathered | lying, head hidden by a villager |

Mirrored versions are generated at load time. **Adding a species**, when a run says "no sheep
visible" while one is on screen:

1. The frame is already saved: `captures/villager_assignment/*_no_target_*.png`.
2. Cut a template around the animal (a box ~20 px larger than it):
   `python -m aoe2_ai.perception.object_detector --extract --image <frame> --box X,Y,W,H --name goose_a`
   (`--kind carcass` for a carcass). The silhouette is cut automatically: pixels clearly less
   saturated than the crop border (the ground), or less saturated and clearly darker, thin
   lines opened away, and the component holding the collar.
3. Check it: `python -m aoe2_ai.perception.object_detector --image <another frame> --save captures/sheep`
   (green box on the animal), ideally on a frame the template was not cut from.
4. Add a crop to `tests/fixtures/sheep/` with its label and a test, like the goose ones.
5. Run the false-positive check over frames with no herd (the `runs/validation/` frames were
   used here): a new small template can match villagers.

The labelled dataset (`tests/fixtures/sheep/`, provenance and boxes in `sheep_labels.json`)
holds crops of real frames: the open-field herd, the herd overlapping under the Town Center,
a carcass being gathered next to two standing buffalo, a dense base with villager crowds and
houses (negatives), plain grass, six white sheep on sand in two captures (three of them never
used as templates), the player's scout on sand (negative), and a goose on savanna in two
frames (one held out). Tests paste them into a real HUD frame.

### Offline results (1680x1050 captures)

| Check | Result |
|---|---|
| Open-field herd, 8 frames (3 buffalo each) | 3/3 found in every frame, scores 0.82-0.96 |
| Leave-one-out (each buffalo without its own template) | 0.95, 0.76, 0.95 (threshold 0.72) |
| White sheep on sand, 2 captures (6 sheep each) | 6/6 in both, 0.79-1.00; the 3 held-out sheep 0.85-0.98 |
| Own scout and villagers on sand | not accepted (best 0.71 before the full-resolution check) |
| Goose on savanna, 11 frames of the 2026-11-19 run (3 positions) | found in all, 0.84-1.00; the held-out frame 0.98 |
| Zoom 0.8x / 1.25x, camera shifts, 1920x1200 | found at the moved / scaled positions (tests) |
| Herd overlapping at the TC (3 buffalo) | 1 accepted, on an animal; the others rejected (low confidence) |
| Carcass being gathered | not accepted |
| 190 frames of a dense base with no herd + all HUD-only fixtures + buffalo pasted on the HUD / minimap | **0** accepted |
| Red-collared copy of a buffalo | 0 candidates |
| Time per scan | ~0.2-0.4 s early game, ~1.7 s median in a dense base (80-candidate cap reached: logged) |

The leave-one-out margin of buffalo b (0.76 vs 0.72) is the tightest: it is the only one
facing left in its own picture, so the others only match it mirrored.

## Limitations

* **Live validation**: 3 assignments on the desert map so far, all SUCCESS; see
  [villager_assignment_live_report.md](villager_assignment_live_report.md).
* **Only the captured species are recognised**: water buffalo, white sheep, goose, each in one
  or a few poses at one zoom. Turkeys, llamas, pigs, cows, goats... need their own templates
  (see "Adding a species"). Snow maps are not tested: white sheep on snow may not stand out
  from the ground.
* **Player colour is fixed to blue** (`owner_hsv_low/high`). Another colour needs another
  HSV range in the config. A green collar on grass would not work with this approach.
* **Herded, overlapping animals** under the Town Center are mostly rejected (safe, but the
  controller may then find "no target" where a human sees sheep).
* **Carcasses**: one carcass template, from one partly hidden carcass. A carcass with no
  visible collar is never a candidate anyway; one with a visible collar is caught only if
  the carcass template scores close to it.
* **The gatherer role name** ("Shepherd" / "Berger") as success signal is an assumption about
  the game to confirm live, also for water buffalo. If the game keeps "Villager" while the
  villager walks to the sheep, live runs will show UNKNOWN instead of SUCCESS (never a false
  SUCCESS); raise `command_verify_timeout` then.
* **Duplicate-target memory** compares surroundings; a large camera zoom change between two
  assignments makes taken sheep look new (then a sheep may get a second villager, which the
  game allows).
* **Sheep outside the camera view** are not searched for: the controller does not move the
  camera beyond what the idle hotkey does.
* **Scan time** grows with the number of templates (8 living views now): ~1.7 s in a dense
  base, where the 80-candidate cap can also be reached (some animals may then be missed;
  the scan reason says so). Each assignment scans twice (choose, re-check).
