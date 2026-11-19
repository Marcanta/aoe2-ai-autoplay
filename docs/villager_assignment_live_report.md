# Villager assignment: live gameplay report

**Status: Phase A and Phase B passed live (3 villagers, 3 SUCCESS, 0 wrong or duplicate commands); edge cases 4, 5 and 7 not run yet, 6 only partly.** The rest of MVP 5 is offline (unit tests and real
screenshots). Nothing below is a result; fill it in from real runs. Do not read the unit
tests as proof that the controller works in the game.

## Log so far

| Date | Run | Outcome |
|---|---|---|
| 2026-11-18 | `assign-sheep --dry-run -v --max-cycles 5`, desert map, white sheep, 3 idle | Safety behaved: waited while PowerShell / Task Switching was in front, settled 1 observation, read 3 idle villagers, sent nothing. **Detection found 0 sheep with 6 in view.** Cause: sand (S ~115) counted as animal body, and only buffalo templates existed. Fixed: `animal_pixels` (bright sand excluded), 3 white-sheep templates, full-resolution re-score, collar-share check. Offline on that capture now 6/6. |
| 2026-11-19 | same dry run, after the fix | 6/6 sheep seen every cycle ("6 free of 6"), the same target chosen each time (748-749, 428), scores 0.82-0.97 (held-out sheep). Nothing sent. |
| 2026-11-19 | Phase A live: `assign-sheep -v --metrics runs/sheep/phase_a` | `.` -> villager verified ("Villa eois" read, conf 0.58, fuzzy-matched) -> sheep at (749, 428), score 0.96 -> right-click -> **SUCCESS** 2.2 s later: panel reads "Berger", idle 3 -> 2. Checked on the after frame: villager at the sheep, cursor tooltip is the sheep's. Answers the open question: the role turns into "Berger" as soon as the order is given (within the first verification frames). |
| 2026-11-19 | Phase B live: `assign-sheep -v --max-assignments 0 --max-cycles 20 --metrics runs/sheep/phase_b` | 2 idle -> 2 cycles, 2 **SUCCESS** (idle 2 -> 1 -> 0), each 1.6 s after the click. **Different sheep**: mapped onto one frame through the Town Center's shift (template match 0.999), the two targets are 54 px apart; the first is already slain and being gathered, the second a separate living sheep. Then 16 cycles "no idle villagers", no input. Cycle 10: PowerShell in front -> waited, nothing sent, then settled 1 observation on return (partial check of scenario 6). |
| 2026-11-19 | Phase A live, savanna map, one goose in view | 2 cycles **NO_TARGET**, no right-click, then holds (30 s) as designed; also waited correctly while Claude / PowerShell / Task Switching were in front; stopped by Ctrl+C. Cause: no goose template (detection is per species), and the fixed body thresholds missed a goose on savanna ground (S ~160). Fixed: `goose_a` template from the saved no-target frame, body test against the local ground's saturation. Offline now: goose found in all 11 frames of that run (0.84-1.00), 0 false accepts on 190 base frames. To re-run live. |

## Setup to record

| | |
|---|---|
| Date | |
| Game build | |
| Resolution / UI scale | |
| Map / herd animal | |
| Player colour | (must be blue, player 1) |
| Commit | |

## Scenarios

Run each with `--metrics runs/sheep/<scenario> --assignments-log runs/sheep/<scenario>/assignments.json -v`.
Start with `--dry-run` to check the detector's choice in the log before any live click.

| # | Scenario | Command | Expected |
|---|---|---|---|
| 1 | 1 idle villager, 1 sheep visible | `assign-sheep` | 1 SUCCESS, then DONE |
| 2 | 1 idle villager, several sheep | `assign-sheep` | nearest free sheep, 1 SUCCESS |
| 3 | several idle villagers, several sheep | `assign-sheep --max-assignments 0` | one per cycle, different sheep, no duplicate |
| 4 | no sheep visible | `assign-sheep --max-assignments 0` | NO_TARGET, then holds; no right-click |
| 5 | sheep outside the camera view | `assign-sheep --max-assignments 0` | NO_TARGET (no camera search) |
| 6 | game paused / alt-tabbed | `assign-sheep --max-assignments 0` | WAITED / BLOCKED; no input to another window |
| 7 | F12 during a cycle | any | STOPPED, no further input |

## Results

| # | Villagers selected | SUCCESS | FAILURE | UNKNOWN | Incorrect commands | Duplicate assignments | Obs. latency mean / P95 (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | Not run as such (Phase A had 6 sheep in view: see row 2). |
| 2 | 1 | 1 | 0 | 0 | 0 | 0 | 452 / 1722 (n=6; median 78) | 2026-11-19 (Phase A), desert map, 6 white sheep in view, nearest chosen. P95 is the first observation (OCR start-up). Click to verified SUCCESS: 2.5 s. |
| 3 | 2 | 2 | 0 | 0 | 0 | 0 | 129 / 921 (n=28; median 52) | 2026-11-19. 2 villagers, 2 different sheep (checked on the frames). Idle 0 afterwards: 16 cycles without input. |
| 4 | 2 | 0 | 0 | 0 | 0 | 0 | 90 / 56 (n=62) | 2026-11-19, savanna map with a goose but no goose template yet: NO_TARGET twice, no right-click, held 30 s each time (behaves as "no sheep visible"). |
| 5 | | | | | | | | |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | | Partly, during run 3: alt-tab to PowerShell -> "game not detected", no input. Pause not tested. |
| 7 | | | | | | | | |

Where to read them: the end-of-run `Summary:` line (`selected`, `success`, `failure`,
`unknown`, `duplicate_commands`, `observation_latency`), `assignments.json` (every
right-click with villager, target and result), and `summary.json` of `--metrics` (latency
mean / median / P95). "Incorrect commands" = right-clicks that did not land on one of the
player's sheep: check the `cycle_NNNN_*_before.png` / `_after.png` frames by eye.

## Open questions to answer live

* ~~Does a villager sent to a sheep read "Berger" right away?~~ Yes for white sheep
  (2026-11-19, Phase A). Still open for water buffalo.
* Scan time in the real game next to OCR: is the re-check still within the 12 px tolerance
  when a sheep wanders?
* Detection scores on white sheep once a sheep template exists.
