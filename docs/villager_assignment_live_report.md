# Villager assignment: live gameplay report

**Status: Phase A passed once live (1 villager, 1 SUCCESS); Phase B and the edge cases not run yet.** The rest of MVP 5 is offline (unit tests and real
screenshots). Nothing below is a result; fill it in from real runs. Do not read the unit
tests as proof that the controller works in the game.

## Log so far

| Date | Run | Outcome |
|---|---|---|
| 2026-11-18 | `assign-sheep --dry-run -v --max-cycles 5`, desert map, white sheep, 3 idle | Safety behaved: waited while PowerShell / Task Switching was in front, settled 1 observation, read 3 idle villagers, sent nothing. **Detection found 0 sheep with 6 in view.** Cause: sand (S ~115) counted as animal body, and only buffalo templates existed. Fixed: `animal_pixels` (bright sand excluded), 3 white-sheep templates, full-resolution re-score, collar-share check. Offline on that capture now 6/6. |
| 2026-11-19 | same dry run, after the fix | 6/6 sheep seen every cycle ("6 free of 6"), the same target chosen each time (748-749, 428), scores 0.82-0.97 (held-out sheep). Nothing sent. |
| 2026-11-19 | Phase A live: `assign-sheep -v --metrics runs/sheep/phase_a` | `.` -> villager verified ("Villa eois" read, conf 0.58, fuzzy-matched) -> sheep at (749, 428), score 0.96 -> right-click -> **SUCCESS** 2.2 s later: panel reads "Berger", idle 3 -> 2. Checked on the after frame: villager at the sheep, cursor tooltip is the sheep's. Answers the open question: the role turns into "Berger" as soon as the order is given (within the first verification frames). |

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
| 1 | 1 | 1 | 0 | 0 | 0 | 0 | 452 / 1722 (n=6; median 78) | 2026-11-19, desert map, white sheep. P95 is the first observation (OCR start-up). Click to verified SUCCESS: 2.5 s. |
| 2 | | | | | | | | |
| 3 | | | | | | | | |
| 4 | | | | | | | | |
| 5 | | | | | | | | |
| 6 | | | | | | | | |
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
