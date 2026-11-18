# Villager assignment: live gameplay report

**Status: NOT RUN YET.** Everything about MVP 5 so far is offline (unit tests and real
screenshots). Nothing below is a result; fill it in from real runs. Do not read the unit
tests as proof that the controller works in the game.

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
| 1 | | | | | | | | |
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

* Does a villager sent to a sheep (or water buffalo) read "Shepherd" / "Berger" right away,
  or only once it starts gathering? (Decides SUCCESS vs UNKNOWN.)
* Scan time in the real game next to OCR: is the re-check still within the 12 px tolerance
  when a sheep wanders?
* Detection scores on white sheep once a sheep template exists.
