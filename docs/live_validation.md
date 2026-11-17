# Live validation: HybridOcr vs Tesseract-only in a real skirmish

Validation of the controllers' OCR default (PR #7) in a real AoE2 DE game,
2026-11-17. No controller decision logic was changed for this; only the
`--metrics` recorder gained OpenCV / fallback counters, per-observation field
values and a screenshot every 15 s.

## Setup

* Skirmish: random map (Côte), Franks vs a very easy AI, 1680x1050 full screen,
  French UI. Cheats enabled in the lobby: food and wood were topped up with
  the chat cheats so the Town Center never ran dry, and houses were placed by
  hand **before** the runs (no manual input during any run).
* Four runs, back to back, same game:

| run | controller | OCR | cycles | duration | game time |
|---|---|---|---|---|---|
| `tc_hybrid` | Town Center | hybrid | 250 | 9.5 min | 0:30 → 0:45 |
| `tc_tesseract` | Town Center | tesseract | 250 | 9.5 min | 0:46 → 1:01 |
| `idle_hybrid` | idle villager | hybrid | 400 | 17.4 min | 1:01 → 1:30 |
| `idle_tesseract` | idle villager | tesseract | 400 | 17.7 min | 1:30 → 1:59 |

The runs follow each other, so the game state differs between the hybrid and
Tesseract runs of a controller (population 12→41 vs 42→60; the idle runs saw a
static 59 idle villagers). Accuracy is therefore also compared on the **same
frames**: every saved screenshot (198) was re-read with OpenCV only and with
Tesseract only, and the values checked by eye against the top bars.

Raw data: `docs/benchmarks/live/` (`*_summary.json`, `*_cycles.jsonl`,
`*_frame_check.json`).

## Perception latency (live, in game, region cache on)

| run | observations | mean | median | P95 | Tesseract launches / obs |
|---|---|---|---|---|---|
| tc_hybrid | 248 | 1065 ms | 1039 ms | 1456 ms | 3.61 |
| tc_tesseract | 267 | 1079 ms | 1065 ms | 1378 ms | 4.22 |
| idle_hybrid | 446 | 1268 ms | 1295 ms | 1492 ms | 4.68 |
| idle_tesseract | 459 | 1278 ms | 1300 ms | 1491 ms | 4.80 |

**No measurable latency gain live.** The controllers' observations are bound
by Tesseract reading *text*: the Town Center queue lines and the selected
unit's name / HP, read on every observation. The numeric fields that HybridOcr
takes over were already cheap live: the region cache skips them whenever their
pixels did not change, which is most frames. The saving is 0.1–0.6 Tesseract
launches per observation, which does not shorten the critical path. The large
gains measured offline (full / economy profiles on saved frames, see
[digit_recognition.md](digit_recognition.md)) hold for numeric-heavy
observations, not for these two controllers today.

## Numeric readings

Live (as recorded by each run, in-game observations):

| run | numeric reads | by OpenCV | Tesseract fallback | fallback rate | not VALID |
|---|---|---|---|---|---|
| tc_hybrid | 496 | 80 | 21 | 20.8 % | 1 (population, stale) |
| tc_tesseract | 534 | – | – | – | 19 (population) |
| idle_hybrid | 446 | 1 | 0 | 0 % | 0 |
| idle_tesseract | 459 | – | – | – | 0 |

"Numeric reads" counts fields per observation; only changed regions are
actually recognised (cache), hence the small OpenCV / fallback counts.

Same 198 frames, each engine alone (1386 field readings with a known truth):

| engine | correct | UNKNOWN | **wrong** |
|---|---|---|---|
| OpenCV only | 1361 | 24 (population 22, food 2) | **0** |
| Tesseract only | 1351 | 25 (food 17, game time 3, population 3, wood 2) | **9** (idle count) |
| hybrid (OpenCV, Tesseract on UNKNOWN) | 1385 | 1 | **0** |

* **Tesseract-only accepts wrong idle counts.** On the idle button, counts
  50–58 are read as 30–38 with enough confidence to be VALID (9 frames). OpenCV
  reads them correctly. The idle-villager runs did not hit this (count 59), but
  a Tesseract-only idle controller would plan a 30-villager pass for 50.
* **Tesseract misses the yellow near-cap population** (`56/60`, `57/60`,
  `59/60`): the 19 UNKNOWN populations of `tc_tesseract`. OpenCV reads them.
* **OpenCV refuses some white populations `NN/60`** where the `6` and `0`
  touch (e.g. `20/60`…`29/60`, `36/60`, `40/60`, `41/60`) and the pink
  "housed" flash. These are the 21 live fallbacks; Tesseract then read them
  correctly. Cause: only two `6` samples in the templates, none touching a `0`.
  The fix is more templates (rebuild including live frames), not looser
  matching; it is left out of this change.
* Every OpenCV value was checked against the screenshots; all were right,
  including the clock passing one hour (`01:00:04`) and 5-digit resources.

## Controller actions

| run | actions | verified on screen | failed / uncertain |
|---|---|---|---|
| tc_hybrid | 31 villagers queued, 1 TC selection | 31/31 (food −50 each) | 0 / 0 |
| tc_tesseract | 17 villagers queued (housed 60/60 for half the run) | 17/17 | 0 / 0 |
| idle_hybrid | 59 idle villagers selected, one full pass | 59/59 (villager name) | 0 / 0 |
| idle_tesseract | 59 idle villagers selected, one full pass | 59/59 | 0 / 0 |

No incorrect action in any run. The foreground guard blocked all input while
another window was in front (34 cycles in `tc_hybrid`, 13 in `idle_hybrid`),
as designed.

## Issues and limitations found

1. **Latency claim was offline-only.** PR #7's docs said the controllers'
   observations get faster; live they do not (table above). Docs corrected.
   Real speed-ups for these controllers need the text reads (TC queue, selected
   name) to stop costing a Tesseract launch each observation.
2. **Population fallback rate ~20 % at `NN/60`** (touching `60`). Correct
   values, but Tesseract is still launched. Template coverage, fixable
   separately.
3. **The idle runs barely exercised the digit reader** (count stuck at 59,
   static economy): the idle-count benefit is shown by the frame re-read, not
   by a live action. Selection verification is unchanged (Tesseract reads the
   name; `Villa eois` at conf 0.58 on 51 selections, accepted by fuzzy match).
4. One game, one resolution, one UI language, very easy AI, cheats for
   resources (5-digit food/wood the whole time).
5. The first live attempt was discarded: manual house placement while the
   Town Center controller ran made clicks land on the Town Center's "train"
   button after the controller re-selected it. Playing alongside a running
   controller is not safe with mouse input on the command grid.
