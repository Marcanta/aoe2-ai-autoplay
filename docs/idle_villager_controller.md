# Idle villager controller (MVP 4.1)

Finds idle villagers and selects them, one at a time, verifying each selection on screen.
It gives them **no orders**: gathering and building come later. No LLM.

```
aoe2_ai/perception/villager.py         idle button: some / none / count; is one villager selected, and which one
aoe2_ai/controllers/idle_villager.py   decide() + IdleVillagerController closed loop + CLI
aoe2_ai/controllers/observation.py     Observation now also carries `idle` and `villager` (this frame only)
aoe2_ai/controllers/runtime.py         live-run safety net shared with the Town Center controller
```

It reuses the existing `LiveObserver`, `ActionExecutor`, `ForegroundWindowGuard`, emergency stop and
`GameState`. The only input it sends is the game's own "next idle villager" hotkey, through the executor.

## Running it

1. Start a skirmish. The "Go to next idle villager" hotkey must be **`.`** (AoE2 DE default);
   otherwise pass `--idle-key`.
2. From the activated venv:

```powershell
python -m aoe2_ai.controllers idle-villager --dry-run -v   # observe + decide, log "would select", send nothing
python -m aoe2_ai.controllers idle-villager                 # live
```

3. Switch to the game within 5 seconds (`--delay`).
4. Stop exactly like the Town Center controller: **F12** / **Pause** (`--stop-key`), **Ctrl+C** in the
   console, or the mouse in a screen corner. No input is sent after a stop.

Do not run it at the same time as the Town Center controller: both change the selection.
Each press also moves the camera to the selected villager.

Failures are saved to `captures/idle_villager/` (frame PNG + observation JSON).
Settings are `IdleVillagerConfig` fields; pass a JSON file with `--config`:

```json
{"idle_hotkey": ".", "verify_timeout": 5.0, "failure_backoff": 10.0}
```

## How the game shows idle villagers (investigation)

Three options were considered:

| Approach | Verdict |
|---|---|
| **Idle-villager button in the top bar** (right of the population) | **Used to detect.** Yellow when at least one villager is idle, grey with `0` when none, count in the corner. |
| **The game's "next idle villager" hotkey** (`.`) | **Used to act.** Selects the next idle villager and centres the camera on it; repeated presses cycle through all of them. Clicking the button does the same, but a key needs no screen coordinates. |
| **Selection panel** (name, HP, command grid) | **Used to verify.** Shows whether a villager, and which one, is now selected. |

Measured on the fixture frames (1680x1050, French UI):

| Signal | Some idle (3, 8, 18, 2, 1) | None idle (grey) | Other (menu, dimmed, focus lost) |
|---|---|---|---|
| Yellow share of `idle_icon` | 0.37–0.38 | 0.00 | 0.00 |
| Light-grey share of `idle_icon` | 0.02–0.06 | 0.47 | 0.00–0.12 |
| OCR of `idle_count` | `3`, `8`, `18`, `2` correct; `1` not read | reads `40` (the grey background bleeds in) | nothing |
| Villager build button template (`villager_build_button`) | 1.00 on every villager command card, incl. multi-selection | | ≤ 0.23 (Town Center, scout, menu, villager build sub-page) |

Conclusions:

- **The button colour decides "idle or not".** Wide margins, no OCR. A colour that is neither clearly
  yellow nor clearly grey (game dimmed after losing focus, menus, overlays) is **UNKNOWN**, which never
  leads to a key press.
- **The count is OCR, used only when the button is yellow**, and only to size a pass (below). A grey
  button means count 0 by itself. A count that contradicts the colour (yellow but `0`) is UNKNOWN.
  The single digit `1` is too thin for Tesseract here: the count is then UNKNOWN and the controller
  selects one villager per pass, which is exactly right for one idle villager.
- **A villager is selected when the selected-object name (read this frame) is a villager role.** An idle
  villager keeps its last task's name (`Bûcheron`, `Chasseur`, …), so every role counts, in English and
  French, matched fuzzily (OCR `Villa eois` → `villageois`). The build button confirms it, but is not
  required: after some keys the command grid shows the build sub-page instead (fixture
  `idle_one_villager_selected_…`). Button without a readable name (multi-selection) or button with a
  non-villager name is UNKNOWN.

## The loop

`IdleVillagerController.cycle()` runs Observe → Evaluate → Act → Wait → Verify; `run()` repeats it.

**Evaluate** (`decide()`, pure) answers *select* only when:

| Check | Not met → |
|---|---|
| Game window in front **and** in-game HUD visible | wait |
| Idle button read (UNKNOWN is neither yes nor no) | wait |
| Button yellow (some idle villagers) | wait ("no idle villagers") |
| Current pass has villagers left to visit | wait |

**Act**: the idle hotkey through `ActionExecutor` (`await_verification=True`, dedupe key
`select_idle_villager`: no second press until the first is verified or ruled out).

**Verify** polls for up to `verify_timeout`:

- a villager is seen selected → **success**. Exception: with several idle villagers, if it is the same
  villager that was selected before the press, the press did not move on; still the same at the
  timeout → **failure** ("unchanged");
- a non-villager is read `other_confirmations` (2) times → **failure**: the press did not take;
- otherwise (game lost focus, selection unreadable) → **uncertain**, recorded like a failure.

Selecting creates nothing and costs nothing, so unlike a training command an uncertain press does not
block the controller: the key is released, a screenshot is saved, and the controller backs off
(`failure_backoff`) before trying again. `max_consecutive_failures` (3) halts it.

### Not selecting the same villager over and over

There is no unit id on screen, so the controller identifies a selected villager by a **fingerprint**:
canonical role name, HP, and a 32x28 grey thumbnail of the middle of the screen (`view_center`). The
hotkey centres the camera on the villager it selects, so the thumbnail mostly shows where that villager
stands. Two fingerprints are the same villager when role and HP match and the thumbnails correlate at
`same_view_threshold` (0.9) or more.

It works in **passes**:

- A pass starts when the idle button changes (yellow ↔ grey, or the count changes) and has a target of
  `N` villagers: the count read on the button, or 1 when the count could not be read.
- Each verified selection of a villager not yet seen in the pass is recorded. After `N`, the pass is
  complete: the game's hotkey would only wrap around to the first one again.
- If the hotkey lands on a villager already visited (the game cycled back early), the pass ends at once
  (`pass_complete`). When two fingerprints cannot be compared (same role and HP, no thumbnail), that
  also counts as "already visited": it is safer to stop than to keep cycling.
- While the pass is complete the controller only observes, until the button changes.

## Tests

- `tests/test_villager_perception.py`: the real `LiveObserver` on 16 labelled skirmish frames
  (`_idle` / `_villager` in `tests/fixtures/labels.json`): no idle (grey), 1, 2, 3, 8 and 18 idle,
  single villager, multi-selection, scout, Town Center, main menu, and the game dimmed after losing
  focus. Plus the pure helpers (colours, count parsing, role matching, fingerprints).
- `tests/test_idle_villager_controller.py`: the loop with scripted observations and mocked input: no
  idle villagers, one, several, failed selection, unchanged selection, unknown HUD state, game losing
  focus (before and during verification), repeated selection, unknown count, pass reset, input error,
  executor de-duplication, dry run, emergency stop, settling.

## Known limits

- **Not yet verified live with this controller.** The perception is measured on frames from real
  skirmishes, and the selection logic is exercised against them and mocks. A live run must confirm that
  the hotkey centres the camera (the view fingerprint relies on it) and the timing of the verification.
- Fixture frames are HUD-only (play area blacked out), so the view part of the fingerprint is only tested
  synthetically.
- Two idle villagers standing next to each other with the same role and HP may look alike in the view
  thumbnail; the pass then ends early (safe side: fewer selections, never repeats).
- Regions and the build-button template were measured at 1680x1050 with the French UI. Other 16:10
  resolutions scale; other aspect ratios, UI mods or a different command-grid layout need recalibrating.
- A full observation still takes ~2 s (Tesseract on all HUD fields).
