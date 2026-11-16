# Town Center controller (MVP 3)

A deterministic controller that keeps the Town Center producing villagers. No LLM.

```
aoe2_ai/perception/town_center.py   TC panel: selected? queue idle/producing? villager button? progress?
aoe2_ai/execution/                  ActionExecutor: the only code that sends input
aoe2_ai/controllers/observation.py  Observation = GameState + TownCenterState + game-detected flag
aoe2_ai/controllers/town_center.py  decide() rules + TownCenterController closed loop + CLI
```

## Running it

1. Start a skirmish and let the game load. Your Town Center hotkey must select it (default **H**) and
   villager training must be on **Q** (the default grid layout). Different keys: `--select-key`, `--train-key`.
2. From the activated venv:

```powershell
python -m aoe2_ai.controllers --dry-run -v   # observe + decide, log "would train", send nothing
python -m aoe2_ai.controllers                 # live
```

3. Switch to the game within 5 seconds (`--delay`).
4. **Stop**: press **F12** (`--stop-key`), or slam the mouse into a screen corner (pyautogui fail-safe), or Ctrl+C.

The controller presses **H** whenever the Town Center is not selected (at most every 2 s), which also
moves the camera. Do not play the same game by hand at the same time.

Failures are written to `captures/controller/` as a PNG of the frame plus the full observation as JSON.

Timings, costs and hotkeys are fields of `TownCenterConfig`; pass a JSON file with `--config`:

```json
{"select_hotkey": "h", "train_hotkey": "q", "verify_timeout": 10.0, "failure_backoff": 10.0}
```

## The loop

`TownCenterController.cycle()` runs Observe → Evaluate → Act → Wait → Verify, and `run()` repeats it.

**Evaluate** (`decide()`, a pure function) only answers *train* when every input is positively known:

| Check | Not met → |
|---|---|
| Game window in front **and** in-game HUD visible | wait |
| Selected object read and named Town Center (`Forum`, `Town Center`, …) | select TC |
| Queue seen **idle** (UNKNOWN is not idle) | wait |
| Food read **this frame** (UNKNOWN and STALE are not trusted) and ≥ 50 | wait |
| Villager button seen in the command grid | wait |
| Not housed (only when population was read; unknown population does not block) | wait |

**Act** goes through `ActionExecutor`. **Verify** polls observations for up to `verify_timeout`:

- queue seen producing → success (food before → after is logged as extra evidence);
- queue seen idle `idle_confirmations` times → the command did not take: failure, outcome known;
- otherwise (TC deselected, game lost focus, queue unreadable) → **UNCERTAIN**.

States (`ControllerState`): `observing`, `selecting`, `verifying_selection`, `training`,
`verifying_training`, `uncertain`, `backoff`, `halted`, `stopped`.

### Failures, uncertainty and retries

- Every failure saves a debug screenshot, is appended to `controller.failures` and starts a
  **back-off** (`failure_backoff`, 10 s) during which no training command is sent. No blind retry.
- **UNCERTAIN** (the training key may or may not have reached the game, e.g. the input call raised or
  verification timed out): no training command at all. The controller keeps observing, re-selecting the
  TC if needed so it can see the queue, until it sees the queue producing or idle. Only then is the
  pending command resolved. After `max_uncertain_cycles` it halts.
- After `max_consecutive_failures` (3) the controller **halts**: something is wrong (wrong hotkey,
  moved UI, different resolution) and a human should look at the screenshots.

## ActionExecutor

Every `execute()` passes these gates, in order, and every attempt is logged (`aoe2_ai.execution`) and kept in `history`:

1. emergency stop not triggered;
2. no **pending** action with the same `dedupe_key`: an action sent with `await_verification=True`
   stays pending until the caller calls `resolve()`. It is marked pending *before* the input is sent, so
   an input error leaves it pending: duplicate training commands are impossible even if the controller
   misbehaves;
3. the action's `cooldown` has elapsed;
4. the **game is the foreground window** (`ForegroundWindowGuard`: process `AoE2DE_s.exe`, or the window
   title when the process can't be read), checked again after `pre_delay`.

`dry_run=True` runs every gate but sends nothing. An input exception returns `FAILED` (uncertain);
pyautogui's corner fail-safe also triggers the emergency stop.

## Perception: Town Center panel

All regions are in `regions_1680x1050.json` (`tc_*`) and scale like the HUD regions.

| Field | How | Evidence needed |
|---|---|---|
| `selected` | existing selection-name OCR, fuzzy match to TC names in several languages | name read this frame (not the stale tracked one) |
| `queue` | **pixel classification**: progress bar (red fill + dark track) → producing; parchment matching `anchors/tc_queue_empty.png` → idle | parchment strip under the bar must be visible, else UNKNOWN |
| `villager_available` | **template match** of `anchors/villager_button.png` in the command grid | score ≥ 0.8, else UNKNOWN (never "false" from a miss) |
| `progress`, `current_item` | OCR of the two text lines next to the queue icon, only while producing | informational only, not used to decide |

### Template / pixel matching vs OCR (investigation)

Measured on the 12 fixture frames (two have the Town Center selected: one researching Feudal Age, one idle):

| Signal | Producing frame | Idle frame | Other frames | Cost |
|---|---|---|---|---|
| Progress-bar fill | 0.99 | 0.00 | 0.00 (main menu: 0.85, rejected by the parchment check) | < 2 ms |
| Empty-queue template | 0.16 | 1.00 | 1.00 on unit panels (hence the TC-selected gate) | (same) |
| Villager button template | 1.00 | 1.00 | ≤ 0.46 | (same) |
| OCR of progress line | `Développement 22%` → 22 | — | — | ~100–200 ms per Tesseract launch |

Conclusions:

- For the **queue and UI indicators, pixel/template matching is more reliable than OCR.** An idle queue
  has no text at all, so OCR can only return "nothing read", which is exactly the failure mode that must
  not be read as "idle". The bar and the empty parchment are large, high-contrast and give wide margins.
- **OCR stays the right tool for names and percentages** (selected object, progress), where the content
  is text and a fuzzy match handles the language.
- Pixel tests alone can be fooled by unrelated dark areas (the main menu scores 0.85 as a "bar"), so every
  queue verdict also requires the panel's parchment to be visible, and the whole TC panel is only
  interpreted when the selected object is a Town Center.

### Known limits

- No fixture yet shows a Town Center **training a villager** (only age research). Production is detected
  from the progress bar, which is the same widget, so it should hold; `current_item` = `villager` relies on
  OCR of the item name and is not used for decisions. Grab a frame with
  `python -m aoe2_ai.perception.capture --name tc_villager` to add a fixture.
- Templates were cut at 1680x1050 with the French UI and the Franks; other villager icons (civ, UI mod)
  or a different command-grid layout need a new `villager_button.png` / region.
- A full observation takes ~2 s (Tesseract on all HUD fields), so a finished villager may wait a few
  seconds before the next one is queued.
