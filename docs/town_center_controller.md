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
python -m aoe2_ai.controllers --ocr tesseract # live, Tesseract only (default: OpenCV digits + Tesseract)
```

3. Switch to the game within 5 seconds (`--delay`).
4. **Stop**, any of:
   - **F12** or **Pause**, with the game in front (`--stop-key f12,pause`; any key from `python -m aoe2_ai.execution --check`);
   - **Ctrl+C** in the controller's console (handled by an OS console handler, so it works even mid-OCR);
   - slam the mouse into a screen corner (pyautogui fail-safe).

   Whatever triggers it, no input is sent afterwards, and if the program has not exited 5 s later it is
   force-closed. If the stop-key watcher itself fails, that also counts as a stop.

### If the stop key does nothing

- **Laptop F-keys** often send media keys unless **Fn** is held. Run `python -m aoe2_ai.execution --check`
  and press your key: it prints what Windows receives. Then use `--stop-key` with a key that shows up.
- **Ctrl+C goes to the window in front.** With the game focused, Ctrl+C goes to the game; switch to the
  console first, or use the stop key.
- **PowerShell QuickEdit**: clicking into the console starts a text selection, which turns Ctrl+C into
  "copy" and pauses the program. The controller now turns QuickEdit off while it runs and restores it after.

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
pyautogui's corner fail-safe also triggers the emergency stop (`aoe2_ai/execution/stop.py`).

## Perception: Town Center panel

All regions are in `regions_1680x1050.json` (`tc_*`) and scale like the HUD regions.

| Field | How | Evidence needed |
|---|---|---|
| `selected` | existing selection-name OCR, fuzzy match to TC names in several languages | name read this frame (not the stale tracked one) |
| `queue` | **pixel classification**: progress bar (red fill + dark track) → producing; parchment matching `anchors/tc_queue_empty.png` → idle | a parchment strip above the icon or under the bar must be visible, else UNKNOWN |
| `villager_available` | **template match** of `anchors/villager_button.png` in the command grid | score ≥ 0.8, else UNKNOWN (never "false" from a miss) |
| `progress`, `current_item` | OCR of the two text lines next to the queue icon, only while producing | informational only, not used to decide |

### Template / pixel matching vs OCR (investigation)

Measured on the 13 fixture frames (three have the Town Center selected: researching Feudal Age, training a villager, idle):

| Signal | Producing frame | Idle frame | Other frames | Cost |
|---|---|---|---|---|
| Progress-bar fill | 0.99 (research and villager) | 0.00 | 0.00 (main menu: 0.85, rejected by the parchment check) | < 2 ms |
| Empty-queue template | 0.16 | 1.00 | 1.00 on unit panels (hence the TC-selected gate) | (same) |
| Villager button template | 1.00 | 1.00 | ≤ 0.46 | (same) |
| OCR of progress / item lines | `Développement 22%` → 22; `Création 61%` → 61, `Villageois` → villager | — | — | ~100–200 ms per Tesseract launch |

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

- Verified live (skirmish, 2026-11-16): TC selected with H, villager queued with Q, verified on screen
  (food 200 → 150, item read as villager), progress followed (37 % → 66 % → 95 %), then "housed (5/5)" and
  waiting. The villager-training frame is a test fixture (`tc_villager_20261116_203424.png`).
- Second live run: four villagers queued back to back, each verified (food 200 → 150 → 100 → 50 → 0), then
  "insufficient food". Two selection checks failed and were caught by verification, with no training key sent.
  Both happened right as the game appeared: the first seconds after loading, and mid alt-tab. The controller
  now waits for `min_in_game_observations` (2) consecutive in-game observations before sending any key.
- Yellow/red population text (nearly housed) is sometimes not read; an unknown population never blocks training.
- The panel counts as visible when either parchment strip (above the queue icon, under the bar) is clean,
  so queued-item icons under the bar don't hide the queue. When the TC is selected but its queue still
  can't be read, a capture is saved (`queue_unknown`, at most once a minute) to see why.
- Templates were cut at 1680x1050 with the French UI and the Franks; other villager icons (civ, UI mod)
  or a different command-grid layout need a new `villager_button.png` / region.
- Observations take ~1.1 s live (mean, P95 ~1.4 s), with `HybridOcr` and with Tesseract only alike:
  the numeric HUD fields are read by OpenCV, but the Town Center progress and item lines and the
  selected name still go through Tesseract on every observation, and they set the pace (see
  [live_validation.md](live_validation.md)). A finished villager may wait a moment before the next one is queued.
