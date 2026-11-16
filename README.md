# aoe2-ai-autoplay

An experimental AI agent that plays **Age of Empires II** by looking at the screen and controlling the mouse and keyboard, the same way a human player does.

The agent runs a simple loop:

1. **See**: capture the game window (`mss`) and pre-process the frame (`opencv-python`, `numpy`, `pillow`).
2. **Think**: send the game state (screenshot and/or extracted info) to an LLM (OpenAI or Anthropic) and get back a structured action.
3. **Act**: execute that action in the game with `pyautogui` (clicks, hotkeys, camera moves).

> ⚠️ Early work in progress. Use it in single-player / vs-AI games only. Automating input in ranked multiplayer may break the game's terms of service.

## Requirements

- Windows 10/11
- Age of Empires II (Definitive Edition recommended)
- Python **3.12** (with the `py` launcher)
- An API key for OpenAI and/or Anthropic

## Setup

```powershell
git clone git@github.com:Marcanta/aoe2-ai-autoplay.git
cd aoe2-ai-autoplay

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install pydantic python-dotenv openai anthropic   # later: LLM decision step
```

Perception reads the HUD with **Tesseract OCR**, a separate program:

```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```

It is found on `PATH`, in `C:\Program Files\Tesseract-OCR\`, or via the `TESSERACT_CMD` environment variable.

If activation fails with *"running scripts is disabled on this system"*, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Configuration

Create a `.env` file at the project root (never commit it):

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Settings are loaded with `python-dotenv` and validated with `pydantic`.

## Dependencies

| Package | Role |
|---|---|
| `mss` | Fast screen capture of the game window |
| `opencv-python` | Image processing (UI detection, template matching) |
| `numpy` | Frame / array manipulation |
| `pillow` | Image conversion and encoding for LLM input |
| `pyautogui` | Mouse and keyboard control |
| `pydantic` | Typed config and structured LLM actions |
| `python-dotenv` | Load API keys from `.env` |
| `openai`, `anthropic` | LLM clients for the decision step |

## Perception

`aoe2_ai/perception/` turns a screenshot into a `GameState`: wood, food, gold, stone, population, game time, age and the selected unit/building. Every field is a `Detection` with `value`, `confidence` (0–1), `timestamp` and `status`:

- `valid`: read from this frame and passed validation.
- `stale`: unreadable now. `value` is the last valid reading, still recent (5 s by default), and `timestamp` is when it was read.
- `unknown`: no trustworthy value, and `value` is `None`. An OCR failure is never reported as 0.

```python
from aoe2_ai.perception import LiveSource, Perception

perception = Perception()
with LiveSource() as source:
    state = perception.analyze(source.grab()).state
print(state.to_json())
```

How a frame is read: check the in-game HUD is visible (template match on the wood icon) → crop each region (`regions_1680x1050.json`, scaled to the screen size) → upscale and binarise with OpenCV → read with Tesseract over several thresholds → vote → parse and validate → `StateTracker` handles staleness.

```powershell
python -m aoe2_ai.perception --live --debug                    # live overlay + OCR crops, q / Esc to quit
python -m aoe2_ai.perception --image shot.png --save captures/perception   # frame, overlay, crops, state JSON
python -m aoe2_ai.perception --calibrate --image shot.png      # draw regions, save raw crops (no OCR)
python -m aoe2_ai.perception.capture --name feudal --count 5 --interval 20  # grab test fixtures
python -m pytest tests
```

Notes:
- The game clock must be shown in game (top bar, right of the age).
- While an age-up is researched, the top bar shows the *target* age over a red progress bar. Perception reports the current age, with "(researching)" in `raw_text`.
- Selected-object names are raw OCR from the English model, so some French accents are lost.
- One frame takes about 2 s, mostly Tesseract process start-up on Windows.
- Regions were measured at 1680x1050. Other 16:10 resolutions scale automatically; other aspect ratios need recalibrating (`--calibrate`).

## Town Center controller

`aoe2_ai/controllers/town_center.py` keeps the Town Center producing villagers, with no LLM: observe → decide → press the hotkey → verify on screen that the queue started → repeat. All input goes through `aoe2_ai/execution/` (`ActionExecutor`), which only sends keys when the game is the foreground window, logs every action, and refuses a second training command until the first one is verified.

```powershell
python -m aoe2_ai.controllers --dry-run -v   # decide and log, send nothing
python -m aoe2_ai.controllers                 # live; F12 = emergency stop
```

Details, safety rules and the perception investigation: [docs/town_center_controller.md](docs/town_center_controller.md).

## Safety

`pyautogui` has a built-in **fail-safe**: slam the mouse into any **screen corner** to abort the agent immediately. Keep it enabled. The Town Center controller also stops on **F12** (`--stop-key`) and never sends input unless Age of Empires II is the foreground window.

## Roadmap

- [ ] Reliable game-window capture and resolution handling
- [x] Read resources, population, game time, age and selected object from the HUD (`aoe2_ai/perception`)
- [ ] Idle villagers; faster OCR
- [x] Town Center: queue idle/producing, villager button (`aoe2_ai/perception/town_center.py`)
- [x] Safe action execution and a deterministic villager-production controller
- [ ] Action schema (build, train, gather, attack, scout) with pydantic
- [ ] LLM decision loop with rate limiting
- [ ] Opening build orders (e.g. scouts / archers)