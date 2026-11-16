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

pip install mss opencv-python pyautogui numpy pillow
pip install pydantic python-dotenv openai anthropic
```

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

## Safety

`pyautogui` has a built-in **fail-safe**: slam the mouse into any **screen corner** to abort the agent immediately. Keep it enabled.

## Roadmap

- [ ] Reliable game-window capture and resolution handling
- [ ] Read resources, population and idle villagers from the HUD
- [ ] Action schema (build, train, gather, attack, scout) with pydantic
- [ ] LLM decision loop with rate limiting
- [ ] Opening build orders (e.g. scouts / archers)