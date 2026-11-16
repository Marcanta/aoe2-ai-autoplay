"""Action descriptions and results. Pure data, no side effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    KEY = "key"  # press one key: keys=("q",)
    HOTKEY = "hotkey"  # chord: keys=("ctrl", "1")
    CLICK = "click"  # mouse click at position


@dataclass(frozen=True)
class Action:
    """One input to send to the game.

    ``dedupe_key`` groups actions that must not be repeated while an earlier one
    is unverified (e.g. every "train villager" shares ``"train_villager"``).
    ``cooldown`` is the minimum time in seconds between two executions with the
    same ``dedupe_key`` (or ``name`` when there is none), verified or not.
    ``pre_delay`` / ``post_delay`` are waits before and after sending the input.
    """

    name: str
    kind: ActionKind
    keys: tuple[str, ...] = ()
    position: tuple[int, int] | None = None
    button: str = "left"
    pre_delay: float = 0.0
    post_delay: float = 0.3
    dedupe_key: str | None = None
    cooldown: float = 0.0

    def __post_init__(self) -> None:
        if self.kind in (ActionKind.KEY, ActionKind.HOTKEY) and not self.keys:
            raise ValueError(f"{self.name}: a {self.kind.value} action needs keys")
        if self.kind is ActionKind.KEY and len(self.keys) != 1:
            raise ValueError(f"{self.name}: a key action takes exactly one key (use HOTKEY for chords)")
        if self.kind is ActionKind.CLICK and self.position is None:
            raise ValueError(f"{self.name}: a click action needs a position")
        if self.pre_delay < 0 or self.post_delay < 0 or self.cooldown < 0:
            raise ValueError(f"{self.name}: delays and cooldown must be >= 0")

    @property
    def guard_key(self) -> str:
        return self.dedupe_key or self.name

    def describe(self) -> str:
        if self.kind is ActionKind.CLICK:
            return f"{self.name}: {self.button} click at {self.position}"
        return f"{self.name}: {self.kind.value} {'+'.join(self.keys)}"

    @classmethod
    def key(cls, name: str, key: str, **kwargs) -> "Action":
        return cls(name, ActionKind.KEY, (key,), **kwargs)

    @classmethod
    def hotkey(cls, name: str, *keys: str, **kwargs) -> "Action":
        return cls(name, ActionKind.HOTKEY, tuple(keys), **kwargs)

    @classmethod
    def click(cls, name: str, x: int, y: int, button: str = "left", **kwargs) -> "Action":
        return cls(name, ActionKind.CLICK, position=(x, y), button=button, **kwargs)


class ActionStatus(str, Enum):
    EXECUTED = "executed"  # input sent
    DRY_RUN = "dry_run"  # would have been sent; nothing was
    BLOCKED_NO_GAME = "blocked_no_game"  # game window not in the foreground
    BLOCKED_PENDING = "blocked_pending"  # an earlier action with the same dedupe key is unverified
    BLOCKED_COOLDOWN = "blocked_cooldown"  # same action too soon after the previous one
    BLOCKED_STOPPED = "blocked_stopped"  # emergency stop is active
    FAILED = "failed"  # the input backend raised: whether the game got it is unknown


@dataclass(frozen=True)
class ActionResult:
    action: Action
    status: ActionStatus
    started: float
    finished: float
    detail: str = ""

    @property
    def sent(self) -> bool:
        """True when input actually went to the game."""
        return self.status is ActionStatus.EXECUTED

    @property
    def blocked(self) -> bool:
        return self.status.value.startswith("blocked")

    @property
    def uncertain(self) -> bool:
        """The input may or may not have reached the game."""
        return self.status is ActionStatus.FAILED
