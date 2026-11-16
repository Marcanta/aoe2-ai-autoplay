"""Action execution: sends keyboard/mouse input to the game, safely.

    from aoe2_ai.execution import Action, ActionExecutor, ForegroundWindowGuard

    executor = ActionExecutor(InputController(), ForegroundWindowGuard(), dry_run=True)
    result = executor.execute(Action.key("train_villager", "q", dedupe_key="train_villager"),
                              await_verification=True)
    ...  # check the screen, then
    executor.resolve("train_villager")

Independent of perception and of any decision logic.
"""

from .actions import Action, ActionKind, ActionResult, ActionStatus
from .executor import ActionExecutor, InputBackend
from .guard import CallableGuard, ForegroundWindowGuard, GameGuard, GuardResult, StaticGuard
from .stop import EmergencyStop, StopKeyWatcher

__all__ = [
    "Action", "ActionExecutor", "ActionKind", "ActionResult", "ActionStatus", "CallableGuard",
    "EmergencyStop", "ForegroundWindowGuard", "GameGuard", "GuardResult", "InputBackend",
    "StaticGuard", "StopKeyWatcher",
]
