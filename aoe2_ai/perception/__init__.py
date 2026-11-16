"""Game perception: screenshots -> structured, confidence-scored GameState.

Independent of any gameplay decision or strategy code.

    from aoe2_ai.perception import Perception, LiveSource

    perception = Perception()
    with LiveSource() as source:
        state = perception.analyze(source.grab()).state
        if state.wood.is_valid:
            print(state.wood.value)
"""

from .capture import Frame, LiveSource, load_frame
from .game_state import Age, Detection, GameState, Population, Selection, StateTracker, Status
from .pipeline import Perception, PerceptionResult
from .regions import Region, RegionSet
from .town_center import QueueStatus, TownCenterDetector, TownCenterState
from .villager import IdleVillagerState, VillagerDetector, VillagerFingerprint, VillagerSelectionState

__all__ = [
    "Age", "Detection", "Frame", "GameState", "IdleVillagerState", "LiveSource", "Perception", "PerceptionResult",
    "Population", "QueueStatus", "Region", "RegionSet", "Selection", "StateTracker", "Status",
    "TownCenterDetector", "TownCenterState", "VillagerDetector", "VillagerFingerprint",
    "VillagerSelectionState", "load_frame",
]
