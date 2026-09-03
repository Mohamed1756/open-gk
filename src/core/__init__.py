"""
Core types, schemas, and geometric utilities for GK modeling.
"""

from src.core.types import (
    ActionType,
    CrossOutcome,
    ShotOutcome,
    DistributionOutcome,
    GKDecision,
    Period,
    DistributionType,
)
from src.core.geometry import (
    PitchPoint,
    is_inside_penalty_box,
    is_inside_six_yard_box,
    calculate_goal_angle,
    calculate_distance_to_goal,
)
from src.core.schemas import DistributionSituation

__all__ = [
    "ActionType",
    "CrossOutcome",
    "ShotOutcome",
    "DistributionOutcome",
    "GKDecision",
    "Period",
    "DistributionType",
    "PitchPoint",
    "is_inside_penalty_box",
    "is_inside_six_yard_box",
    "calculate_goal_angle",
    "calculate_distance_to_goal",
    "DistributionSituation",
]
