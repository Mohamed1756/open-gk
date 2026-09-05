"""Transferable pressing types: matched on team_id, never on names or colors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.geometry import PitchPoint


class DefenderRole(str, Enum):
    """Exhaustive per-defender designation. Every opponent in camera gets one.

    ENGAGE closes on ball/man (arrival race). CONTAIN holds goal-side space
    and lanes without sprinting (lane influence, possibly split across two).
    RECOVER sprints toward own goal (no threat to this pass). SCREEN occupies
    a passing ray (cover shadow). DEEP is beyond ball-reach within the
    decision horizon (explicitly nothing to do).
    """

    ENGAGE = "ENGAGE"
    CONTAIN = "CONTAIN"
    RECOVER = "RECOVER"
    SCREEN = "SCREEN"
    DEEP = "DEEP"


class DefenderTaskKind(str, Enum):
    MAN = "MAN"
    LANE = "LANE"
    NONE = "NONE"


@dataclass(frozen=True)
class DefenderTask:
    """Sparse designation: only real jobs. MAN tracks a track_id, LANE covers
    lane keys (fractional weights split a planted defender across two lanes),
    NONE carries the reason (recovering, deep, ...) instead of fake precision."""

    kind: DefenderTaskKind
    target_id: Optional[str] = None
    weight: float = 1.0
    second_target_id: Optional[str] = None
    second_weight: float = 0.0
    reason: str = ""
    role: Optional[DefenderRole] = None


@dataclass(frozen=True)
class PressingActor:
    track_id: int
    team_id: str
    pos_m: PitchPoint
    vel_ms: Tuple[float, float] = (0.0, 0.0)
    facing_rad: Optional[float] = None
    facing_source: str = "unknown"
    is_keeper: bool = False
    pitch_valid: bool = True
    role: Optional[DefenderRole] = None
    task: Optional[DefenderTask] = None


@dataclass(frozen=True)
class PressingSnapshot:
    passer: PressingActor
    receivers: List[PressingActor] = field(default_factory=list)
    opponents: List[PressingActor] = field(default_factory=list)
    ball_pos_m: Optional[PitchPoint] = None
    attack_dir_x: float = 1.0
    timestamp_s: float = 0.0
    memory_weights: Dict[int, float] = field(default_factory=dict)
