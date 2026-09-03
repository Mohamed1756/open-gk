"""Transferable pressing types: matched on team_id, never on names or colors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.geometry import PitchPoint


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


@dataclass(frozen=True)
class PressingSnapshot:
    passer: PressingActor
    receivers: List[PressingActor] = field(default_factory=list)
    opponents: List[PressingActor] = field(default_factory=list)
    ball_pos_m: Optional[PitchPoint] = None
    attack_dir_x: float = 1.0
    timestamp_s: float = 0.0
    memory_weights: Dict[int, float] = field(default_factory=dict)
