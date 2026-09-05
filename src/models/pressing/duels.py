"""Duel timelines: presser-vs-receiver behaviour carried across frames.

A duel is one receiver-presser pair tracked through the pre-decision window
(and later to resolution). Per-frame margins reuse arrival_margin; phases reuse
the window hysteresis family; incumbency is sticky (Schmitt-style) so Hungarian
re-runs cannot flicker the mark mid-duel. All functions are pure over plain
tracking data: same inputs, same outputs, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.core.geometry import PitchPoint
from src.models.pressing.arrival import arrival_margin
from src.models.pressing.types import PressingActor
from src.models.pressing.units import classify_defender_role
from src.physics.gk_constraints import (
    WINDOW_CLOSE_THRESHOLD_S,
    WINDOW_OPEN_THRESHOLD_S,
)

# Incumbent keeps the mark unless a challenger beats it by more than this.
# Schmitt-style, mirrors the 0.10-0.20 window hysteresis family (algorithmic,
# not physics, so it lives here rather than in gk_constraints).
STICKY_CHALLENGE_TOL_S = 0.15

SHADOWING = "SHADOWING"
CLOSING = "CLOSING"
CONTESTED = "CONTESTED"


@dataclass(frozen=True)
class DuelFrame:
    timestamp_s: float
    margin_s: float
    t_ball_s: float
    t_press_s: float
    presser_role: str
    phase: str


@dataclass(frozen=True)
class DuelTrack:
    receiver_track_id: int
    presser_track_id: int
    frames: List[DuelFrame] = field(default_factory=list)


def duel_phase(margin_s: float) -> str:
    """Phase from margin using the window hysteresis thresholds."""
    if margin_s >= WINDOW_OPEN_THRESHOLD_S:
        return SHADOWING
    if margin_s > WINDOW_CLOSE_THRESHOLD_S:
        return CLOSING
    return CONTESTED


def duel_frame_at(
    receiver: PressingActor,
    presser: PressingActor,
    ball_pos: PitchPoint,
    timestamp_s: float,
    corridor_targets: Optional[List[PitchPoint]] = None,
    goal_sign: float = 1.0,
) -> DuelFrame:
    """One duel frame: margin, presser role, and phase for the pair."""
    res = arrival_margin(receiver, [presser], ball_pos)
    role = classify_defender_role(
        presser, ball_pos, corridor_targets, goal_sign=goal_sign
    )
    return DuelFrame(
        timestamp_s=round(timestamp_s, 3),
        margin_s=res.margin_s,
        t_ball_s=res.t_ball_s,
        t_press_s=res.t_press_s,
        presser_role=role.value,
        phase=duel_phase(res.margin_s),
    )


def track_duel(
    receiver_track_id: int,
    presser_track_id: int,
    receiver_frames: List[PressingActor],
    presser_frames: List[PressingActor],
    ball_positions: List[PitchPoint],
    timestamps_s: List[float],
    corridor_targets: Optional[List[PitchPoint]] = None,
    goal_sign: float = 1.0,
) -> DuelTrack:
    """Build a duel timeline from aligned per-frame sequences."""
    if not (
        len(receiver_frames)
        == len(presser_frames)
        == len(ball_positions)
        == len(timestamps_s)
    ):
        raise ValueError("duel sequences must align in length")
    frames = [
        duel_frame_at(rec, pre, ball, t, corridor_targets, goal_sign)
        for rec, pre, ball, t in zip(
            receiver_frames, presser_frames, ball_positions, timestamps_s
        )
    ]
    return DuelTrack(
        receiver_track_id=receiver_track_id,
        presser_track_id=presser_track_id,
        frames=frames,
    )


def select_incumbent(
    prev_presser_id: Optional[int],
    candidacies: List[Tuple[int, float]],
) -> Optional[int]:
    """Sticky mark: the incumbent keeps the duel unless a challenger beats
    their margin by more than STICKY_CHALLENGE_TOL_S. Lower margin wins."""
    if not candidacies:
        return None
    best_id, best_margin = min(candidacies, key=lambda c: c[1])
    if prev_presser_id is None:
        return best_id
    prev = next((m for i, m in candidacies if i == prev_presser_id), None)
    if prev is None:
        return best_id
    if prev - best_margin <= STICKY_CHALLENGE_TOL_S:
        return prev_presser_id
    return best_id
