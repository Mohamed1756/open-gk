"""Unit structure: press lines vs buildup shape, jump triggers, intensity.

Lines are 1D clusters on the attack axis. A jump trigger fires when the
most advanced line's mean closing velocity toward the ball spikes, i.e. the
press jumps as a unit rather than drifting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from src.models.pressing.types import PressingActor

LINE_GAP_M = 8.0
JUMP_SPEED_MS = 4.0


@dataclass(frozen=True)
class UnitLines:
    team_id: str
    line_counts: Tuple[int, ...]
    label: str


@dataclass(frozen=True)
class PressState:
    jumping: bool
    intensity_ms: float
    first_line_count: int
    note: str


def cluster_lines(
    actors: List[PressingActor], team_id: str, attack_dir_x: float = 1.0
) -> UnitLines:
    members = [a for a in actors if a.team_id == team_id and a.pitch_valid]
    if not members:
        return UnitLines(team_id=team_id, line_counts=(), label="0")
    ordered = sorted(members, key=lambda a: attack_dir_x * a.pos_m.x, reverse=True)
    lines: List[int] = [1]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = attack_dir_x * (prev.pos_m.x - cur.pos_m.x)
        if gap >= LINE_GAP_M:
            lines.append(1)
        else:
            lines[-1] += 1
    counts = tuple(lines)
    return UnitLines(
        team_id=team_id, line_counts=counts, label="-".join(str(c) for c in counts)
    )


def press_state(
    opponents: List[PressingActor],
    ball_x: float,
    ball_y: float,
    attack_dir_x: float = 1.0,
) -> PressState:
    valid = [o for o in opponents if o.pitch_valid]
    if not valid:
        return PressState(False, 0.0, 0, "no-press")
    first_x = max(attack_dir_x * o.pos_m.x for o in valid)
    first_line = [o for o in valid if first_x - attack_dir_x * o.pos_m.x < LINE_GAP_M]
    closings = []
    for o in first_line:
        dx = ball_x - o.pos_m.x
        dy = ball_y - o.pos_m.y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            closings.append(0.0)
            continue
        vx, vy = o.vel_ms
        closings.append((vx * dx + vy * dy) / dist)
    intensity = sum(closings) / len(closings)
    jumping = intensity >= JUMP_SPEED_MS
    note = "jump" if jumping else "set"
    return PressState(
        jumping=jumping,
        intensity_ms=round(intensity, 2),
        first_line_count=len(first_line),
        note=note,
    )
