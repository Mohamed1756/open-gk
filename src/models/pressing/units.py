"""Unit structure: press lines vs buildup shape, jump triggers, intensity.

Lines are 1D clusters on the attack axis. A jump trigger fires when the
most advanced line's mean closing velocity toward the ball spikes, i.e. the
press jumps as a unit rather than drifting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from src.core.geometry import PitchPoint
from src.models.pressing.types import DefenderRole, PressingActor
from src.physics.gk_constraints import PASS_GROUND_INITIAL_SPEED_MS

LINE_GAP_M = 8.0
JUMP_SPEED_MS = 4.0
# Committed recovery run vs settled shuffle: above MIN_PRESSER_SPEED_MS (1.0),
# below unit-jump intensity (4.0).
RECOVER_SPEED_MS = 2.0
# Same closing floor as the memory press-decay trigger: a genuine sprint at the ball.
ENGAGE_CLOSING_MS = 2.5
# Mirrors the LOS obstacle radius: occupying the ray is defending.
SCREEN_RADIUS_M = 0.75
# Mirrors the LOS behind-observer cutoff: obstacles at the carrier's feet don't screen.
SCREEN_MIN_PROJ_M = 0.2
# Mirrors the approach-cut time horizon: beyond ball-reach inside it, DEEP.
INFLUENCE_HORIZON_S = 1.2


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


def classify_defender_role(
    actor: PressingActor,
    ball_pos: PitchPoint,
    corridor_targets: List[PitchPoint] | None = None,
    corridor_radius_m: float = SCREEN_RADIUS_M,
    pass_speed_ms: float = PASS_GROUND_INITIAL_SPEED_MS,
    t_ball_s: float = 0.0,
    horizon_s: float = INFLUENCE_HORIZON_S,
    goal_sign: float = 1.0,
) -> DefenderRole:
    """Exhaustive per-defender role from velocity and geometry.

    Intent lives in the legs: a sprinting scanner's head lies, so ENGAGE keys
    on closing speed, not facing. Ball-closing sprints win over recovery runs
    (a receiver-chaser also runs away from the ball); RECOVER strictly means
    sprinting toward the defended goal (goal_sign: +1 defends x=105, -1 x=0).
    Then reach (DEEP), then ray occupation (SCREEN); settled remainder is
    CONTAIN, including ahead-of-ball shufflers whose goal-side leverage is
    priced at influence time, not here.
    """
    vx, vy = actor.vel_ms
    dx = ball_pos.x - actor.pos_m.x
    dy = ball_pos.y - actor.pos_m.y
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return DefenderRole.ENGAGE

    ux, uy = dx / dist, dy / dist
    closing = vx * ux + vy * uy
    if closing >= ENGAGE_CLOSING_MS:
        return DefenderRole.ENGAGE
    goalward = vx if goal_sign >= 0 else -vx
    if goalward >= RECOVER_SPEED_MS:
        return DefenderRole.RECOVER
    if dist > pass_speed_ms * (t_ball_s + horizon_s):
        return DefenderRole.DEEP
    for target in corridor_targets or []:
        rx = target.x - ball_pos.x
        ry = target.y - ball_pos.y
        ray_len = math.hypot(rx, ry)
        if ray_len < 1e-3:
            continue
        rux, ruy = rx / ray_len, ry / ray_len
        ox = actor.pos_m.x - ball_pos.x
        oy = actor.pos_m.y - ball_pos.y
        proj = ox * rux + oy * ruy
        if SCREEN_MIN_PROJ_M <= proj <= ray_len:
            perp = math.sqrt(max(0.0, ox * ox + oy * oy - proj * proj))
            if perp <= corridor_radius_m:
                return DefenderRole.SCREEN
    return DefenderRole.CONTAIN
