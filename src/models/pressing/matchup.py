"""Presser-to-outlet assignment: who jumps whom, who is free.

Cost = arrival time at the receiver's lead target. The nearest-to-ball
presser is pre-committed to the passer; everyone else is matched by the
Hungarian algorithm. Unmatched outlets are structurally free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.geometry import PitchPoint
from src.models.pressing.arrival import (
    flight_time_s,
    lead_target,
    presser_arrival_s,
    presser_engagement_latency_s,
)
from src.models.pressing.types import (
    DefenderRole,
    DefenderTask,
    DefenderTaskKind,
)
from src.models.pressing.units import (
    SCREEN_MIN_PROJ_M,
    SCREEN_RADIUS_M,
    classify_defender_role,
)
from src.models.pressing.types import PressingActor, PressingSnapshot

PINCER_TOLERANCE_S = 0.35


@dataclass(frozen=True)
class OutletMatchup:
    receiver_track_id: int
    presser_track_id: Optional[int]
    arrival_delta_s: float
    is_free: bool
    is_pincer: bool
    second_presser_id: Optional[int] = None


@dataclass(frozen=True)
class MatchupResult:
    outlets: List[OutletMatchup] = field(default_factory=list)
    ball_presser_id: Optional[int] = None
    defender_tasks: Dict[int, DefenderTask] = field(default_factory=dict)


def _lane_proximity(op_pos: PitchPoint, ball: PitchPoint, target: PitchPoint) -> float:
    """Perpendicular distance from a defender to a passing corridor ray.

    Returns inf when the defender projects outside the ball-to-target segment,
    mirroring the SCREEN occupation rule, so off-ray defenders collect no lane.
    """
    rx = target.x - ball.x
    ry = target.y - ball.y
    ray_len = math.hypot(rx, ry)
    if ray_len < 1e-3:
        return math.inf
    rux, ruy = rx / ray_len, ry / ray_len
    ox = op_pos.x - ball.x
    oy = op_pos.y - ball.y
    proj = ox * rux + oy * ruy
    if not SCREEN_MIN_PROJ_M <= proj <= ray_len:
        return math.inf
    return math.sqrt(max(0.0, ox * ox + oy * oy - proj * proj))


def _arrival_cost(
    presser: PressingActor, target: PitchPoint, ball: PitchPoint
) -> float:
    engage, _ = presser_engagement_latency_s(presser, target, ball)
    arrival, _ = presser_arrival_s(presser, target, engagement_latency_s=engage)
    return arrival


def assign_pressers(snapshot: PressingSnapshot) -> MatchupResult:
    valid_opp = [o for o in snapshot.opponents if o.pitch_valid]
    if not valid_opp or not snapshot.receivers:
        return MatchupResult(
            outlets=[
                OutletMatchup(r.track_id, None, 99.0, True, False)
                for r in snapshot.receivers
            ]
        )

    ball = snapshot.ball_pos_m or snapshot.passer.pos_m
    ball_presser = min(
        valid_opp,
        key=lambda o: math.hypot(o.pos_m.x - ball.x, o.pos_m.y - ball.y),
    )
    rest = [o for o in valid_opp if o.track_id != ball_presser.track_id]

    targets: Dict[int, PitchPoint] = {}
    for r in snapshot.receivers:
        raw = math.hypot(
            r.pos_m.x - snapshot.passer.pos_m.x, r.pos_m.y - snapshot.passer.pos_m.y
        )
        targets[r.track_id] = lead_target(r.pos_m, r.vel_ms, flight_time_s(raw))

    from scipy.optimize import linear_sum_assignment

    rec_ids = [r.track_id for r in snapshot.receivers]
    cost = np.zeros((len(rest), len(rec_ids)), dtype=np.float64)
    for i, op in enumerate(rest):
        for j, rid in enumerate(rec_ids):
            cost[i, j] = _arrival_cost(op, targets[rid], ball)
    row_ind, col_ind = linear_sum_assignment(cost)

    assigned: Dict[int, Tuple[int, float]] = {}
    for i, j in zip(row_ind, col_ind):
        assigned[rec_ids[j]] = (rest[i].track_id, float(cost[i, j]))

    outlets = []
    for r in snapshot.receivers:
        target = targets[r.track_id]
        if r.track_id in assigned:
            pid, delta = assigned[r.track_id]
            second: Optional[int] = None
            best_second_arrival = math.inf
            pincer = False
            for op in rest:
                if op.track_id == pid:
                    continue
                engage, _ = presser_engagement_latency_s(op, target, ball)
                arrival, _ = presser_arrival_s(op, target, engagement_latency_s=engage)
                if abs(arrival - delta) <= PINCER_TOLERANCE_S and arrival < 99.0:
                    pincer = True
                    if arrival < best_second_arrival:
                        second = op.track_id
                        best_second_arrival = arrival
            outlets.append(
                OutletMatchup(r.track_id, pid, round(delta, 3), False, pincer, second)
            )
        else:
            nearest = min(
                rest,
                key=lambda o: math.hypot(o.pos_m.x - target.x, o.pos_m.y - target.y),
                default=None,
            )
            if nearest is None:
                delta = 99.0
            else:
                engage, _ = presser_engagement_latency_s(nearest, target, ball)
                delta, _ = presser_arrival_s(
                    nearest, target, engagement_latency_s=engage
                )
            outlets.append(
                OutletMatchup(
                    r.track_id,
                    nearest.track_id if nearest else None,
                    round(delta, 3),
                    True,
                    False,
                )
            )
    rid_of_pid = {pid: rid for rid, (pid, _) in assigned.items()}
    corridors = [(f"lane:{rid}", targets[rid]) for rid in rec_ids]
    corridor_pts = [t for _, t in corridors]
    goal_sign = 1.0 if snapshot.attack_dir_x >= 0 else -1.0
    tasks: Dict[int, DefenderTask] = {}
    roles = {
        op.track_id: classify_defender_role(op, ball, corridor_pts, goal_sign=goal_sign)
        for op in valid_opp
    }
    for op in valid_opp:
        role = roles[op.track_id]
        if op.track_id == ball_presser.track_id:
            tasks[op.track_id] = DefenderTask(
                DefenderTaskKind.MAN, "ball", reason="ball-press", role=role
            )
        elif op.track_id in rid_of_pid:
            tasks[op.track_id] = DefenderTask(
                DefenderTaskKind.MAN,
                str(rid_of_pid[op.track_id]),
                reason="hungarian-mark",
                role=role,
            )
        else:
            if role is DefenderRole.ENGAGE:
                near = min(
                    rec_ids,
                    key=lambda rid: math.hypot(
                        targets[rid].x - op.pos_m.x, targets[rid].y - op.pos_m.y
                    ),
                )
                tasks[op.track_id] = DefenderTask(
                    DefenderTaskKind.MAN,
                    str(near),
                    reason="surplus-converger",
                    role=role,
                )
            elif role in (DefenderRole.SCREEN, DefenderRole.CONTAIN):
                ranked = sorted(
                    (_lane_proximity(op.pos_m, ball, t), key) for key, t in corridors
                )
                close = [(d, k) for d, k in ranked[:2] if d <= SCREEN_RADIUS_M]
                if not close:
                    tasks[op.track_id] = DefenderTask(
                        DefenderTaskKind.NONE, reason="no-lane-in-reach", role=role
                    )
                elif len(close) == 1:
                    tasks[op.track_id] = DefenderTask(
                        DefenderTaskKind.LANE,
                        close[0][1],
                        reason="lane-cover",
                        role=role,
                    )
                else:
                    tasks[op.track_id] = DefenderTask(
                        DefenderTaskKind.LANE,
                        close[0][1],
                        weight=0.5,
                        second_target_id=close[1][1],
                        second_weight=0.5,
                        reason="split-stance-cover",
                        role=role,
                    )
            else:
                tasks[op.track_id] = DefenderTask(
                    DefenderTaskKind.NONE,
                    reason=(
                        "recovering"
                        if role is DefenderRole.RECOVER
                        else "deep-beyond-horizon"
                    ),
                    role=role,
                )
    return MatchupResult(
        outlets=outlets,
        ball_presser_id=ball_presser.track_id,
        defender_tasks=tasks,
    )
