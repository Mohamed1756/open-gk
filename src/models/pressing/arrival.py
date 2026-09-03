"""Arrival-time core: margin(j) = t_press(j) - t_ball(j), per outlet.

A pass is trapped when any opponent reaches the receiver's lead target
before the ball does, regardless of the static cushion at decision time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.core.geometry import PitchPoint
from src.models.pressing.types import PressingActor

PASS_SPEED_MS = 19.0
MIN_PASS_SPEED_MS = 5.0
MAX_FLIGHT_TIME_S = 1.2
MIN_PRESSER_SPEED_MS = 1.0


@dataclass(frozen=True)
class ArrivalResult:
    receiver_track_id: int
    margin_s: float
    t_ball_s: float
    t_press_s: float
    best_presser_id: Optional[int]
    marking_dist_m: float


def flight_time_s(dist_m: float) -> float:
    return min(MAX_FLIGHT_TIME_S, dist_m / max(MIN_PASS_SPEED_MS, PASS_SPEED_MS))


def lead_target(
    pos: PitchPoint, vel_ms: Tuple[float, float], flight_s: float
) -> PitchPoint:
    return PitchPoint(
        x=round(max(0.0, min(105.0, pos.x + vel_ms[0] * flight_s)), 2),
        y=round(max(0.0, min(68.0, pos.y + vel_ms[1] * flight_s)), 2),
    )


def presser_arrival_s(
    presser: PressingActor, target: PitchPoint
) -> Tuple[float, float]:
    dx = target.x - presser.pos_m.x
    dy = target.y - presser.pos_m.y
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return 0.0, 0.0
    vx, vy = presser.vel_ms
    closing = (vx * dx + vy * dy) / dist
    speed = max(MIN_PRESSER_SPEED_MS, closing)
    return dist / speed, dist


def arrival_margin(
    receiver: PressingActor,
    opponents: List[PressingActor],
    passer_pos: PitchPoint,
    pass_speed_ms: float = PASS_SPEED_MS,
) -> ArrivalResult:
    raw_dist = math.hypot(
        receiver.pos_m.x - passer_pos.x, receiver.pos_m.y - passer_pos.y
    )
    flight_s = min(MAX_FLIGHT_TIME_S, raw_dist / max(MIN_PASS_SPEED_MS, pass_speed_ms))
    target = lead_target(receiver.pos_m, receiver.vel_ms, flight_s)
    t_ball = flight_s

    best_presser: Optional[int] = None
    t_press = math.inf
    marking = math.inf
    for op in opponents:
        if not op.pitch_valid:
            continue
        arrival, dist = presser_arrival_s(op, target)
        if dist < marking:
            marking = dist
        if arrival < t_press:
            t_press = arrival
            best_presser = op.track_id

    if math.isinf(t_press):
        t_press = 99.0
    if math.isinf(marking):
        marking = 99.0
    return ArrivalResult(
        receiver_track_id=receiver.track_id,
        margin_s=round(t_press - t_ball, 3),
        t_ball_s=round(t_ball, 3),
        t_press_s=round(t_press, 3),
        best_presser_id=best_presser,
        marking_dist_m=round(marking, 2),
    )
