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
from src.models.pressing.arrival import flight_time_s, lead_target, presser_arrival_s
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


def _arrival_cost(presser: PressingActor, target: PitchPoint) -> float:
    arrival, _ = presser_arrival_s(presser, target)
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
            cost[i, j] = _arrival_cost(op, targets[rid])
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
            pincer = False
            for op in rest:
                if op.track_id == pid:
                    continue
                arrival, _ = presser_arrival_s(op, target)
                if abs(arrival - delta) <= PINCER_TOLERANCE_S and arrival < 99.0:
                    pincer = True
                    if (
                        second is None
                        or arrival
                        < presser_arrival_s(
                            next(o for o in rest if o.track_id == second), target
                        )[0]
                    ):
                        second = op.track_id
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
                delta, _ = presser_arrival_s(nearest, target)
            outlets.append(
                OutletMatchup(
                    r.track_id,
                    nearest.track_id if nearest else None,
                    round(delta, 3),
                    True,
                    False,
                )
            )
    return MatchupResult(outlets=outlets, ball_presser_id=ball_presser.track_id)
