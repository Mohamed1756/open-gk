"""
Geometric utilities and spatial pitch models for goalkeeper positioning and actions.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from src.config import (
    PITCH_LENGTH_METERS,
    GOAL_Y_MIN,
    GOAL_Y_MAX,
    GOAL_Y_CENTER,
    PENALTY_BOX_LENGTH,
    PENALTY_BOX_Y_MIN,
    PENALTY_BOX_Y_MAX,
    SIX_YARD_BOX_LENGTH,
    SIX_YARD_BOX_Y_MIN,
    SIX_YARD_BOX_Y_MAX,
)
from src.physics.gk_constraints import (
    GK_REACTION_TIME_DEFAULT_S,
    GK_LATERAL_BURST_SPEED_MAX_MS,
    GK_WINGSPAN_REACH_DEFAULT_M,
    GK_MAX_BURST_ACCELERATION_MSS,
)


@dataclass(frozen=True)
class PitchPoint:
    """A 2D coordinate on the normalized pitch (meters)."""

    x: float
    y: float

    def distance_to(self, other: PitchPoint) -> float:
        """Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)


def is_inside_penalty_box(x: float, y: float, defending_end: bool = True) -> bool:
    """
    Check whether a coordinate (x, y) is inside the 18-yard penalty box.
    If defending_end is True, checks x in [0, 16.5].
    If defending_end is False, checks x in [105 - 16.5, 105].
    """
    if defending_end:
        x_in = 0.0 <= x <= PENALTY_BOX_LENGTH
    else:
        x_in = (PITCH_LENGTH_METERS - PENALTY_BOX_LENGTH) <= x <= PITCH_LENGTH_METERS

    y_in = PENALTY_BOX_Y_MIN <= y <= PENALTY_BOX_Y_MAX
    return bool(x_in and y_in)


def is_inside_six_yard_box(x: float, y: float, defending_end: bool = True) -> bool:
    """
    Check whether a coordinate (x, y) is inside the 6-yard goal area.
    """
    if defending_end:
        x_in = 0.0 <= x <= SIX_YARD_BOX_LENGTH
    else:
        x_in = (PITCH_LENGTH_METERS - SIX_YARD_BOX_LENGTH) <= x <= PITCH_LENGTH_METERS

    y_in = SIX_YARD_BOX_Y_MIN <= y <= SIX_YARD_BOX_Y_MAX
    return bool(x_in and y_in)


def calculate_distance_to_goal(x: float, y: float, defending_end: bool = True) -> float:
    """Calculate Euclidean distance to the center of the specified goal."""
    target_x = 0.0 if defending_end else PITCH_LENGTH_METERS
    target_y = GOAL_Y_CENTER
    return math.hypot(x - target_x, y - target_y)


def calculate_goal_angle(x: float, y: float, defending_end: bool = True) -> float:
    """
    Calculate the visible angle (in radians) subtended by the goal posts from (x, y).
    Uses the law of cosines / vector dot products.
    """
    goal_x = 0.0 if defending_end else PITCH_LENGTH_METERS
    post1 = (goal_x, GOAL_Y_MIN)
    post2 = (goal_x, GOAL_Y_MAX)

    v1 = np.array([post1[0] - x, post1[1] - y])
    v2 = np.array([post2[0] - x, post2[1] - y])

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return math.pi

    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def calculate_optimal_position(
    ball_x: float,
    ball_y: float,
    distance_fraction: float = 0.25,
    defending_end: bool = True,
) -> PitchPoint:
    """
    Compute the geometric bisector position between the ball and the defending goal posts.
    distance_fraction: how far out from the goal line the GK steps along the bisector (e.g. 25% towards ball, capped).
    """
    goal_center_x = 0.0 if defending_end else PITCH_LENGTH_METERS
    goal_center_y = GOAL_Y_CENTER

    # Vector from goal center to ball
    dx = ball_x - goal_center_x
    dy = ball_y - goal_center_y
    dist = math.hypot(dx, dy)

    if dist < 1e-3:
        return PitchPoint(goal_center_x, goal_center_y)

    # Step out towards the ball, capped at penalty box boundary
    step_out = min(dist * distance_fraction, PENALTY_BOX_LENGTH * 0.8)
    gk_x = goal_center_x + (dx / dist) * step_out
    gk_y = goal_center_y + (dy / dist) * step_out

    return PitchPoint(x=gk_x, y=gk_y)


def compute_reach_envelope_radius(
    reaction_time: float = GK_REACTION_TIME_DEFAULT_S,
    burst_speed: float = GK_LATERAL_BURST_SPEED_MAX_MS,
    reach_wingspan: float = GK_WINGSPAN_REACH_DEFAULT_M,
) -> float:
    """
    Calculate dynamic reach envelope radius (meters) for a goalkeeper.
    burst_speed: max lateral acceleration/sprint speed (m/s)
    reaction_time: time floor before movement starts (seconds)
    reach_wingspan: diving extension length beyond body center (meters)
    """
    return reach_wingspan + max(0.0, burst_speed * (0.5 - reaction_time))


def compute_reach_envelope_v2(
    flight_time_s: float,
    reaction_time_s: float = GK_REACTION_TIME_DEFAULT_S,
    burst_speed_ms: float = GK_LATERAL_BURST_SPEED_MAX_MS,
    max_accel_mss: float = GK_MAX_BURST_ACCELERATION_MSS,
    wingspan_reach_m: float = GK_WINGSPAN_REACH_DEFAULT_M,
) -> float:
    """
    Computes reach radius r(t) under Reach Envelope v2 (Spec v0.2 §4).
    Enforces reaction floor (r=0 before reaction), acceleration limits, and quadratic-to-linear kinematics.
    """
    if flight_time_s < reaction_time_s:
        return 0.0

    delta_t = flight_time_s - reaction_time_s
    tau = burst_speed_ms / max(max_accel_mss, 0.1)

    if delta_t <= tau:
        distance = 0.5 * max_accel_mss * (delta_t**2)
    else:
        dist_accel = 0.5 * max_accel_mss * (tau**2)
        dist_linear = burst_speed_ms * (delta_t - tau)
        distance = dist_accel + dist_linear

    return wingspan_reach_m + distance


def compute_cover_shadow(
    origin: PitchPoint,
    target: PitchPoint,
    defender: PitchPoint,
    defender_reach_m: float = 1.8,
) -> float:
    """
    Computes cover shadow obstruction of a defender along the passing vector (origin -> target).
    Returns 0.0 (no obstruction) to 1.0 (direct blocking).
    """
    dx = target.x - origin.x
    dy = target.y - origin.y
    pass_len = math.hypot(dx, dy)

    if pass_len < 1e-4:
        return 0.0

    # Unit vector along pass
    ux = dx / pass_len
    uy = dy / pass_len

    # Vector from origin to defender
    fx = defender.x - origin.x
    fy = defender.y - origin.y

    # Projection along pass line
    proj = fx * ux + fy * uy

    # If defender is behind origin or beyond target, minimal shadow
    if proj < 0.5 or proj > pass_len:
        return 0.0

    # Perpendicular distance to pass ray
    perp_dist = abs(fx * uy - fy * ux)

    if perp_dist <= defender_reach_m:
        return max(0.0, 1.0 - (perp_dist / defender_reach_m))

    return 0.0
