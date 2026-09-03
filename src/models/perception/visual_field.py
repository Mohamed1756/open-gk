"""
Visual Field and Perception Frustum Engine.
Models human gaze visual cones, line-of-sight occlusions,
and decaying spatial memory buffers for ego-aware decision modeling.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional

from src.core.geometry import PitchPoint


def wrap_angle_rad(angle_rad: float) -> float:
    """Wraps an angle into [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def is_in_visual_cone(
    observer_pos: PitchPoint,
    observer_angle_rad: float,
    target_pos: PitchPoint,
    fov_deg: float = 140.0,
) -> Tuple[bool, float]:
    """
    Evaluates whether a target is inside the observer's visual field of regard (FOV).
    Returns (is_in_cone, abs_angular_offset_rad).
    """
    dx = target_pos.x - observer_pos.x
    dy = target_pos.y - observer_pos.y
    dist = math.hypot(dx, dy)

    if dist < 1e-3:
        return True, 0.0

    target_angle = math.atan2(dy, dx)
    angle_diff = wrap_angle_rad(target_angle - observer_angle_rad)
    abs_diff = abs(angle_diff)

    half_fov_rad = math.radians(fov_deg / 2.0)
    is_in_cone = abs_diff <= half_fov_rad

    return is_in_cone, round(abs_diff, 4)


def compute_los_occlusion(
    observer_pos: PitchPoint,
    target_pos: PitchPoint,
    obstacles: List[PitchPoint],
    obstacle_radius_m: float = 0.75,
    max_check_dist_m: float = 12.0,
) -> Tuple[bool, Optional[PitchPoint]]:
    """
    Evaluates whether direct line-of-sight between observer and target is blocked
    by any obstacle (e.g. pressing striker).
    """
    dx = target_pos.x - observer_pos.x
    dy = target_pos.y - observer_pos.y
    ray_len = math.hypot(dx, dy)

    if ray_len < 1e-3:
        return False, None

    ux = dx / ray_len
    uy = dy / ray_len

    for obs in obstacles:
        vox_x = obs.x - observer_pos.x
        vox_y = obs.y - observer_pos.y
        dist_obs = math.hypot(vox_x, vox_y)

        # Only obstacles between observer and target, within check radius
        if dist_obs > min(ray_len, max_check_dist_m):
            continue

        # Projection along ray
        proj = vox_x * ux + vox_y * uy
        if proj <= 0.2:  # Behind or at observer
            continue

        # Perpendicular distance to ray
        perp_dist = math.sqrt(max(0.0, dist_obs * dist_obs - proj * proj))
        if perp_dist <= obstacle_radius_m:
            return True, obs

    return False, None


def update_spatial_memory_buffer(
    last_seen_timestamps_s: Dict[int, float],
    currently_visible_ids: List[int],
    current_time_s: float,
    decay_tau_s: float = 2.0,
    memory_horizon_s: float = 3.5,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Updates spatial memory buffer given current visual perception.
    Returns (updated_last_seen_timestamps, current_memory_weights).
    Weight = 1.0 when currently visible, decaying exponentially to 0.0 after memory_horizon.
    """
    updated_timestamps: Dict[int, float] = dict(last_seen_timestamps_s)
    memory_weights: Dict[int, float] = {}

    for tid in currently_visible_ids:
        updated_timestamps[tid] = current_time_s
        memory_weights[tid] = 1.0

    # Process remembered but not currently visible targets
    expired_ids = []
    for tid, last_seen in updated_timestamps.items():
        if tid in currently_visible_ids:
            continue

        elapsed_s = current_time_s - last_seen
        if elapsed_s > memory_horizon_s:
            expired_ids.append(tid)
            memory_weights[tid] = 0.0
        else:
            weight = math.exp(-elapsed_s / decay_tau_s)
            memory_weights[tid] = round(float(weight), 3)

    for tid in expired_ids:
        del updated_timestamps[tid]

    return updated_timestamps, memory_weights


SCAN_KNOWN_THRESHOLD = 0.5
SCAN_KNOWN_PENALTY = 10.0
OUT_OF_VISION_PENALTY = 20.0
VISION_LOGIT_BASE = 0.55


def facing_to_ball(
    observer_xy: Tuple[float, float],
    ball_xy: Optional[Tuple[float, float]],
    fallback_rad: float,
) -> float:
    if ball_xy is None:
        return fallback_rad
    dx = ball_xy[0] - observer_xy[0]
    dy = ball_xy[1] - observer_xy[1]
    if math.hypot(dx, dy) < 1e-3:
        return fallback_rad
    return math.atan2(dy, dx)


def visible_person_ids(
    observer_xy: Tuple[float, float],
    facing_rad: float,
    persons: List[Tuple[int, Tuple[float, float]]],
    fov_deg: float = 140.0,
) -> List[int]:
    observer = PitchPoint(x=observer_xy[0], y=observer_xy[1])
    visible = []
    for tid, xy in persons:
        in_cone, _ = is_in_visual_cone(
            observer, facing_rad, PitchPoint(x=xy[0], y=xy[1]), fov_deg=fov_deg
        )
        if in_cone:
            visible.append(tid)
    return visible


def vision_logit_penalty(memory_weight: float) -> float:
    return VISION_LOGIT_BASE * (1.0 - max(0.0, min(1.0, memory_weight)))


def evaluate_pressing_approach_cut(
    passer_pos: PitchPoint,
    target_pos: PitchPoint,
    presser_pos: PitchPoint,
    presser_vel_xy_ms: tuple[float, float],
    pass_speed_ms: float = 19.0,
    prep_latency_s: float = 0.25,
    time_horizon_s: float = 1.2,
    corridor_width_m: float = 1.5,
    interception_time_tolerance_s: float = 0.35,
) -> tuple[bool, float, float]:
    """
    Evaluates whether the pressing defender's running trajectory will cut the passing lane
    with spatiotemporal arrival synchronization.

    Calculates:
    1. Spatial proximity to passing corridor: perp_dist <= corridor_width_m.
    2. Temporal synchronization: ball arrival time t_ball vs presser arrival time t_presser.
       t_ball = prep_latency_s + (proj_m / pass_speed_ms)
       |t_ball - t_presser| <= interception_time_tolerance_s.

    Returns (is_cutting_lane, min_intercept_dist_m, min_time_delta_s).
    """
    vx, vy = presser_vel_xy_ms
    speed = math.hypot(vx, vy)
    if speed < 1.0:
        # Presser is stationary or jogging slowly; no dynamic interception
        return False, 99.0, 99.0

    # Ray from passer to target
    dx = target_pos.x - passer_pos.x
    dy = target_pos.y - passer_pos.y
    ray_len = math.hypot(dx, dy)
    if ray_len < 1e-3:
        return False, 99.0, 99.0

    ux = dx / ray_len
    uy = dy / ray_len

    min_dist = 99.0
    min_time_delta = 99.0
    is_cutting = False

    steps = 16
    dt = time_horizon_s / steps

    for step in range(steps + 1):
        t_presser = step * dt
        # Projected presser position at time t
        px = presser_pos.x + vx * t_presser
        py = presser_pos.y + vy * t_presser

        # Vector from passer to projected presser
        v_px = px - passer_pos.x
        v_py = py - passer_pos.y
        dist_from_passer = math.hypot(v_px, v_py)

        # Only evaluate if presser is between passer and target
        proj = v_px * ux + v_py * uy
        if 0.5 <= proj <= ray_len:
            perp_dist = math.sqrt(
                max(0.0, dist_from_passer * dist_from_passer - proj * proj)
            )
            # Spatiotemporal synchronization: time ball takes to reach this projection
            t_ball = prep_latency_s + (proj / max(5.0, pass_speed_ms))
            time_delta = abs(t_ball - t_presser)

            if perp_dist < min_dist:
                min_dist = perp_dist
            if time_delta < min_time_delta:
                min_time_delta = time_delta

            # True interception requires both spatial collision AND temporal arrival overlap
            if (
                perp_dist <= corridor_width_m
                and time_delta <= interception_time_tolerance_s
            ):
                is_cutting = True

    return is_cutting, round(min_dist, 2), round(min_time_delta, 3)
