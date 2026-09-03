"""
Biomechanical Execution Cost and Physical Latency Models.
Calculates hip rotation latency, foot plant delay, and compressed presser
interception distance for goalkeeper distribution execution.
"""

from __future__ import annotations
import math

# Baseline biomechanical constants (Lees & Nolan 1998, Kicking Biomechanics in Soccer)
MIN_PLANT_FOOT_DELAY_S = 0.15  # Minimum time to plant non-kicking foot
MAX_HIP_PIVOT_DELAY_S = 0.70  # Full 180-degree pivot and strike delay
OPEN_HIP_THRESHOLD_RAD = math.radians(35.0)  # No swivel needed within 35 deg
RIGHT_ANGLE_HIP_RAD = math.radians(90.0)  # 90 deg half-turn


def wrap_angle_rad(angle_rad: float) -> float:
    """Wraps an angle into [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def compute_hip_pivot_latency(
    body_facing_angle_rad: float, target_pass_angle_rad: float
) -> float:
    """
    Computes physical execution latency (seconds) required for a goalkeeper
    to align hips, plant the standing foot, and strike the pass.
    """
    diff_rad = abs(wrap_angle_rad(target_pass_angle_rad - body_facing_angle_rad))

    if diff_rad <= OPEN_HIP_THRESHOLD_RAD:
        # Open body shape: only standing foot plant needed
        return MIN_PLANT_FOOT_DELAY_S

    if diff_rad <= RIGHT_ANGLE_HIP_RAD:
        # Moderate rotation (35 to 90 degrees)
        frac = (diff_rad - OPEN_HIP_THRESHOLD_RAD) / (
            RIGHT_ANGLE_HIP_RAD - OPEN_HIP_THRESHOLD_RAD
        )
        return round(MIN_PLANT_FOOT_DELAY_S + frac * 0.25, 3)

    # Extreme pivot (> 90 degrees)
    frac = min(1.0, (diff_rad - RIGHT_ANGLE_HIP_RAD) / (math.pi - RIGHT_ANGLE_HIP_RAD))
    return round(0.40 + frac * 0.30, 3)


def evaluate_effective_press_closure(
    initial_dist_m: float, presser_closing_speed_ms: float, prep_latency_s: float
) -> float:
    """
    Calculates the remaining distance (meters) between the goalkeeper and nearest
    presser at the exact instant the ball is struck, after accounting for execution latency.
    """
    closure_m = presser_closing_speed_ms * prep_latency_s
    effective_dist_m = max(0.0, initial_dist_m - closure_m)
    return round(effective_dist_m, 2)


def compute_biomechanical_execution_penalty(
    body_facing_angle_rad: float,
    target_pass_angle_rad: float,
    is_weak_foot: bool = False,
) -> float:
    """
    Calculates execution completion multiplier in [0.20, 1.0].
    Applies angle-dependent difficulty and weak-foot penalty.
    """
    diff_rad = abs(wrap_angle_rad(target_pass_angle_rad - body_facing_angle_rad))

    # Base difficulty curve from alignment
    if diff_rad <= OPEN_HIP_THRESHOLD_RAD:
        alignment_mult = 1.0
    elif diff_rad <= RIGHT_ANGLE_HIP_RAD:
        frac = (diff_rad - OPEN_HIP_THRESHOLD_RAD) / (
            RIGHT_ANGLE_HIP_RAD - OPEN_HIP_THRESHOLD_RAD
        )
        alignment_mult = 1.0 - frac * 0.25  # Drops from 1.0 to 0.75
    else:
        frac = min(
            1.0, (diff_rad - RIGHT_ANGLE_HIP_RAD) / (math.pi - RIGHT_ANGLE_HIP_RAD)
        )
        alignment_mult = 0.75 - frac * 0.40  # Drops from 0.75 to 0.35

    if is_weak_foot:
        alignment_mult *= 0.88  # 12% weak-foot variance penalty

    return round(max(0.20, alignment_mult), 3)


# First-touch momentum constants (Biemans et al. 2017, Deceleration & Directional Change in Football)
MIN_TOUCH_SPEED_FOR_INERTIA_MS = 0.50  # Below 0.5 m/s, touch is considered dead/static
MAX_TOUCH_INERTIA_DELAY_S = (
    0.45  # Maximum latency to brake, replant, and reverse ball momentum
)
INERTIA_LOCK_THRESHOLD_M = 1.80  # Minimum viable cushion before tackle is guaranteed


def compute_touch_momentum_divergence(
    touch_vel_xy_ms: tuple[float, float],
    target_pass_angle_rad: float,
) -> tuple[float, float]:
    """
    Calculates angular divergence (radians) between the first-touch ball displacement
    vector and the intended pass direction, returning (divergence_rad, inertia_delay_s).

    If touch speed < 0.5 m/s (dead ball), inertia delay is 0.0s.
    If pass is aligned with touch (divergence <= 45 deg), delay is 0.0s.
    If pass opposes touch (> 90 deg), requires deceleration, foot replant, and redirect delay.
    """
    vx, vy = touch_vel_xy_ms
    touch_speed = math.hypot(vx, vy)

    if touch_speed < MIN_TOUCH_SPEED_FOR_INERTIA_MS:
        return 0.0, 0.0

    touch_angle = math.atan2(vy, vx)
    divergence_rad = abs(wrap_angle_rad(target_pass_angle_rad - touch_angle))

    # Aligned with touch direction (within 45 deg)
    if divergence_rad <= math.radians(45.0):
        return round(divergence_rad, 3), 0.0

    # Moderate divergence (45 to 90 deg)
    if divergence_rad <= RIGHT_ANGLE_HIP_RAD:
        frac = (divergence_rad - math.radians(45.0)) / (
            RIGHT_ANGLE_HIP_RAD - math.radians(45.0)
        )
        delay_s = frac * 0.20
        return round(divergence_rad, 3), round(delay_s, 3)

    # Counter-momentum redirection (> 90 deg)
    frac = min(
        1.0, (divergence_rad - RIGHT_ANGLE_HIP_RAD) / (math.pi - RIGHT_ANGLE_HIP_RAD)
    )
    delay_s = 0.20 + frac * (MAX_TOUCH_INERTIA_DELAY_S - 0.20)
    return round(divergence_rad, 3), round(delay_s, 3)


def is_pass_inertia_locked(
    touch_vel_xy_ms: tuple[float, float],
    target_pass_angle_rad: float,
    body_facing_angle_rad: float,
    nearest_presser_dist_m: float,
    presser_closing_speed_ms: float,
) -> tuple[bool, float, float]:
    """
    Determines whether a passing option is physically locked out by the first touch momentum.
    Returns (is_locked, total_delay_s, effective_cushion_m).

    Total delay = hip rotation latency + touch momentum reversal delay.
    If effective cushion <= INERTIA_LOCK_THRESHOLD_M, option is locked out.
    """
    hip_latency = compute_hip_pivot_latency(
        body_facing_angle_rad, target_pass_angle_rad
    )
    _, inertia_delay = compute_touch_momentum_divergence(
        touch_vel_xy_ms, target_pass_angle_rad
    )
    total_delay = round(hip_latency + inertia_delay, 3)

    effective_cushion = evaluate_effective_press_closure(
        initial_dist_m=nearest_presser_dist_m,
        presser_closing_speed_ms=presser_closing_speed_ms,
        prep_latency_s=total_delay,
    )

    is_locked = (inertia_delay > 0.15) and (
        effective_cushion <= INERTIA_LOCK_THRESHOLD_M
    )
    return is_locked, total_delay, effective_cushion


def compute_projected_closing_velocity(
    presser_pos_m: tuple[float, float],
    presser_vel_xy_ms: tuple[float, float],
    target_pos_m: tuple[float, float],
    min_floor_speed_ms: float = 2.0,
) -> float:
    """
    Computes the scalar closing speed of the presser towards a target point
    via vector projection. If presser is running directly towards target,
    returns full speed. If running laterally or away, returns bounded floor.
    """
    dx = target_pos_m[0] - presser_pos_m[0]
    dy = target_pos_m[1] - presser_pos_m[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return min_floor_speed_ms

    ux = dx / dist
    uy = dy / dist

    vx, vy = presser_vel_xy_ms
    proj_speed = vx * ux + vy * uy
    return max(min_floor_speed_ms, round(proj_speed, 2))
