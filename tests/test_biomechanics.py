"""
Unit tests for biomechanical execution latency and presser compression.
"""

import math
from src.physics.biomechanics import (
    compute_hip_pivot_latency,
    evaluate_effective_press_closure,
    compute_biomechanical_execution_penalty,
    compute_touch_momentum_divergence,
    is_pass_inertia_locked,
    compute_projected_closing_velocity,
    MIN_PLANT_FOOT_DELAY_S,
)


def test_hip_pivot_latency():
    """Verifies hip rotation latency scales monotonically with turn angle."""
    body_angle = 0.0  # Facing +X

    # Open body (within 35 deg)
    t_open = compute_hip_pivot_latency(body_angle, math.radians(20.0))
    assert t_open == MIN_PLANT_FOOT_DELAY_S

    # Moderate turn (60 deg)
    t_mid = compute_hip_pivot_latency(body_angle, math.radians(60.0))
    assert MIN_PLANT_FOOT_DELAY_S < t_mid < 0.40

    # Extreme pivot (150 deg)
    t_closed = compute_hip_pivot_latency(body_angle, math.radians(150.0))
    assert t_closed > 0.50
    assert t_closed <= 0.70

    # Monotonicity test
    assert t_open <= t_mid <= t_closed


def test_effective_press_closure():
    """Verifies that longer prep latency allows presser to close down passing window."""
    initial_dist_m = 6.0
    presser_speed_ms = 7.0  # 7 m/s sprint

    # Open pass (0.15s delay): closes 1.05m -> 4.95m remaining
    d_open = evaluate_effective_press_closure(
        initial_dist_m, presser_speed_ms, prep_latency_s=0.15
    )
    assert abs(d_open - 4.95) < 0.05

    # Closed pivot (0.65s delay): closes 4.55m -> 1.45m remaining
    d_closed = evaluate_effective_press_closure(
        initial_dist_m, presser_speed_ms, prep_latency_s=0.65
    )
    assert abs(d_closed - 1.45) < 0.05

    # Extreme delay exceeds distance: clamped at 0.0m
    d_tackled = evaluate_effective_press_closure(
        initial_dist_m, presser_speed_ms, prep_latency_s=1.20
    )
    assert d_tackled == 0.0


def test_biomechanical_execution_penalty():
    """Verifies completion multiplier drops on sharp turns and weak foot."""
    body_angle = 0.0

    # Open pass strong foot
    p_open = compute_biomechanical_execution_penalty(
        body_angle, 0.0, is_weak_foot=False
    )
    assert p_open == 1.0

    # Closed pass strong foot
    p_closed = compute_biomechanical_execution_penalty(
        body_angle, math.radians(120.0), is_weak_foot=False
    )
    assert p_closed < 0.75

    # Weak foot penalty applied
    p_weak = compute_biomechanical_execution_penalty(body_angle, 0.0, is_weak_foot=True)
    assert abs(p_weak - 0.88) < 0.02
    assert p_weak < p_open


def test_touch_momentum_divergence():
    """Verifies momentum divergence between touch velocity vector and pass direction."""
    # Touch vector moving rightwards (+X direction at 1.5 m/s)
    touch_vel = (1.5, 0.0)

    # Pass in same direction (0.0 rad): zero divergence and zero delay
    div_aligned, delay_aligned = compute_touch_momentum_divergence(touch_vel, 0.0)
    assert div_aligned == 0.0
    assert delay_aligned == 0.0

    # Pass at 90 degrees (+Y direction): moderate delay
    div_90, delay_90 = compute_touch_momentum_divergence(touch_vel, math.pi / 2.0)
    assert abs(div_90 - math.pi / 2.0) < 0.01
    assert 0.15 <= delay_90 <= 0.25

    # Pass in opposite direction (-X direction, 180 degrees): maximum delay
    div_rev, delay_rev = compute_touch_momentum_divergence(touch_vel, math.pi)
    assert delay_rev > 0.40


def test_is_pass_inertia_locked():
    """Verifies that counter-momentum touch locks out passing options under press."""
    # Keeper took touch inward (+Y, toward center of pitch at 1.8 m/s)
    touch_vel = (0.0, 1.8)
    body_angle = math.pi / 2.0  # Facing +Y

    # Option A: In same direction as touch (+Y): NOT locked
    locked_fwd, total_fwd, cushion_fwd = is_pass_inertia_locked(
        touch_vel_xy_ms=touch_vel,
        target_pass_angle_rad=math.pi / 2.0,
        body_facing_angle_rad=body_angle,
        nearest_presser_dist_m=5.0,
        presser_closing_speed_ms=7.0,
    )
    assert locked_fwd is False
    assert cushion_fwd > 3.0

    # Option B: Reversing against touch (-Y, back to flank) under high press: LOCKED
    locked_rev, total_rev, cushion_rev = is_pass_inertia_locked(
        touch_vel_xy_ms=touch_vel,
        target_pass_angle_rad=-math.pi / 2.0,
        body_facing_angle_rad=body_angle,
        nearest_presser_dist_m=5.0,
        presser_closing_speed_ms=7.0,
    )
    assert locked_rev is True
    assert cushion_rev < 1.80


def test_compute_projected_closing_velocity():
    """Verifies vector projection of presser velocity along approach axis."""
    presser = (10.0, 30.0)
    target = (10.0, 35.0)  # Direct vertical approach (+Y)

    # Running straight at target (+Y at 7.0 m/s): full speed projected
    v_direct = (0.0, 7.0)
    speed_direct = compute_projected_closing_velocity(presser, v_direct, target)
    assert abs(speed_direct - 7.0) < 0.05

    # Running perpendicular (+X at 7.0 m/s): zero along approach -> bounded at min_floor (2.0 m/s)
    v_perp = (7.0, 0.0)
    speed_perp = compute_projected_closing_velocity(presser, v_perp, target)
    assert speed_perp == 2.0

    # Running opposite (-Y at 7.0 m/s, retreating): negative projection -> bounded at min_floor (2.0 m/s)
    v_opp = (0.0, -7.0)
    speed_opp = compute_projected_closing_velocity(presser, v_opp, target)
    assert speed_opp == 2.0
