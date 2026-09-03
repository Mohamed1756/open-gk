"""
Unit tests for visual field of regard, line-of-sight occlusion, and spatial memory.
"""

import math
from src.core.geometry import PitchPoint
from src.models.perception.visual_field import (
    is_in_visual_cone,
    compute_los_occlusion,
    update_spatial_memory_buffer,
    evaluate_pressing_approach_cut,
    wrap_angle_rad,
    facing_to_ball,
    visible_person_ids,
    vision_logit_penalty,
    SCAN_KNOWN_THRESHOLD,
)


def test_angle_wrapping():
    """Verifies angle wrapping into [-pi, pi]."""
    assert abs(wrap_angle_rad(3.0 * math.pi) - math.pi) < 1e-5
    assert abs(wrap_angle_rad(-3.0 * math.pi) - (-math.pi)) < 1e-5
    assert abs(wrap_angle_rad(0.5)) == 0.5


def test_is_in_visual_cone():
    """Verifies 140-degree visual cone boundaries."""
    gk = PitchPoint(x=10.0, y=34.0)
    # Facing directly upfield (+X direction, angle = 0.0)
    angle_0 = 0.0

    # Target straight ahead (in cone)
    t_ahead = PitchPoint(x=25.0, y=34.0)
    in_cone, diff = is_in_visual_cone(gk, angle_0, t_ahead, fov_deg=140.0)
    assert in_cone is True
    assert diff == 0.0

    # Target at 45 degrees left (in cone)
    t_left_45 = PitchPoint(x=20.0, y=44.0)
    in_cone_45, diff_45 = is_in_visual_cone(gk, angle_0, t_left_45, fov_deg=140.0)
    assert in_cone_45 is True
    assert diff_45 < math.radians(70.0)

    # Target at 85 degrees (outside 140 deg cone, which cuts off at 70 deg)
    t_outside = PitchPoint(x=12.0, y=55.0)
    in_cone_out, diff_out = is_in_visual_cone(gk, angle_0, t_outside, fov_deg=140.0)
    assert in_cone_out is False
    assert diff_out > math.radians(70.0)

    # Target behind keeper (180 degrees, out of cone)
    t_behind = PitchPoint(x=5.0, y=34.0)
    in_cone_behind, _ = is_in_visual_cone(gk, angle_0, t_behind, fov_deg=140.0)
    assert in_cone_behind is False


def test_line_of_sight_occlusion():
    """Verifies line-of-sight raycasting detects foreground pressers."""
    gk = PitchPoint(x=10.0, y=34.0)
    target = PitchPoint(x=30.0, y=34.0)

    # Striker standing right in the passing lane
    striker_blocking = PitchPoint(x=15.0, y=34.2)
    is_blocked, occluder = compute_los_occlusion(
        gk, target, obstacles=[striker_blocking], obstacle_radius_m=0.75
    )
    assert is_blocked is True
    assert occluder is not None
    assert occluder.x == 15.0

    # Striker 5m away laterally (passing lane clear)
    striker_away = PitchPoint(x=15.0, y=39.0)
    is_blocked_away, _ = compute_los_occlusion(
        gk, target, obstacles=[striker_away], obstacle_radius_m=0.75
    )
    assert is_blocked_away is False


def test_spatial_memory_decay():
    """Verifies decaying spatial memory buffer retains recently scanned players."""
    timestamps = {}
    currently_visible = [1, 2]

    # t = 0.0s: IDs 1 and 2 visible
    ts_0, w_0 = update_spatial_memory_buffer(
        timestamps, currently_visible, current_time_s=0.0
    )
    assert w_0[1] == 1.0
    assert w_0[2] == 1.0

    # t = 1.0s: Only ID 1 visible, ID 2 looked away
    ts_1, w_1 = update_spatial_memory_buffer(
        ts_0, [1], current_time_s=1.0, decay_tau_s=2.0
    )
    assert w_1[1] == 1.0
    assert 0.50 < w_1[2] < 0.70  # exp(-1.0 / 2.0) ~ 0.606

    # t = 4.0s: ID 2 exceeded 3.5s memory horizon
    ts_4, w_4 = update_spatial_memory_buffer(
        ts_1, [1], current_time_s=4.0, memory_horizon_s=3.5
    )
    assert w_4[1] == 1.0
    assert w_4.get(2, 0.0) == 0.0  # Expired from active memory


def test_facing_follows_ball_with_fallback():
    assert facing_to_ball((100.0, 29.0), (95.0, 29.0), -1.571) == math.pi
    assert facing_to_ball((100.0, 29.0), None, -1.571) == -1.571
    assert facing_to_ball((100.0, 29.0), (100.0, 29.0), -1.571) == -1.571


def test_visible_person_ids_respects_cone():
    gk = (100.0, 29.0)
    persons = [(5, (104.2, 48.3)), (11, (105.0, 9.2))]
    visible = visible_person_ids(gk, -1.571, persons)
    assert 11 in visible
    assert 5 not in visible


def test_vision_logit_penalty_decays_with_memory():
    assert vision_logit_penalty(0.0) == 0.55
    assert vision_logit_penalty(1.0) == 0.0
    assert 0.0 < vision_logit_penalty(SCAN_KNOWN_THRESHOLD) < 0.55


def test_evaluate_pressing_approach_cut():
    """Verifies spatiotemporal arrival synchronization prevents ghost interceptions."""
    passer = PitchPoint(x=10.0, y=34.0)
    target = PitchPoint(x=25.0, y=14.0)  # Flank target

    # Case 1: Synchronized interception (ball and presser arrive together at ~0.6s)
    # Pass distance to intercept point is ~12.5m -> t_ball = 0.25s + (12.5 / 20.0) = 0.875s
    # Presser is at (17.5, 29.0) sprinting down towards lane (vy = -6.0 m/s) -> reaches y=24 in ~0.83s
    presser = PitchPoint(x=17.5, y=29.0)
    presser_vel_sync = (0.0, -6.0)

    is_cut_sync, min_dist_sync, dt_sync = evaluate_pressing_approach_cut(
        passer_pos=passer,
        target_pos=target,
        presser_pos=presser,
        presser_vel_xy_ms=presser_vel_sync,
        pass_speed_ms=20.0,
        prep_latency_s=0.25,
        time_horizon_s=1.5,
        corridor_width_m=1.5,
        interception_time_tolerance_s=0.35,
    )
    assert is_cut_sync is True
    assert min_dist_sync < 1.5
    assert dt_sync <= 0.35

    # Case 2: "Ghost" interception (crosses corridor physically, but 1.5s after ball has passed)
    # Presser is 20m away and slowly jogging (vy = -1.5 m/s)
    presser_slow = PitchPoint(x=17.5, y=40.0)
    presser_vel_slow = (0.0, -1.5)

    is_cut_ghost, min_dist_ghost, dt_ghost = evaluate_pressing_approach_cut(
        passer_pos=passer,
        target_pos=target,
        presser_pos=presser_slow,
        presser_vel_xy_ms=presser_vel_slow,
        pass_speed_ms=20.0,
        prep_latency_s=0.25,
        time_horizon_s=1.5,
        corridor_width_m=1.5,
    )
    # Physically crosses line, but ball is long gone: MUST NOT BE CUT
    assert is_cut_ghost is False

    # Case 3: Presser sprinting away from corridor
    presser_away = PitchPoint(x=18.0, y=26.0)
    presser_vel_away = (0.0, 6.0)
    is_cut_away, _, _ = evaluate_pressing_approach_cut(
        passer_pos=passer,
        target_pos=target,
        presser_pos=presser_away,
        presser_vel_xy_ms=presser_vel_away,
        time_horizon_s=1.5,
    )
    assert is_cut_away is False
