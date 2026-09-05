"""
Unit tests for visual field of regard, line-of-sight occlusion, and spatial memory.
"""

import math
import pytest
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


def test_accumulate_window_scan_memory():
    """Verifies that scanning across multiple directions accumulates memory weights."""
    from src.models.perception.visual_field import accumulate_window_scan_memory

    obs_xy = (10.0, 34.0)
    # Player 1 is to the left: (15.0, 45.0) -> dy = +11, angle ~ +1.14 rad (~65 deg)
    # Player 2 is to the right: (15.0, 23.0) -> dy = -11, angle ~ -1.14 rad (~-65 deg)
    persons = [(1, (15.0, 45.0)), (2, (15.0, 23.0))]

    # Frame 1: t = 0.0s, keeper looks left (+1.1 rad) -> sees Player 1, Player 2 out of FOV
    # Frame 2: t = 0.5s, keeper turns head right (-1.1 rad) -> sees Player 2, Player 1 retained in memory
    scan_seq = [
        (0.0, 1.1, obs_xy, persons),
        (0.5, -1.1, obs_xy, persons),
    ]

    ts, weights = accumulate_window_scan_memory(scan_seq, decay_tau_s=2.0)
    # Both players should have positive memory weights!
    # Player 2 is currently visible at t=0.5 -> weight = 1.0
    assert weights[2] == 1.0
    # Player 1 was seen 0.5s ago -> weight = exp(-0.5/2.0) ~ 0.779
    assert 0.75 <= weights[1] <= 0.82


def test_dynamic_closing_press_accelerates_decay():
    """Verifies that high closing presser velocity accelerates memory decay on that outlet."""
    from src.models.perception.visual_field import update_spatial_memory_buffer

    # Target 1 has calm coverage; Target 2 has a presser charging at 5.0 m/s
    ts_0 = {1: 0.0, 2: 0.0}
    closing_speeds = {2: 5.0}  # 5.0 m/s sprint

    # At t = 1.0s, neither target is currently visible
    ts_1, weights = update_spatial_memory_buffer(
        ts_0,
        currently_visible_ids=[],
        current_time_s=1.0,
        decay_tau_s=2.0,
        closing_press_speeds=closing_speeds,
    )
    # Target 1 standard decay: exp(-1.0 / 2.0) ~ 0.606
    assert 0.58 <= weights[1] <= 0.63
    # Target 2 dynamic decay: tau_eff = 2.0 / (1 + 0.4*2.5) = 1.0s -> exp(-1.0 / 1.0) ~ 0.368
    assert weights[2] < weights[1] - 0.15
    assert 0.34 <= weights[2] <= 0.40


def test_los_occlusion_blocks_scan():
    """Verifies that a pressing forward directly in front blocks the scan ray to a teammate."""
    from src.models.perception.visual_field import visible_person_ids

    obs_xy = (10.0, 34.0)
    # Teammate straight ahead: (25.0, 34.0)
    persons = [(10, (25.0, 34.0))]

    # Facing straight ahead (0.0 rad)
    # Obstacle (striker) standing 2m ahead directly in line: (12.0, 34.0)
    striker = PitchPoint(x=12.0, y=34.0)
    visible = visible_person_ids(
        obs_xy,
        facing_rad=0.0,
        persons=persons,
        obstacles=[striker],
        max_obstacle_check_dist_m=3.5,
    )
    # Ray blocked by striker: teammate must NOT register as visible
    assert 10 not in visible

    # Without the striker, teammate is visible
    visible_clear = visible_person_ids(
        obs_xy,
        facing_rad=0.0,
        persons=persons,
        obstacles=[],
    )
    assert 10 in visible_clear


def test_confirmed_scan_zero_vision_penalty():
    """Verifies that a recently confirmed scan (weight >= 0.70) receives zero vision penalty."""
    from src.models.perception.visual_field import (
        vision_logit_penalty,
        SCAN_CONFIRMED_THRESHOLD,
    )

    # Scanned recently: weight = 0.85 >= 0.70
    assert vision_logit_penalty(0.85) == 0.0
    assert vision_logit_penalty(SCAN_CONFIRMED_THRESHOLD) == 0.0

    # Stale scan: weight = 0.40 < 0.70
    assert vision_logit_penalty(0.40) > 0.0
    # Completely unscanned: weight = 0.0
    assert vision_logit_penalty(0.0) == pytest.approx(0.55)


def test_disguised_and_scanned_outlet_valuation():
    """
    Verifies that outlets outside current gaze with recent confirmed scans (weight >= 0.70)
    are classified as DISGUISED_OUTLET (when torso angle differs > 45 deg) or SCANNED_OPEN,
    both incurring zero vision penalty.
    """
    from scripts.evaluate_distribution import evaluate_distribution_decision
    from src.models.distribution.evaluator import DistributionEvaluator

    evaluator = DistributionEvaluator()
    gk_pos = PitchPoint(x=10.0, y=34.0)

    # Receiver 1: Central midfielder ahead at (25.0, 34.0) -> pass angle = 0.0 rad
    # Receiver 2: Fullback upfield at (20.0, 50.0)
    receivers = [
        {
            "track_id": 1,
            "team_label": "midfielder",
            "pitch_xy": [25.0, 34.0],
            "pitch_vel_ms": [0.0, 0.0],
        },
        {
            "track_id": 2,
            "team_label": "fullback",
            "pitch_xy": [20.0, 50.0],
            "pitch_vel_ms": [0.0, 0.0],
        },
    ]
    opponents = [PitchPoint(x=40.0, y=34.0)]  # Well clear, > 14m away

    # Gaze is looking toward fullback at +1.4 rad (~80 deg) -> Midfielder at 0 rad is outside 140-deg cone
    # Case A: Torso also facing +1.4 rad (shaped for fullback). Diff to pass (0 rad) is ~80 deg (> 45 deg)
    # Memory weight for Midfielder = 0.85 (recently confirmed scan)
    evals_disguised = evaluate_distribution_decision(
        passer_pos=gk_pos,
        receiver_candidates=receivers,
        opponents=opponents,
        evaluator=evaluator,
        passer_facing_angle_rad=1.4,
        passer_gaze_angle_rad=1.4,
        memory_weights={1: 0.85, 2: 1.0},
    )
    mid_eval_a = next(o for o in evals_disguised if o["track_id"] == 1)
    assert mid_eval_a["path_status"] == "DISGUISED_OUTLET"
    assert mid_eval_a["grade_color"] == "#38bdf8"
    assert mid_eval_a["in_visual_cone"] is False

    # Case B: Gaze is looking at fullback (+1.4 rad), but Torso is square to Midfielder (0.0 rad). Diff <= 45 deg.
    evals_scanned_open = evaluate_distribution_decision(
        passer_pos=gk_pos,
        receiver_candidates=receivers,
        opponents=opponents,
        evaluator=evaluator,
        passer_facing_angle_rad=0.0,
        passer_gaze_angle_rad=1.4,
        memory_weights={1: 0.85, 2: 1.0},
    )
    mid_eval_b = next(o for o in evals_scanned_open if o["track_id"] == 1)
    assert mid_eval_b["path_status"] == "SCANNED_OPEN"
    assert mid_eval_b["grade_color"] == "#22c55e"
    assert mid_eval_b["in_visual_cone"] is False

    # Case C: Stale scan: weight = 0.55 (>= 0.50, < 0.70)
    evals_stale = evaluate_distribution_decision(
        passer_pos=gk_pos,
        receiver_candidates=receivers,
        opponents=opponents,
        evaluator=evaluator,
        passer_facing_angle_rad=1.4,
        passer_gaze_angle_rad=1.4,
        memory_weights={1: 0.55, 2: 1.0},
    )
    mid_eval_c = next(o for o in evals_stale if o["track_id"] == 1)
    assert mid_eval_c["path_status"] == "KNOWN_FROM_SCAN"
    assert mid_eval_c["grade_color"] == "#2dd4bf"

    # Case D: Unscanned: weight = 0.20 (< 0.50)
    evals_blind = evaluate_distribution_decision(
        passer_pos=gk_pos,
        receiver_candidates=receivers,
        opponents=opponents,
        evaluator=evaluator,
        passer_facing_angle_rad=1.4,
        passer_gaze_angle_rad=1.4,
        memory_weights={1: 0.20, 2: 1.0},
    )
    mid_eval_d = next(o for o in evals_blind if o["track_id"] == 1)
    assert mid_eval_d["path_status"] == "OUT_OF_VISION"
    assert mid_eval_d["grade_color"] == "#94a3b8"


def test_merge_swept_cones_keeps_gap():
    """Union must not claim the unscanned gap between two narrow looks."""
    from src.models.perception.visual_field import merge_swept_cones

    union = merge_swept_cones([2.5, -2.5], math.radians(20.0))
    assert len(union) == 2

    overlapping = merge_swept_cones([1.1, -1.1], math.radians(70.0))
    assert len(overlapping) == 1


def test_scan_windows_track_first_and_last_seen():
    """Windows record per-outlet first/last sightings from measured frames."""
    from src.models.perception.visual_field import (
        accumulate_window_scan_windows,
        summarize_scan_windows,
    )

    obs_xy = (10.0, 34.0)
    persons = [(1, (15.0, 45.0)), (2, (15.0, 23.0))]
    seq = [(0.0, 1.1, obs_xy, persons), (0.5, -1.1, obs_xy, persons)]
    first, last, weights = accumulate_window_scan_windows(seq, decay_tau_s=2.0)
    assert first[1] == 0.0
    assert last[2] == 0.5
    windows = summarize_scan_windows(first, last, weights, 0.5)
    assert windows[2]["status"] == "CONFIRMED"
    assert windows[1]["status"] == "CONFIRMED"
    assert windows[1]["age_s"] == pytest.approx(0.5)


def test_low_gaze_confidence_discounts_visible_weight():
    """A noisy glimpse must not mint a full-confidence memory."""
    from src.models.perception.visual_field import update_spatial_memory_buffer

    _, weights = update_spatial_memory_buffer(
        {}, [1], 0.0, visible_confidences={1: 0.3}
    )
    assert weights[1] == pytest.approx(0.3)


def test_decay_scales_from_discounted_weight():
    """Decay after a noisy glimpse scales from 0.3, never above it."""
    from src.models.perception.visual_field import update_spatial_memory_buffer

    _, w0 = update_spatial_memory_buffer({}, [1], 0.0, visible_confidences={1: 0.3})
    _, w1 = update_spatial_memory_buffer(
        {1: 0.0}, [], 1.0, decay_tau_s=2.0, prior_weights=w0
    )
    assert w1[1] == pytest.approx(0.3 * math.exp(-0.5), abs=1e-3)
    assert w1[1] < 0.3


def test_accumulate_discounts_noisy_frames():
    """Occluded-head frames (gaze == torso, conf ~0.1) barely register."""
    from src.models.perception.visual_field import accumulate_window_scan_memory

    obs_xy = (10.0, 34.0)
    persons = [(1, (15.0, 34.0))]
    seq = [(0.0, 0.0, obs_xy, persons), (0.5, 0.0, obs_xy, persons)]
    _, weights = accumulate_window_scan_memory(
        seq, decay_tau_s=2.0, gaze_confidences=[0.1, 0.1]
    )
    assert weights[1] == pytest.approx(0.1)


def test_chipped_pass_parabolic_clearance():
    from src.models.perception.visual_field import evaluate_chipped_clearance

    passer = PitchPoint(x=10.0, y=34.0)
    receiver = PitchPoint(x=35.0, y=34.0)  # 25m pass

    # Intermediate obstacle (striker) at 7m downfield (s = 7 / 25 = 0.28)
    # Apex = 3.5m -> z(0.28) = 4 * 3.5 * 0.28 * 0.72 = 2.82m > 2.45m -> cleared!
    striker = PitchPoint(x=17.0, y=34.0)
    is_cleared, clearance_h = evaluate_chipped_clearance(passer, receiver, striker)
    assert is_cleared is True
    assert clearance_h > 2.45

    # Close obstacle at receiver's boots (24m downfield, s = 24 / 25 = 0.96)
    # Beyond intermediate zone (s > 0.85) -> descending ball cannot clear a close marker!
    close_defender = PitchPoint(x=34.0, y=34.0)
    is_cleared_close, _ = evaluate_chipped_clearance(passer, receiver, close_defender)
    assert is_cleared_close is False
