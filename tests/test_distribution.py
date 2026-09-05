"""
Unit tests for Goalkeeper Distribution Valuation Engine (Module M7).
"""

import pytest
from src.core.geometry import PitchPoint
from src.core.schemas import DistributionSituation
from src.core.types import Period, DistributionType, DistributionOutcome
from src.models.distribution.evaluator import DistributionEvaluator


def test_completion_probability_scaling():
    evaluator = DistributionEvaluator(press_danger_radius_m=6.0)

    # Short pass with no press
    xp_short = evaluator.estimate_completion_probability(
        pass_length_m=12.0,
        nearest_presser_dist_m=15.0,
        dist_type=DistributionType.SHORT_PASS,
    )
    assert xp_short > 0.85

    # Short pass under heavy immediate press (2m)
    xp_short_pressed = evaluator.estimate_completion_probability(
        pass_length_m=12.0,
        nearest_presser_dist_m=2.0,
        dist_type=DistributionType.SHORT_PASS,
    )
    assert xp_short_pressed < xp_short

    # Long launch (55m)
    xp_long = evaluator.estimate_completion_probability(
        pass_length_m=55.0,
        nearest_presser_dist_m=15.0,
        dist_type=DistributionType.LONG_PASS,
    )
    assert xp_long < xp_short
    assert 0.15 <= xp_long <= 0.60


def test_progression_and_turnover_risk():
    evaluator = DistributionEvaluator()

    # Progression from box (x=5) to midfield (x=45)
    prog = evaluator.compute_progression_threat(
        origin=PitchPoint(5.0, 34.0),
        target=PitchPoint(45.0, 34.0),
        opponents_pressed_count=2,
    )
    assert prog > 0.02

    # Dangerous central turnover near own goal
    cost_box = evaluator.compute_turnover_risk_cost(PitchPoint(12.0, 34.0))
    # Safe turnover near opponent corner
    cost_wide = evaluator.compute_turnover_risk_cost(PitchPoint(85.0, 60.0))

    assert cost_box > cost_wide
    assert cost_box > 0.30


def test_distribution_evaluation_and_aggregation():
    evaluator = DistributionEvaluator()

    sit_good = DistributionSituation(
        action_id="d1",
        match_id="m1",
        period=Period.FIRST_HALF,
        timestamp_seconds=400.0,
        gk_player_id="gk1",
        gk_player_name="Ederson",
        team_id="mcfc",
        pass_origin=PitchPoint(5.0, 34.0),
        pass_target=PitchPoint(35.0, 34.0),
        pass_length_m=30.0,
        distribution_type=DistributionType.SHORT_PASS,
        outcome=DistributionOutcome.SUCCESS_RETAINED,
        press_opponents_count=2,
        nearest_presser_dist_m=3.5,
    )

    sit_bad = DistributionSituation(
        action_id="d2",
        match_id="m1",
        period=Period.FIRST_HALF,
        timestamp_seconds=800.0,
        gk_player_id="gk1",
        gk_player_name="Ederson",
        team_id="mcfc",
        pass_origin=PitchPoint(5.0, 34.0),
        pass_target=PitchPoint(18.0, 34.0),
        pass_length_m=13.0,
        distribution_type=DistributionType.SHORT_PASS,
        outcome=DistributionOutcome.TURNOVER,
        press_opponents_count=1,
        nearest_presser_dist_m=2.0,
    )

    eval_good = evaluator.evaluate_distribution(sit_good)
    eval_bad = evaluator.evaluate_distribution(sit_bad)

    assert eval_good.distribution_value > 0.0
    assert eval_bad.distribution_value < 0.0

    summary = evaluator.aggregate_distribution_performance([eval_good, eval_bad])
    assert summary["total_distributions"] == 2
    assert summary["completion_rate"] == pytest.approx(0.5)
    assert "press_resistance_grade" in summary


def test_cover_shadow_direct_blocking():
    from src.core.geometry import compute_cover_shadow

    gk = PitchPoint(x=4.0, y=34.0)
    target = PitchPoint(x=20.0, y=34.0)

    # Defender directly on passing line
    opp_blocking = PitchPoint(x=12.0, y=34.0)
    shadow = compute_cover_shadow(gk, target, opp_blocking, defender_reach_m=1.8)
    assert shadow == pytest.approx(1.0, abs=1e-2)

    # Defender far away from passing line
    opp_far = PitchPoint(x=12.0, y=10.0)
    shadow_far = compute_cover_shadow(gk, target, opp_far, defender_reach_m=1.8)
    assert shadow_far == 0.0


def test_evaluate_all_options_structure():
    from src.core.tactical_shapes import (
        get_standard_buildup_teammates,
        create_high_433_press,
        create_mid_442_block,
        create_man_to_man_lock,
    )

    evaluator = DistributionEvaluator()
    gk_pos = PitchPoint(x=4.0, y=34.0)
    teammates = get_standard_buildup_teammates()
    press = create_high_433_press()

    evals = evaluator.evaluate_all_options(gk_pos, teammates, press.opponents)

    assert len(evals) == 10
    # Exactly one optimal pass
    optimal_count = sum(1 for e in evals if e.is_optimal_pass)
    assert optimal_count == 1

    # Probability bounds
    for e in evals:
        assert 0.0 <= e.xp_completion_prob <= 1.0
        assert e.progression_xt > 0.0
        assert e.turnover_risk_cost > 0.0
        assert -1.0 <= e.expected_value <= 1.0

    # Ensure presets instantiate correctly
    p2 = create_mid_442_block()
    p3 = create_man_to_man_lock()
    assert len(p2.opponents) == 10
    assert len(p3.opponents) == 10


def test_continuous_2d_spatial_turnover_hazard():
    from src.physics.gk_constraints import (
        CENTRAL_TURNOVER_PEAK_XG,
        TURNOVER_HAZARD_FLOOR_XG,
    )

    evaluator = DistributionEvaluator()

    # 1. Defending goal center (X=0, Y=34) when attacking left-to-right
    h_goal_mouth = evaluator.compute_turnover_risk_cost(
        PitchPoint(x=0.0, y=34.0), attack_dir_x=1.0
    )
    assert h_goal_mouth == pytest.approx(CENTRAL_TURNOVER_PEAK_XG, abs=1e-3)

    # 2. Penalty box edge centrally (X=16.5, Y=34.0)
    h_box_edge = evaluator.compute_turnover_risk_cost(
        PitchPoint(x=16.5, y=34.0), attack_dir_x=1.0
    )
    assert 0.30 <= h_box_edge <= 0.35

    # 3. Touchline at same depth (X=16.5, Y=0.0) -> heavily decayed by lateral Gaussian
    h_touchline = evaluator.compute_turnover_risk_cost(
        PitchPoint(x=16.5, y=0.0), attack_dir_x=1.0
    )
    assert h_touchline == pytest.approx(TURNOVER_HAZARD_FLOOR_XG, abs=1e-3)

    # 4. Inverted attack direction (attacking right-to-left, own goal at X=105.0)
    h_inverted_goal = evaluator.compute_turnover_risk_cost(
        PitchPoint(x=105.0, y=34.0), attack_dir_x=-1.0
    )
    assert h_inverted_goal == pytest.approx(CENTRAL_TURNOVER_PEAK_XG, abs=1e-3)

    h_inverted_box = evaluator.compute_turnover_risk_cost(
        PitchPoint(x=105.0 - 16.5, y=34.0), attack_dir_x=-1.0
    )
    assert 0.30 <= h_inverted_box <= 0.35


def test_exit_affordance_scaling():
    evaluator = DistributionEvaluator()
    origin = PitchPoint(x=5.0, y=34.0)
    target = PitchPoint(x=35.0, y=34.0)

    # Base threat with 0 exit lanes (pinned/constrained receiver)
    t_pinned = evaluator.compute_progression_threat(
        origin, target, opponents_pressed_count=1, exit_lanes_count=0
    )

    # Threat with 1 exit lane
    t_single = evaluator.compute_progression_threat(
        origin, target, opponents_pressed_count=1, exit_lanes_count=1
    )

    # Threat with 3 unblocked exit lanes
    t_multi = evaluator.compute_progression_threat(
        origin, target, opponents_pressed_count=1, exit_lanes_count=3
    )

    # Pinned should be discounted (0.75x) relative to single (1.00x) and multi (1.30x)
    assert t_pinned < t_single < t_multi
    assert t_single / t_pinned == pytest.approx(1.00 / 0.75, abs=1e-2)
    assert t_multi / t_single == pytest.approx(1.30 / 1.00, abs=1e-2)


def test_pressure_relief_value():
    from src.physics.gk_constraints import (
        GK_PRESS_URGENCY_RADIUS_M,
        GK_PRESS_CRITICAL_RADIUS_M,
    )

    evaluator = DistributionEvaluator()
    gk_pos = PitchPoint(x=5.0, y=34.0)
    fullback_pos = PitchPoint(x=25.0, y=55.0)

    origin_h = evaluator.compute_turnover_risk_cost(gk_pos, attack_dir_x=1.0)
    target_h = evaluator.compute_turnover_risk_cost(fullback_pos, attack_dir_x=1.0)

    # 1. Unpressed goalkeeper (> 8.0m): urgency is 0 -> relief is 0
    relief_unpressed = evaluator.compute_pressure_relief_value(
        origin_hazard=origin_h,
        target_hazard=target_h,
        gk_press_dist_m=GK_PRESS_URGENCY_RADIUS_M + 2.0,
    )
    assert relief_unpressed == 0.0

    # 2. Heavily pressed goalkeeper (at critical radius <= 2.0m): maximum urgency
    relief_critical = evaluator.compute_pressure_relief_value(
        origin_hazard=origin_h,
        target_hazard=target_h,
        gk_press_dist_m=GK_PRESS_CRITICAL_RADIUS_M,
    )
    assert relief_critical > 0.30
    assert relief_critical == pytest.approx(origin_h - target_h, abs=1e-3)

    # 3. Intermediate press (e.g. 5.0m closing at 3.0 m/s): scaled relief
    relief_inter = evaluator.compute_pressure_relief_value(
        origin_hazard=origin_h,
        target_hazard=target_h,
        gk_press_dist_m=5.0,
        closing_speed_ms=3.0,
    )
    assert 0.0 < relief_inter <= relief_critical

    # 4. Net EV with relief produces positive value for viable open pass
    prog_threat = evaluator.compute_progression_threat(
        origin=gk_pos,
        target=fullback_pos,
        opponents_pressed_count=1,
    )
    net_ev = evaluator.compute_net_distribution_ev(
        retention_xp=0.85,
        prog_threat=prog_threat,
        target_turnover_cost=target_h,
        relief_value=relief_inter,
    )
    assert net_ev > 0.0


def test_space_pass_and_off_screen_constants():
    from src.physics.gk_constraints import (
        OFF_SCREEN_DEFENDER_PRIOR_M,
        SPACE_PASS_LATERAL_OFFSET_M,
        RELEASE_EQUIVALENCE_DELTA_SCORE,
        MIN_RECOMMENDED_SCORE,
    )

    assert OFF_SCREEN_DEFENDER_PRIOR_M == 14.0
    assert SPACE_PASS_LATERAL_OFFSET_M == 2.0
    assert RELEASE_EQUIVALENCE_DELTA_SCORE == 8.0
    assert MIN_RECOMMENDED_SCORE == 50.0
