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
