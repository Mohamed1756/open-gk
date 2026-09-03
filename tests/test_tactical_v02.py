"""
Validation & Acceptance Tests for 2D Tactical Engine Specification v0.2 (Spec v0.2 §8).
Enforces Laws-of-the-Game pitch coordinates, Reach Envelope v2, Taxonomy, and Schema Contract.
"""

import pytest
from src.core.geometry import (
    PITCH_LENGTH_METERS,
    GOAL_Y_MIN,
    GOAL_Y_MAX,
    GOAL_Y_CENTER,
    PENALTY_BOX_LENGTH,
    SIX_YARD_BOX_LENGTH,
    compute_reach_envelope_v2,
)
from src.core.schemas import GKDecisionRecord
from src.core.types import CrossOutcome


def test_penalty_spot_and_laws_of_the_game_coordinates():
    """Tier 1: Asserts strict Laws-of-the-Game pitch coordinates (Spec v0.2 §8)."""
    # 1. Pitch length
    assert PITCH_LENGTH_METERS == 105.0

    # 2. Goal width: 7.32m (8 yards) centered at Y=34.0m
    goal_width = GOAL_Y_MAX - GOAL_Y_MIN
    assert goal_width == pytest.approx(7.32, abs=1e-3)
    assert GOAL_Y_CENTER == 34.0

    # 3. Penalty box depth: 16.5m (18 yards)
    assert PENALTY_BOX_LENGTH == 16.5

    # 4. Six yard box depth: 5.5m (6 yards)
    assert SIX_YARD_BOX_LENGTH == 5.5


def test_taxonomy_mutual_exclusivity():
    """Tier 1: Verifies punch is strictly distinct from clean claim (Spec v0.2 §5, §8)."""
    punch = CrossOutcome.PUNCH_CLEAR
    clean = CrossOutcome.CLAIM_CLEAN
    fumble = CrossOutcome.FUMBLE_DANGER
    miss = CrossOutcome.MISSED_CROSS

    # All outcome categories are mutually exclusive
    outcomes = [punch, clean, fumble, miss]
    assert len(set(outcomes)) == 4
    assert punch != clean


def test_reach_envelope_v2_acceleration_and_reaction_floor():
    """Tier 2: Validates Reach Envelope v2 kinematics (Spec v0.2 §4)."""
    t_react = 0.25
    v_burst = 4.2
    a_max = 3.5
    wingspan = 1.2

    # 1. Zero reach before reaction time floor
    r_pre_react = compute_reach_envelope_v2(
        flight_time_s=0.20,
        reaction_time_s=t_react,
        burst_speed_ms=v_burst,
        max_accel_mss=a_max,
        wingspan_reach_m=wingspan,
    )
    assert r_pre_react == 0.0

    # 2. Immediate post-reaction: quadratic acceleration phase (delta_t < tau)
    # tau = 4.2 / 3.5 = 1.2s. At t = 0.25 + 0.4 = 0.65s (delta_t = 0.4s)
    # dist = 0.5 * 3.5 * (0.4^2) = 0.28m -> r = 1.2 + 0.28 = 1.48m
    r_accel = compute_reach_envelope_v2(
        flight_time_s=0.65,
        reaction_time_s=t_react,
        burst_speed_ms=v_burst,
        max_accel_mss=a_max,
        wingspan_reach_m=wingspan,
    )
    assert r_accel == pytest.approx(1.48, abs=1e-2)

    # 3. Post-tau linear phase: delta_t = 2.0s > 1.2s
    # dist_accel = 0.5 * 3.5 * (1.2^2) = 2.52m
    # dist_linear = 4.2 * (2.0 - 1.2) = 3.36m
    # total dist = 5.88m -> r = 1.2 + 5.88 = 7.08m
    r_linear = compute_reach_envelope_v2(
        flight_time_s=2.25,
        reaction_time_s=t_react,
        burst_speed_ms=v_burst,
        max_accel_mss=a_max,
        wingspan_reach_m=wingspan,
    )
    assert r_linear == pytest.approx(7.08, abs=1e-2)


def test_schema_contract_and_interval_enforcement():
    """Tier 2: Validates versioned schema contract gk_decision_record v0.2.0 (Spec v0.2 §7)."""
    record = GKDecisionRecord(
        schema_version="0.2.0",
        action_id="act-1",
        match_id="m1",
        game_minute="18:48",
        timestamp_s=1128.0,
        keeper_id="gk-1",
        keeper_name="Jordan Pickford",
        team_name="England",
        decision_taken="COME",
        outcome="Claim Clean",
        decision_value=0.25,
        decision_value_interval=[0.13, 0.37],
        feasibility_prob=63.4,
        feasibility_interval=[45.0, 81.0],
        sensitivity_unstable=False,
    )

    # Asserts schema version and interval compliance
    assert record.schema_version == "0.2.0"
    assert len(record.decision_value_interval) == 2
    assert (
        record.decision_value_interval[0]
        < record.decision_value
        < record.decision_value_interval[1]
    )
    assert "dataset" in record.provenance
