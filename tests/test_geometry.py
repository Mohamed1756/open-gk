"""
Unit tests for pitch geometry, coordinate bounds, and spatial calculations.
"""

import pytest
from src.config import GOAL_Y_CENTER
from src.core.geometry import (
    PitchPoint,
    is_inside_penalty_box,
    is_inside_six_yard_box,
    calculate_distance_to_goal,
    calculate_goal_angle,
    calculate_optimal_position,
    compute_reach_envelope_radius,
)


def test_pitch_point_distance():
    p1 = PitchPoint(0.0, 0.0)
    p2 = PitchPoint(3.0, 4.0)
    assert p1.distance_to(p2) == pytest.approx(5.0)


def test_penalty_box_containment():
    # Inside defending penalty box
    assert is_inside_penalty_box(5.0, 34.0, defending_end=True) is True
    assert is_inside_penalty_box(16.5, 20.0, defending_end=True) is True

    # Outside defending penalty box (beyond 16.5m length or too wide)
    assert is_inside_penalty_box(20.0, 34.0, defending_end=True) is False
    assert is_inside_penalty_box(5.0, 10.0, defending_end=True) is False
    assert is_inside_penalty_box(5.0, 60.0, defending_end=True) is False

    # Attacking end penalty box
    assert is_inside_penalty_box(100.0, 34.0, defending_end=False) is True
    assert is_inside_penalty_box(50.0, 34.0, defending_end=False) is False


def test_six_yard_box_containment():
    assert is_inside_six_yard_box(3.0, 34.0, defending_end=True) is True
    assert is_inside_six_yard_box(10.0, 34.0, defending_end=True) is False


def test_distance_and_goal_angle():
    # Central shot from edge of box
    dist = calculate_distance_to_goal(16.5, GOAL_Y_CENTER, defending_end=True)
    assert dist == pytest.approx(16.5)

    angle = calculate_goal_angle(11.0, GOAL_Y_CENTER, defending_end=True)
    # At penalty spot (11m), visible angle is ~ 2 * atan((7.32/2) / 11) = 2 * atan(0.3327) ~= 0.64 rad (~36.8 deg)
    assert 0.5 < angle < 0.8


def test_optimal_position_bisector():
    ball_x = 20.0
    ball_y = 34.0
    opt_pos = calculate_optimal_position(
        ball_x, ball_y, distance_fraction=0.25, defending_end=True
    )
    assert opt_pos.y == pytest.approx(34.0)
    assert 0.0 < opt_pos.x < 20.0


def test_reach_envelope_radius():
    radius = compute_reach_envelope_radius(
        reaction_time=0.25, burst_speed=4.0, reach_wingspan=1.2
    )
    assert radius > 1.2
