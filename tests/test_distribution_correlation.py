"""Tests for outlet independence (hand-computed angles and ratios)."""

import math

import pytest

from src.models.distribution.correlation import (
    angular_gap_rad,
    correlation_threshold_rad,
    find_shared_fate,
    frame_correlated,
    shared_fate_ratio,
)


def test_angular_gap_is_hand_exact():
    assert angular_gap_rad((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)) == pytest.approx(
        math.pi / 4.0
    )
    assert angular_gap_rad((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)) == pytest.approx(0.0)
    assert angular_gap_rad((5.0, 5.0), (5.0, 5.0), (9.0, 5.0)) == pytest.approx(0.0)


def test_threshold_shrinks_with_distance():
    assert correlation_threshold_rad(10.0) == pytest.approx(0.24)
    assert correlation_threshold_rad(20.0) == pytest.approx(0.12)
    assert correlation_threshold_rad(10.0) > correlation_threshold_rad(40.0)


def test_frame_needs_angle_and_shared_threat():
    passer = (100.0, 29.0)
    near_a, near_b = (86.0, 28.0), (86.0, 30.0)
    assert frame_correlated(passer, near_a, near_b, 7, 7) is True
    assert frame_correlated(passer, near_a, near_b, 7, 8) is False
    assert frame_correlated(passer, near_a, near_b, None, 7) is False
    far_b = (70.0, 50.0)
    assert frame_correlated(passer, near_a, far_b, 7, 7) is False


def test_shared_fate_ratio_counts_hits():
    passer = (100.0, 29.0)
    leads = [(86.0, 28.0)] * 4
    ratio = shared_fate_ratio(passer, leads, leads, [7, 7, 7, 9], [7, 7, 8, 8])
    assert ratio == pytest.approx(0.5)
    with pytest.raises(ValueError):
        shared_fate_ratio(passer, leads, leads[:2], [7] * 4, [7] * 4)
    with pytest.raises(ValueError):
        shared_fate_ratio(passer, [], [], [], [])


def test_find_shared_fate_majority_rule():
    passer = (100.0, 29.0)
    leads = {
        9: [(86.0, 28.0)] * 4,
        5: [(86.0, 30.0)] * 4,
        2: [(70.0, 50.0)] * 4,
    }
    pressers = {
        9: [7, 7, 7, 7],
        5: [7, 7, 7, 8],
        2: [3, 3, 3, 3],
    }
    shared = find_shared_fate([9, 5, 2], passer, leads, pressers)
    assert shared[9] == [5]
    assert shared[5] == [9]
    assert shared[2] == []
