"""Unit tests for first-touch anchor search (pure, no mocks)."""

from src.models.distribution.first_touch import (
    find_first_touch,
    decision_possession_frames,
)


def test_finds_closest_approach_as_first_touch():
    ball = [(90.0, 30.0), (95.0, 30.0), (99.0, 29.0), (100.0, 29.0)]
    gk = [(100.0, 29.0)] * 4
    res = find_first_touch(ball, gk, decision_offset=0)
    assert res is not None
    assert res.frame_offset == 3
    assert res.ball_xy_m == (100.0, 29.0)
    assert res.gap_m == 0.0
    assert res.gap_at_decision_m == 10.05


def test_returns_none_without_overlap():
    assert find_first_touch([None, None], [(1.0, 1.0)] * 2, 0) is None
    assert find_first_touch([], [], 0) is None


def test_possession_window_expands_hold():
    gaps = [10.0, 5.0, 1.2, 0.8, 2.0, 8.0, 12.0]
    frames = decision_possession_frames(
        gaps, gap_threshold_m=3.0, pre_frames=1, post_frames=2
    )
    assert frames == [1, 2, 3, 4, 5, 6]


def test_possession_window_empty_without_contact():
    assert decision_possession_frames([9.0, None, 8.0]) == []


def test_gap_at_decision_none_when_missing_there():
    ball = [(90.0, 30.0), (99.5, 29.0)]
    gk = [None, (100.0, 29.0)]
    res = find_first_touch(ball, gk, decision_offset=0)
    assert res is not None
    assert res.frame_offset == 1
    assert res.gap_at_decision_m is None
