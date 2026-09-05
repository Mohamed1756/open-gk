"""Tests for duel resolution outcomes (hand-built ball/person sequences)."""

from src.models.distribution.outcomes import (
    DuelOutcome,
    detect_deflection,
    first_controller,
    release_frame_idx,
    resolve_duel_outcome,
    team_retained,
)

FPS = 25.0
OWN, OPP = "own", "opp"


def _pf(*frames):
    return [list(f) for f in frames]


def test_release_is_first_frame_outside_control_radius():
    assert release_frame_idx([0.1, 0.5, 1.3, 2.0], 0) == 2
    assert release_frame_idx([None, 0.5, 1.3], 0) == 2
    assert release_frame_idx([0.1, 0.5, 0.9], 0) is None
    assert release_frame_idx([None, None], 0) is None


def test_first_controller_picks_nearest():
    ball = [(10.0, 10.0)] * 3
    persons = _pf(
        [],
        [(1, OWN, 10.5, 10.0), (2, OWN, 10.2, 10.0)],
        [(9, OPP, 50.0, 50.0)],
    )
    assert first_controller(ball, persons, 0) == (1, 2, OWN)
    assert first_controller(ball, persons, 2) is None


def test_clean_retention():
    ball = [(50.0, 34.0)] * 80
    team = [(7, OWN, 50.0, 34.0)]
    persons = _pf(*([team] * 80))
    gaps = [0.1, 0.1] + [5.0] * 78
    res = resolve_duel_outcome(ball, gaps, persons, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.RETAINED
    assert res.controller_track_id == 7
    assert res.release_frame_idx == 2
    assert res.contested is False


def test_tackle_names_the_taker():
    ball = [(50.0, 34.0)] * 80
    frames = []
    for i in range(80):
        if i < 11:
            frames.append([(7, OWN, 50.0, 34.0), (3, OPP, 60.0, 34.0)])
        else:
            frames.append([(7, OWN, 60.0, 34.0), (3, OPP, 50.0, 34.0)])
    gaps = [0.1, 0.1] + [5.0] * 78
    res = resolve_duel_outcome(ball, gaps, frames, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.TACKLED
    assert res.controller_track_id == 3
    assert res.contested is True


def test_interception_by_opponent_first_touch():
    ball = [(50.0, 34.0)] * 80
    frames = [[(7, OWN, 60.0, 34.0), (3, OPP, 50.2, 34.0)]] * 80
    gaps = [0.1, 0.1] + [5.0] * 78
    res = resolve_duel_outcome(ball, gaps, frames, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.INTERCEPTED
    assert res.controller_track_id == 3


def test_contested_but_retained():
    ball = [(50.0, 34.0)] * 80
    frames = []
    for i in range(80):
        if i < 10:
            frames.append([(7, OWN, 50.0, 34.0), (3, OPP, 50.5, 34.0)])
        else:
            frames.append([(7, OWN, 50.0, 34.0), (3, OPP, 60.0, 34.0)])
    gaps = [0.1, 0.1] + [5.0] * 78
    res = resolve_duel_outcome(ball, gaps, frames, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.CONTESTED_RETAINED
    assert res.contested is True


def test_ball_out_of_play_after_control():
    ball = [(50.0, 34.0)] * 10 + [(106.0, 34.0)] * 70
    team = [(7, OWN, 50.0, 34.0)]
    persons = _pf(*([team] * 80))
    gaps = [0.1, 0.1] + [5.0] * 78
    res = resolve_duel_outcome(ball, gaps, persons, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.OUT_OF_PLAY


def test_truncated_clip_stays_unknown():
    ball = [(50.0, 34.0)] * 10
    persons = _pf(*([[(7, OWN, 50.0, 34.0)]] * 10))
    gaps = [0.1, 0.1] + [5.0] * 8
    res = resolve_duel_outcome(ball, gaps, persons, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.TRUNCATED_UNKNOWN


def test_no_release_when_ball_never_leaves():
    ball = [(50.0, 34.0)] * 10
    persons = _pf(*([[(7, OWN, 50.0, 34.0)]] * 10))
    res = resolve_duel_outcome(ball, [0.1] * 10, persons, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.NO_RELEASE


def test_mid_flight_deflection_is_blocked():
    # Synthetic unit-level speeds; geometry is what matters here.
    ball = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (3.0, 0.0), (2.0, 0.0), (1.5, 0.0)]
    persons = _pf(*([[(3, OPP, 3.5, 0.0)]] * 6))
    hit = detect_deflection(ball, persons, 1, 3, [OPP], fps=FPS)
    assert hit == (2, 3)
    gaps = [0.1, 2.0, 2.0, 2.0, 2.0, 2.0]
    res = resolve_duel_outcome(ball, gaps, persons, 0, OWN, [OPP], fps=FPS)
    assert res.outcome is DuelOutcome.BLOCKED


def test_retention_breaks_on_long_ball_gap():
    ball = [(50.0, 34.0)] * 5 + [None] * 4 + [(50.0, 34.0)] * 71
    persons = _pf(*([[(7, OWN, 50.0, 34.0)]] * 80))
    kept, _ = team_retained(ball, persons, OWN, 2, 75)
    assert kept is False
