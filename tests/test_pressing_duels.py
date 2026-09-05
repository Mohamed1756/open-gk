"""Tests for duel timelines (hand-computed phases and sticky incumbency)."""

import pytest

from src.core.geometry import PitchPoint
from src.models.pressing.duels import (
    duel_frame_at,
    duel_phase,
    select_incumbent,
    track_duel,
)
from src.models.pressing.types import PressingActor


def _actor(tid, x, y, vx=0.0, vy=0.0, team="opp"):
    return PressingActor(
        track_id=tid, team_id=team, pos_m=PitchPoint(x=x, y=y), vel_ms=(vx, vy)
    )


def test_phase_boundaries_are_hand_exact():
    assert duel_phase(0.20) == "SHADOWING"
    assert duel_phase(0.199) == "CLOSING"
    assert duel_phase(0.0) == "CLOSING"
    assert duel_phase(-0.10) == "CONTESTED"
    assert duel_phase(-5.0) == "CONTESTED"


def test_sticky_incumbent_holds_ties_and_flicker():
    assert select_incumbent(None, []) is None
    assert select_incumbent(None, [(2, 1.0), (1, 0.5)]) == 1
    # Incumbent 2 at 0.6 vs challenger 1 at 0.5: gap 0.1 <= 0.15, keeps mark.
    assert select_incumbent(2, [(2, 0.6), (1, 0.5)]) == 2
    # Gap 0.2 > 0.15: loses it.
    assert select_incumbent(2, [(2, 0.7), (1, 0.5)]) == 1
    # Vanished incumbent falls through to best.
    assert select_incumbent(9, [(1, 0.5)]) == 1


def test_track_duel_rejects_misaligned_sequences():
    rec = [_actor(9, 86.0, 25.0, team="own")]
    with pytest.raises(ValueError):
        track_duel(9, 1, rec, [], [PitchPoint(x=100.0, y=29.0)], [0.0])


def test_distant_presser_shadows_adjacent_one_contests():
    ball = PitchPoint(x=100.0, y=29.0)
    rec = _actor(9, 86.0, 25.0, team="own")
    far = duel_frame_at(rec, _actor(1, 60.0, 29.0), ball, 0.0)
    assert far.margin_s > 0.20
    assert far.phase == "SHADOWING"
    assert far.presser_role == "DEEP"
    near = duel_frame_at(rec, _actor(2, 86.5, 25.5, vx=-4.0, vy=-2.0), ball, 0.04)
    assert near.margin_s < 0.0
    assert near.phase == "CONTESTED"


def test_track_duel_builds_aligned_timeline():
    ball = PitchPoint(x=100.0, y=29.0)
    rec = _actor(9, 86.0, 25.0, team="own")
    pre = _actor(1, 60.0, 29.0)
    track = track_duel(9, 1, [rec, rec], [pre, pre], [ball, ball], [0.0, 0.04])
    assert track.receiver_track_id == 9
    assert track.presser_track_id == 1
    assert [f.timestamp_s for f in track.frames] == [0.0, 0.04]
    assert all(f.phase == "SHADOWING" for f in track.frames)
