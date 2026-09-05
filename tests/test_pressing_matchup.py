"""Tests for presser-to-outlet assignment (hand-computed geometry)."""

from src.core.geometry import PitchPoint
from src.models.pressing.matchup import assign_pressers
from src.models.pressing.types import (
    DefenderTaskKind,
    PressingActor,
    PressingSnapshot,
)


def _actor(tid, x, y, vx=0.0, vy=0.0, team="opp"):
    return PressingActor(
        track_id=tid, team_id=team, pos_m=PitchPoint(x=x, y=y), vel_ms=(vx, vy)
    )


def _snap(receivers, opponents):
    return PressingSnapshot(
        passer=_actor(1, 100.0, 29.0, team="own"),
        receivers=receivers,
        opponents=opponents,
        ball_pos_m=PitchPoint(x=100.0, y=29.0),
    )


def test_two_pressers_three_outlets_leaves_free_man():
    receivers = [
        _actor(9, 86.0, 25.0, team="own"),
        _actor(5, 96.0, 48.0, team="own"),
        _actor(11, 98.0, 10.0, team="own"),
    ]
    opponents = [
        _actor(21, 99.0, 29.0, vx=-2.0, team="opp"),
        _actor(22, 88.0, 27.0, vx=-3.0, team="opp"),
    ]
    res = assign_pressers(_snap(receivers, opponents))
    assert res.ball_presser_id == 21
    by_id = {o.receiver_track_id: o for o in res.outlets}
    assert len(res.outlets) == 3
    frees = [o for o in res.outlets if o.is_free]
    assert len(frees) == 2
    assert by_id[9].presser_track_id == 22
    assert not by_id[9].is_free


def test_pincer_detected_on_synchronized_pair():
    receivers = [_actor(9, 86.0, 25.0, team="own")]
    opponents = [
        _actor(21, 99.0, 29.0, team="opp"),
        _actor(22, 88.0, 27.0, vx=-4.0, vy=-2.0),
        _actor(23, 84.0, 23.0, vx=4.0, vy=3.0),
    ]
    res = assign_pressers(_snap(receivers, opponents))
    assert len(res.outlets) == 1
    assert res.outlets[0].is_pincer
    assert res.outlets[0].second_presser_id is not None


def test_no_opponents_all_free():
    receivers = [_actor(9, 86.0, 25.0, team="own")]
    res = assign_pressers(_snap(receivers, []))
    assert res.outlets[0].is_free
    assert res.ball_presser_id is None


def test_deterministic_repeated_runs():
    receivers = [_actor(9, 86.0, 25.0, team="own"), _actor(5, 96.0, 48.0, team="own")]
    opponents = [_actor(21, 99.0, 29.0), _actor(22, 88.0, 27.0, vx=-3.0)]
    first = assign_pressers(_snap(receivers, opponents))
    second = assign_pressers(_snap(receivers, opponents))
    assert [(o.receiver_track_id, o.presser_track_id) for o in first.outlets] == [
        (o.receiver_track_id, o.presser_track_id) for o in second.outlets
    ]


def test_every_opponent_leaves_with_a_task():
    receivers = [_actor(9, 86.0, 25.0, team="own")]
    opponents = [
        _actor(21, 99.0, 29.0, team="opp"),  # on the ball: ball-press
        _actor(22, 88.0, 27.0, vx=-4.0, vy=-2.0),  # converging: mark or surplus
        _actor(23, 60.0, 29.0),  # 40m out, settled: deep, task NONE
    ]
    res = assign_pressers(_snap(receivers, opponents))
    assert set(res.defender_tasks) == {21, 22, 23}
    assert res.defender_tasks[21].kind is DefenderTaskKind.MAN
    assert res.defender_tasks[21].target_id == "ball"
    assert res.defender_tasks[22].kind is DefenderTaskKind.MAN
    deep = res.defender_tasks[23]
    assert deep.kind is DefenderTaskKind.NONE
    assert deep.reason == "deep-beyond-horizon"


def test_surplus_converger_gets_man_task():
    receivers = [_actor(9, 86.0, 25.0, team="own")]
    opponents = [
        _actor(21, 99.0, 29.0, team="opp"),
        _actor(22, 82.0, 24.0, vx=3.0, vy=1.0),
        _actor(23, 70.0, 20.0, vx=5.0, vy=2.0),
    ]
    res = assign_pressers(_snap(receivers, opponents))
    man_tasks = [
        t for t in res.defender_tasks.values() if t.kind is DefenderTaskKind.MAN
    ]
    # Ball-press plus mark plus the surplus converger: nobody vanishes.
    assert len(man_tasks) == 3
    surplus = [
        t for t in res.defender_tasks.values() if t.reason == "surplus-converger"
    ]
    assert len(surplus) >= 1


def test_split_stance_covers_two_lanes():
    receivers = [
        _actor(9, 88.0, 27.0, team="own"),
        _actor(5, 88.0, 31.0, team="own"),
    ]
    opponents = [
        _actor(21, 99.0, 29.0, team="opp"),
        _actor(24, 89.0, 27.5, team="opp"),
        _actor(25, 89.0, 30.5, team="opp"),
        _actor(22, 97.0, 29.0),  # settled where both corridors converge
    ]
    res = assign_pressers(_snap(receivers, opponents))
    task = res.defender_tasks[22]
    assert task.kind is DefenderTaskKind.LANE
    assert task.weight + task.second_weight == 1.0
    assert task.second_target_id is not None
