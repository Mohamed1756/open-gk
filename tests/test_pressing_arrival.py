"""Golden + property tests for the arrival-time pressing core."""

from src.core.geometry import PitchPoint
from src.models.pressing.arrival import (
    arrival_margin,
    flight_time_s,
    lead_target,
    presser_arrival_s,
)
from src.models.pressing.types import PressingActor


def _actor(tid, x, y, vx=0.0, vy=0.0, team="opp", valid=True):
    return PressingActor(
        track_id=tid,
        team_id=team,
        pos_m=PitchPoint(x=x, y=y),
        vel_ms=(vx, vy),
        pitch_valid=valid,
    )


def test_flight_time_caps_at_1_2s():
    assert flight_time_s(19.0) == 1.0
    assert flight_time_s(100.0) == 1.2


def test_lead_target_projects_velocity():
    tgt = lead_target(PitchPoint(x=80.0, y=30.0), (4.0, 0.0), 1.0)
    assert tgt.x == 84.0 and tgt.y == 30.0


def test_converging_3v1_is_trapped():
    rec = _actor(9, 86.0, 25.0, team="own")
    passer = PitchPoint(x=100.0, y=29.0)
    pressers = [
        _actor(1, 87.5, 26.5, vx=-4.0, vy=-2.0),
        _actor(2, 84.5, 23.5, vx=3.0, vy=2.0),
        _actor(3, 87.5, 23.0, vx=-3.0, vy=3.0),
    ]
    res = arrival_margin(rec, pressers, passer)
    assert res.margin_s < 0.0
    assert res.best_presser_id in (1, 2, 3)


def test_static_pressers_leave_positive_margin():
    rec = _actor(9, 86.0, 25.0, team="own")
    passer = PitchPoint(x=100.0, y=29.0)
    pressers = [_actor(1, 80.0, 40.0), _actor(2, 95.0, 50.0)]
    res = arrival_margin(rec, pressers, passer)
    assert res.margin_s > 0.5


def test_invalid_opponents_ignored():
    rec = _actor(9, 86.0, 25.0, team="own")
    passer = PitchPoint(x=100.0, y=29.0)
    pressers = [_actor(1, 86.5, 25.5, valid=False)]
    res = arrival_margin(rec, pressers, passer)
    assert res.best_presser_id is None
    assert res.t_press_s == 99.0


def test_margin_falls_as_presser_accelerates():
    rec = _actor(9, 86.0, 25.0, team="own")
    passer = PitchPoint(x=100.0, y=29.0)
    slow = arrival_margin(rec, [_actor(1, 90.0, 29.0, vx=-1.0)], passer)
    fast = arrival_margin(rec, [_actor(1, 90.0, 29.0, vx=-6.0)], passer)
    assert fast.margin_s < slow.margin_s


def test_presser_running_away_does_not_count():
    away = presser_arrival_s(
        _actor(1, 88.0, 27.0, vx=6.0, vy=6.0), PitchPoint(x=86.0, y=25.0)
    )
    toward = presser_arrival_s(
        _actor(1, 88.0, 27.0, vx=-6.0, vy=-6.0), PitchPoint(x=86.0, y=25.0)
    )
    assert away[0] > toward[0]
