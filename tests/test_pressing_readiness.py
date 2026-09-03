"""Tests for receiver readiness body model (hand-verified angles)."""

import math

from src.core.geometry import PitchPoint
from src.models.pressing.readiness import (
    receiver_readiness,
    resolve_facing,
)
from src.models.pressing.types import PressingActor


def _actor(tid, x, y, facing=None, source="pose", vx=0.0, vy=0.0, team="own"):
    return PressingActor(
        track_id=tid,
        team_id=team,
        pos_m=PitchPoint(x=x, y=y),
        vel_ms=(vx, vy),
        facing_rad=facing,
        facing_source=source,
    )


def test_open_cb_ready_now():
    rec = _actor(4, 80.0, 30.0, facing=0.0)
    res = receiver_readiness(rec, PitchPoint(x=100.0, y=29.0))
    assert res.turn_latency_s == 0.15
    assert res.readiness_mult == 1.0
    assert res.facing_source == "pose"
    assert res.note == "open"


def test_back_to_passer_prices_turn_not_veto():
    rec = _actor(4, 80.0, 29.0, facing=math.pi)
    res = receiver_readiness(rec, PitchPoint(x=100.0, y=29.0))
    assert res.turn_latency_s == 0.70
    assert res.readiness_mult == 0.35
    assert "needs-turn" in res.note


def test_back_to_presser_shields():
    rec = _actor(9, 86.0, 25.0, facing=-3.0 * math.pi / 4.0)
    presser = _actor(21, 88.0, 27.0, team="opp")
    res = receiver_readiness(rec, PitchPoint(x=100.0, y=29.0), nearest_presser=presser)
    assert "shielded" in res.note
    assert res.readiness_mult > 0.35


def test_chest_open_to_closing_presser_exposed():
    rec = _actor(9, 86.0, 25.0, facing=0.0)
    presser = _actor(21, 90.0, 25.0, team="opp")
    res = receiver_readiness(
        rec,
        PitchPoint(x=100.0, y=29.0),
        nearest_presser=presser,
        presser_closing=True,
    )
    assert "exposed" in res.note
    assert res.readiness_mult == round(1.0 * 0.80, 3)


def test_facing_fallback_chain():
    moving = _actor(7, 80.0, 30.0, facing=None, vx=3.0, vy=0.0)
    assert resolve_facing(moving)[1] == "velocity"
    static = _actor(7, 80.0, 30.0, facing=None)
    angle, source = resolve_facing(static, PitchPoint(x=100.0, y=30.0))
    assert source == "ball"
    assert angle == 0.0
    lost = _actor(7, 80.0, 30.0, facing=None)
    res = receiver_readiness(lost, PitchPoint(x=80.0, y=10.0))
    assert res.facing_source == "unknown"
    assert res.readiness_mult == round(1.0 * 0.85, 3)
