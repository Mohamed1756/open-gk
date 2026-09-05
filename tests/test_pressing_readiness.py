"""Tests for receiver readiness body model (hand-verified angles)."""

import math
import pytest

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


def test_reception_latency_and_post_cushion():
    from src.physics.gk_constraints import (
        MIN_POST_RECEIPT_CUSHION_S,
        RECEIVER_FIRST_TOUCH_LATENCY_S,
    )

    rec = _actor(4, 80.0, 30.0, facing=0.0)
    res = receiver_readiness(rec, PitchPoint(x=100.0, y=29.0))
    expected_reception = round(RECEIVER_FIRST_TOUCH_LATENCY_S + res.turn_latency_s, 3)
    assert res.reception_latency_s == expected_reception

    # Margin shorter than reception commitment yields negative or sub-threshold post-cushion
    short_margin_s = expected_reception + 0.10
    assert res.post_cushion_s(short_margin_s) == pytest.approx(0.10, abs=1e-3)
    assert res.post_cushion_s(short_margin_s) < MIN_POST_RECEIPT_CUSHION_S

    # Generous margin leaves ample control cushion
    wide_margin_s = expected_reception + 0.80
    assert res.post_cushion_s(wide_margin_s) == pytest.approx(0.80, abs=1e-3)
    assert res.post_cushion_s(wide_margin_s) >= MIN_POST_RECEIPT_CUSHION_S


def test_hospital_pass_classification():
    from scripts.evaluate_distribution import evaluate_distribution_decision
    from src.models.distribution.evaluator import DistributionEvaluator

    evaluator = DistributionEvaluator()
    gk_pos = PitchPoint(x=10.0, y=34.0)

    # Receiver 15m away at (25.0, 34.0)
    receivers = [
        {
            "track_id": 10,
            "team_label": "receiver",
            "pitch_xy": [25.0, 34.0],
            "pitch_vel_ms": [0.0, 0.0],
        }
    ]
    # Opponent 4.0m behind receiver closing at 1.0 m/s -> arrives in ~1.25s under burst acceleration.
    # Pass flight for 15m + 0.15s prep latency is ~0.86s -> arrival margin is positive (+0.39s).
    # But total control window (0.45s touch + 0.15s turn = 0.60s) exceeds margin -> hospital pass.
    opponents = [PitchPoint(x=29.0, y=34.0)]
    opp_actors = [
        PressingActor(
            track_id=-1,
            team_id="opp",
            pos_m=PitchPoint(x=29.0, y=34.0),
            vel_ms=(-1.0, 0.0),
        )
    ]

    evals = evaluate_distribution_decision(
        passer_pos=gk_pos,
        receiver_candidates=receivers,
        opponents=opponents,
        evaluator=evaluator,
        passer_facing_angle_rad=0.0,
        passer_gaze_angle_rad=0.0,
        opponent_actors=opp_actors,
    )
    rec_eval = evals[0]
    assert rec_eval["margin_s"] > 0.0  # Ball arrives first
    assert rec_eval["t_post_cushion_s"] < 0.20  # But tackled during first touch
    assert rec_eval["path_status"] == "CONTROL_TRAP"
    assert rec_eval["grade_color"] == "#ef4444"
    assert rec_eval["in_release_window"] is False
    assert rec_eval["is_optimal"] is False
    assert rec_eval["path_score"] <= 15.0


def test_retention_smoothness_and_monotonicity():
    from src.physics.gk_constraints import (
        RETENTION_CRITICAL_CUSHION_S,
        RETENTION_LOGISTIC_STEEPNESS_K,
    )

    def retention(t: float) -> float:
        return 1.0 / (
            1.0
            + math.exp(
                -RETENTION_LOGISTIC_STEEPNESS_K * (t - RETENTION_CRITICAL_CUSHION_S)
            )
        )

    # Inflection point at critical cushion threshold gives exactly 50%
    assert retention(RETENTION_CRITICAL_CUSHION_S) == pytest.approx(0.50, abs=1e-5)

    # Monotonicity and C^1 continuity across continuous range
    steps = [round(-0.20 + i * 0.01, 3) for i in range(81)]  # [-0.20, +0.60]
    vals = [retention(t) for t in steps]

    for i in range(len(vals) - 1):
        assert vals[i] < vals[i + 1], "Retention must be strictly monotonic"
        delta = abs(vals[i + 1] - vals[i])
        assert delta < 0.05, f"Discontinuity detected: step jump of {delta:.4f}"
        assert 0.0 < vals[i] < 1.0, "Retention must stay strictly in (0, 1)"


def test_net_ev_tanh_score_monotonicity_and_calibration():
    from src.physics.gk_constraints import EV_SCORE_SCALE_DELTA_V

    def score(net_ev: float) -> float:
        return 50.0 + 50.0 * math.tanh(net_ev / EV_SCORE_SCALE_DELTA_V)

    # Break-even maps to 50.0
    assert score(0.0) == pytest.approx(50.0, abs=1e-5)

    # Monotonicity
    ev_values = [-0.15, -0.08, -0.04, 0.0, 0.04, 0.08, 0.15]
    scores = [score(ev) for ev in ev_values]
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1]

    # High progression reward and severe turnover hazard bounds
    assert score(0.08) >= 85.0
    assert score(-0.08) <= 15.0
