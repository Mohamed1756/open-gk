"""Unit tests for Continuous Action Manifold (Module M7 extension)."""

import pytest

from src.core.geometry import PitchPoint
from src.models.distribution.manifold import (
    clamp_to_pitch_buffer,
    check_target_ambiguity,
    generate_receiver_manifold,
)
from src.models.pressing.types import PressingActor
from src.physics.gk_constraints import (
    KICK_DISPERSION_RADIAL_RATIO,
    MANIFOLD_MAX_BURST_LEAD_M,
    MANIFOLD_MIN_BURST_LEAD_M,
    MANIFOLD_SHIELD_OFFSET_DIST_M,
    PITCH_TOUCHLINE_BUFFER_M,
)


def test_clamp_to_pitch_buffer():
    # 1. Center of pitch: no clamping, no boundary discount
    pt_center = PitchPoint(x=50.0, y=34.0)
    clamped, is_disc, mult = clamp_to_pitch_buffer(pt_center)
    assert clamped.x == 50.0
    assert clamped.y == 34.0
    assert not is_disc
    assert mult == 1.0

    # 2. Out-of-bounds touchline point (y=67.8 on 68m wide pitch)
    pt_touchline = PitchPoint(x=50.0, y=67.8)
    clamped_tl, is_disc_tl, mult_tl = clamp_to_pitch_buffer(pt_touchline)
    assert clamped_tl.y == 68.0 - PITCH_TOUCHLINE_BUFFER_M
    assert is_disc_tl
    assert mult_tl < 1.0

    # 3. Behind own goal-line point (x=-2.0)
    pt_behind = PitchPoint(x=-2.0, y=34.0)
    clamped_bh, is_disc_bh, mult_bh = clamp_to_pitch_buffer(pt_behind)
    assert clamped_bh.x == PITCH_TOUCHLINE_BUFFER_M
    assert is_disc_bh
    assert mult_bh < 1.0


def test_check_target_ambiguity():
    target = PitchPoint(x=30.0, y=34.0)
    # Receiver 1 at (20.0, 34.0) -> dist = 10m
    tm1 = PressingActor(track_id=1, team_id="own", pos_m=PitchPoint(x=20.0, y=34.0))
    # Receiver 2 at (40.0, 34.0) -> dist = 10m (identical distance, concurrent arrival)
    tm2 = PressingActor(track_id=2, team_id="own", pos_m=PitchPoint(x=40.0, y=34.0))
    # Teammate 3 far away at (10.0, 10.0)
    tm3 = PressingActor(track_id=3, team_id="own", pos_m=PitchPoint(x=10.0, y=10.0))

    # Conflict between tm1 and tm2
    is_amb, partner_id = check_target_ambiguity(
        target=target,
        intended_receiver=tm1,
        teammates=[tm1, tm2, tm3],
        flight_s=2.5,
        threshold_s=0.30,
    )
    assert is_amb
    assert partner_id == 2

    # No conflict when only tm1 and tm3 are present
    is_amb_solo, partner_solo = check_target_ambiguity(
        target=target,
        intended_receiver=tm1,
        teammates=[tm1, tm3],
        flight_s=2.5,
        threshold_s=0.30,
    )
    assert not is_amb_solo
    assert partner_solo is None


def test_generate_receiver_manifold_unpressed():
    passer = PitchPoint(x=5.0, y=34.0)
    # Receiver jogging upfield at 3 m/s
    receiver = PressingActor(
        track_id=10,
        team_id="own",
        pos_m=PitchPoint(x=25.0, y=50.0),
        vel_ms=(3.0, 0.0),
    )
    # Unpressed (no opponent near receiver)
    manifold = generate_receiver_manifold(
        passer_pos=passer,
        receiver=receiver,
        nearest_presser=None,
        teammates=[receiver],
        attack_dir_x=1.0,
    )

    action_types = [t.action_type for t in manifold]
    assert "FEET" in action_types
    assert "PROGRESSIVE_CHANNEL" in action_types
    # No shielded pocket when no presser
    assert "SHIELDED_POCKET" not in action_types

    feet_target = next(t for t in manifold if t.action_type == "FEET")
    channel_target = next(t for t in manifold if t.action_type == "PROGRESSIVE_CHANNEL")

    # Channel target should be ahead of feet target along running vector
    assert channel_target.target_pos.x > feet_target.target_pos.x
    assert (
        MANIFOLD_MIN_BURST_LEAD_M
        <= channel_target.offset_dist_m
        <= MANIFOLD_MAX_BURST_LEAD_M
    )

    # Dispersion should scale with pass distance
    assert feet_target.dispersion_sigma_m == pytest.approx(
        feet_target.pass_dist_m * KICK_DISPERSION_RADIAL_RATIO, abs=1e-2
    )


def test_generate_receiver_manifold_pressed():
    passer = PitchPoint(x=5.0, y=34.0)
    receiver = PressingActor(
        track_id=5,
        team_id="own",
        pos_m=PitchPoint(x=20.0, y=34.0),
        vel_ms=(0.0, 0.0),
    )
    # Opponent marking tightly from behind at (18.0, 34.0)
    opponent = PressingActor(
        track_id=99,
        team_id="opp",
        pos_m=PitchPoint(x=18.0, y=34.0),
        vel_ms=(2.0, 0.0),
    )

    manifold = generate_receiver_manifold(
        passer_pos=passer,
        receiver=receiver,
        nearest_presser=opponent,
        teammates=[receiver],
        attack_dir_x=1.0,
    )

    action_types = [t.action_type for t in manifold]
    assert "FEET" in action_types
    assert "PROGRESSIVE_CHANNEL" in action_types
    assert "SHIELDED_POCKET" in action_types

    shield_target = next(t for t in manifold if t.action_type == "SHIELDED_POCKET")
    # Vector away from opponent (opponent is at x=18, receiver at x=20 -> away is +x)
    assert shield_target.target_pos.x > receiver.pos_m.x
    assert shield_target.offset_dist_m == MANIFOLD_SHIELD_OFFSET_DIST_M


def test_evaluate_distribution_with_manifold():
    from scripts.evaluate_distribution import evaluate_distribution_decision
    from src.models.distribution.evaluator import DistributionEvaluator

    evaluator = DistributionEvaluator()
    gk_pos = PitchPoint(x=10.0, y=34.0)

    receivers = [
        {
            "track_id": 1,
            "team_label": "free_runner",
            "pitch_xy": [30.0, 50.0],
            "pitch_vel_ms": [2.5, 0.0],
        },
        {
            "track_id": 2,
            "team_label": "pressed_target",
            "pitch_xy": [25.0, 34.0],
            "pitch_vel_ms": [0.0, 0.0],
        },
    ]
    opponents = [
        PitchPoint(x=23.0, y=34.0),
    ]

    evals = evaluate_distribution_decision(
        passer_pos=gk_pos,
        receiver_candidates=receivers,
        opponents=opponents,
        evaluator=evaluator,
        passer_facing_angle_rad=0.0,
        passer_gaze_angle_rad=0.0,
        attack_dir_x=1.0,
    )

    assert len(evals) == 2
    for opt in evals:
        assert "action_type" in opt
        assert opt["action_type"] in (
            "FEET",
            "PROGRESSIVE_CHANNEL",
            "SHIELDED_POCKET",
        )
        assert "dispersion_sigma_m" in opt
        assert opt["dispersion_sigma_m"] > 0.0
        assert "manifold_evals" in opt
        assert len(opt["manifold_evals"]) >= 2


def test_progressive_channel_non_regressive_when_tracking_back():
    passer = PitchPoint(x=10.0, y=34.0)
    # Receiver sprinting back toward own goal at -3.0 m/s
    receiver = PressingActor(
        track_id=15,
        team_id="own",
        pos_m=PitchPoint(x=35.0, y=34.0),
        vel_ms=(-3.0, 0.0),
    )
    manifold = generate_receiver_manifold(
        passer_pos=passer,
        receiver=receiver,
        nearest_presser=None,
        teammates=[receiver],
        attack_dir_x=1.0,
    )
    feet = next(t for t in manifold if t.action_type == "FEET")
    channel = next(t for t in manifold if t.action_type == "PROGRESSIVE_CHANNEL")

    # Progressive channel must move forward in attacking direction relative to feet target
    assert channel.target_pos.x > feet.target_pos.x


def test_dynamic_lead_scales_with_flight_time():
    passer = PitchPoint(x=10.0, y=34.0)
    # Short receiver 10m away
    rec_short = PressingActor(
        track_id=1,
        team_id="own",
        pos_m=PitchPoint(x=20.0, y=34.0),
        vel_ms=(2.0, 0.0),
    )
    # Deep receiver 45m away
    rec_deep = PressingActor(
        track_id=2,
        team_id="own",
        pos_m=PitchPoint(x=55.0, y=34.0),
        vel_ms=(2.0, 0.0),
    )
    m_short = generate_receiver_manifold(
        passer, rec_short, None, [rec_short], attack_dir_x=1.0
    )
    m_deep = generate_receiver_manifold(
        passer, rec_deep, None, [rec_deep], attack_dir_x=1.0
    )

    ch_short = next(t for t in m_short if t.action_type == "PROGRESSIVE_CHANNEL")
    ch_deep = next(t for t in m_deep if t.action_type == "PROGRESSIVE_CHANNEL")

    # Deep pass with longer flight time must afford larger dynamic stride lead
    assert ch_deep.offset_dist_m > ch_short.offset_dist_m


def test_shielded_pocket_orthogonal_body_shield():
    passer = PitchPoint(x=5.0, y=34.0)
    receiver = PressingActor(
        track_id=10,
        team_id="own",
        pos_m=PitchPoint(x=25.0, y=34.0),
        vel_ms=(0.0, 0.0),
    )
    # Opponent pressing from the top flank at (25.0, 37.0)
    opponent = PressingActor(
        track_id=99,
        team_id="opp",
        pos_m=PitchPoint(x=25.0, y=37.0),
        vel_ms=(0.0, -1.5),
    )
    manifold = generate_receiver_manifold(
        passer, receiver, opponent, [receiver], attack_dir_x=1.0
    )
    shield = next(t for t in manifold if t.action_type == "SHIELDED_POCKET")

    # Shielded target must step to opposite flank (y < 34.0) away from presser (y=37.0)
    assert shield.target_pos.y < receiver.pos_m.y


def test_target_ambiguity_spatial_filter():
    target = PitchPoint(x=30.0, y=50.0)
    intended = PressingActor(
        track_id=1,
        team_id="own",
        pos_m=PitchPoint(x=27.0, y=50.0),
        vel_ms=(2.0, 0.0),
    )
    # Distant teammate 35m away on opposite flank
    distant_tm = PressingActor(
        track_id=2,
        team_id="own",
        pos_m=PitchPoint(x=27.0, y=15.0),
        vel_ms=(2.0, 0.0),
    )
    # Distant teammate must NOT trigger ambiguity despite similar arrival scalar
    is_amb, _ = check_target_ambiguity(
        target=target,
        intended_receiver=intended,
        teammates=[intended, distant_tm],
        flight_s=1.2,
    )
    assert not is_amb


def test_clamp_to_pitch_buffer_dispersion_decay():
    # Pass aimed outside pitch bounds (y = 70.0 on 68m pitch)
    out_target = PitchPoint(x=50.0, y=70.0)
    clamped, is_disc, mult = clamp_to_pitch_buffer(out_target, dispersion_sigma_m=2.5)
    assert is_disc
    assert clamped.y == 66.0
    # Out of bounds pass with dispersion receives heavy discount (< 0.40)
    assert mult < 0.40
