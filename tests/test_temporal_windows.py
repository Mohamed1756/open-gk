"""Tests for continuous temporal passing window engine (Module M7).

Verifies continuous differential kinematics and all 8 edge cases:
1. Multi-defender infimum
2. Dynamic raycast corridor occlusion
3. Biomechanical perception-action latency floor (0.35s)
4. Touchline / pitch boundary buffer clipping
5. Passer harassment time-to-contact clock
6. Offside line coordinate suppression
7. Schmitt-trigger hysteresis sensor noise filtering
8. Stationary receiver decoupling (derivative != admissibility)
"""

from src.core.geometry import PitchPoint
from src.models.distribution.temporal_windows import (
    OFFSIDE_CUSHION_PENALTY_S,
    OUT_OF_BOUNDS_PENALTY_S,
    calculate_offside_line_x,
    evaluate_frame_cushions,
    evaluate_temporal_sequence,
    extract_passing_windows,
    is_corridor_occluded,
    separation_rate_ms,
)
from src.models.pressing.types import PressingActor, PressingSnapshot


def test_separation_rate_kinematics() -> None:
    """Verifies differential separation velocity d_dot = n . (v_a - v_b)."""
    pos_rec = PitchPoint(x=20.0, y=34.0)
    pos_press = PitchPoint(x=10.0, y=34.0)

    # Receiver running away from presser (separating)
    rate_open = separation_rate_ms(pos_rec, (6.0, 0.0), pos_press, (2.0, 0.0))
    assert rate_open == 4.0

    # Presser gaining on receiver (closing down)
    rate_close = separation_rate_ms(pos_rec, (2.0, 0.0), pos_press, (6.0, 0.0))
    assert rate_close == -4.0

    # Stationary pocket (zero derivative)
    rate_stat = separation_rate_ms(pos_rec, (0.0, 0.0), pos_press, (0.0, 0.0))
    assert rate_stat == 0.0


def test_multi_defender_infimum_switch() -> None:
    """Verifies margin is governed by the covering defender, not the primary marker."""
    passer = PressingActor(track_id=1, team_id="home", pos_m=PitchPoint(x=5.0, y=34.0))
    # Receiver at x=25, y=34, moving right
    receiver = PressingActor(
        track_id=10,
        team_id="home",
        pos_m=PitchPoint(x=25.0, y=34.0),
        vel_ms=(4.0, 0.0),
    )
    # Marker 1 is far behind and wide (beaten by receiver)
    marker1 = PressingActor(
        track_id=20,
        team_id="away",
        pos_m=PitchPoint(x=15.0, y=20.0),
        vel_ms=(2.0, 0.0),
    )
    # Covering defender 2 is right at the receiver's anticipated target (x=30, y=34)
    covering_def = PressingActor(
        track_id=21,
        team_id="away",
        pos_m=PitchPoint(x=31.0, y=34.0),
        vel_ms=(0.0, 0.0),
    )

    snapshot = PressingSnapshot(
        passer=passer,
        receivers=[receiver],
        opponents=[marker1, covering_def],
        timestamp_s=0.0,
    )

    result = evaluate_frame_cushions(snapshot)
    # Margin must be tiny/negative because of covering_def, NOT large because of marker1
    assert result.cushions_s[10] < 0.5
    assert result.closest_presser_ids[10] == 21


def test_corridor_occlusion_invalidation() -> None:
    """Verifies that an opponent standing in the passing lane occludes the pass."""
    passer_pos = PitchPoint(x=10.0, y=34.0)
    target_pos = PitchPoint(x=40.0, y=34.0)

    # Opponent right on the corridor line at x=25, y=34
    blocker = PressingActor(
        track_id=30,
        team_id="away",
        pos_m=PitchPoint(x=25.0, y=34.0),
        vel_ms=(0.0, 0.0),
    )
    # Opponent far away laterally at x=25, y=10.0
    wide_opp = PressingActor(
        track_id=31,
        team_id="away",
        pos_m=PitchPoint(x=25.0, y=10.0),
        vel_ms=(0.0, 0.0),
    )

    assert is_corridor_occluded(passer_pos, target_pos, [blocker]) is True
    assert is_corridor_occluded(passer_pos, target_pos, [wide_opp]) is False


def test_biomechanical_sub_perceptual_noise() -> None:
    """Verifies that a flash window shorter than 0.35s is tagged as SUB_PERCEPTUAL_NOISE."""
    timestamps = [round(i * 0.04, 2) for i in range(25)]
    # Window opens at index 5 (t=0.20) and drops at index 8 (t=0.32) -> duration = 0.12s < 0.35s
    margins = [-0.5] * 5 + [0.35, 0.40, 0.30] + [-0.5] * 17
    rates = [0.0] * 25
    pressers = [20] * 25

    windows = extract_passing_windows(
        margins=margins,
        timestamps=timestamps,
        separation_rates=rates,
        closest_presser_ids=pressers,
        receiver_track_id=10,
    )

    assert len(windows) == 1
    w = windows[0]
    assert w.is_actionable is False
    assert w.rejection_reason == "SUB_PERCEPTUAL_NOISE"
    assert w.duration_s < 0.35


def test_hysteresis_noise_rejection() -> None:
    """Verifies that sensor jitter around 0.0s does not open spurious windows."""
    # Jitter oscillates between -0.05 and +0.08 (never crosses OPEN threshold +0.20)
    jitter = [-0.05, 0.08, -0.02, 0.07, -0.04, 0.06, -0.03]
    timestamps = [round(i * 0.04, 2) for i in range(len(jitter))]
    rates = [0.0] * len(jitter)
    pressers = [20] * len(jitter)

    windows = extract_passing_windows(
        margins=jitter,
        timestamps=timestamps,
        separation_rates=rates,
        closest_presser_ids=pressers,
        receiver_track_id=10,
    )
    assert len(windows) == 0


def test_offside_suppression() -> None:
    """Verifies that receiver beyond the second-to-last defender is suppressed."""
    passer = PressingActor(track_id=1, team_id="home", pos_m=PitchPoint(x=15.0, y=34.0))
    # Receiver at x=80 in attacking half
    receiver = PressingActor(
        track_id=10,
        team_id="home",
        pos_m=PitchPoint(x=80.0, y=34.0),
        vel_ms=(0.0, 0.0),
    )
    # Opponent GK at x=100, last outfield defender at x=70
    opp_gk = PressingActor(
        track_id=99, team_id="away", pos_m=PitchPoint(x=100.0, y=34.0)
    )
    opp_def = PressingActor(
        track_id=30, team_id="away", pos_m=PitchPoint(x=70.0, y=34.0)
    )

    offside_x = calculate_offside_line_x([opp_gk, opp_def], attack_dir_x=1.0)
    assert offside_x == 70.0

    snapshot = PressingSnapshot(
        passer=passer,
        receivers=[receiver],
        opponents=[opp_gk, opp_def],
        timestamp_s=0.0,
    )
    result = evaluate_frame_cushions(snapshot)
    assert result.cushions_s[10] == OFFSIDE_CUSHION_PENALTY_S
    assert 10 in result.offside_receivers


def test_passer_harassment_truncation() -> None:
    """Verifies that windows peaking after the goalkeeper is tackled are invalidated."""
    timestamps = [round(i * 0.04, 2) for i in range(30)]
    # Window opens at t=0.40 and peaks at t=0.80
    margins = [-0.5] * 10 + [0.5] * 15 + [-0.5] * 5
    rates = [1.0] * 30
    pressers = [20] * 30

    # Goalkeeper is tackled at t=0.50 (cutoff = 0.50)
    windows = extract_passing_windows(
        margins=margins,
        timestamps=timestamps,
        separation_rates=rates,
        closest_presser_ids=pressers,
        receiver_track_id=10,
        gk_tackle_cutoff_s=0.50,
    )

    assert len(windows) == 1
    w = windows[0]
    assert w.is_actionable is False
    assert w.rejection_reason == "PASSER_HARASSED"


def test_stationary_free_man_admissibility() -> None:
    """Verifies an unmarked stationary player (d_dot=0) is a valid actionable option."""
    timestamps = [round(i * 0.04, 2) for i in range(25)]
    # Player has steady +1.5s cushion for 1.0s duration
    margins = [1.5] * 25
    rates = [0.0] * 25  # Zero separation rate (stationary)
    pressers = [20] * 25

    windows = extract_passing_windows(
        margins=margins,
        timestamps=timestamps,
        separation_rates=rates,
        closest_presser_ids=pressers,
        receiver_track_id=10,
    )

    assert len(windows) == 1
    w = windows[0]
    assert w.is_actionable is True
    assert w.peak_cushion_s == 1.5
    assert w.peak_separation_rate_ms == 0.0
    assert w.rejection_reason is None


def test_boundary_buffer_clipping() -> None:
    """Verifies that running out of pitch bounds incurs an out-of-bounds penalty."""
    passer = PressingActor(track_id=1, team_id="home", pos_m=PitchPoint(x=15.0, y=34.0))
    # Receiver sprinting over the sideline at y = 67.5 (pitch width is 68m, buffer 1.5m)
    oob_receiver = PressingActor(
        track_id=10,
        team_id="home",
        pos_m=PitchPoint(x=30.0, y=67.0),
        vel_ms=(0.0, 4.0),
    )
    opp = PressingActor(track_id=20, team_id="away", pos_m=PitchPoint(x=20.0, y=50.0))

    snapshot = PressingSnapshot(
        passer=passer,
        receivers=[oob_receiver],
        opponents=[opp],
        timestamp_s=0.0,
    )
    result = evaluate_frame_cushions(snapshot)
    assert result.cushions_s[10] == OUT_OF_BOUNDS_PENALTY_S


def test_evaluate_temporal_sequence_end_to_end() -> None:
    """Verifies multi-frame sequence evaluation extracting best window and summaries."""
    passer = PressingActor(track_id=1, team_id="home", pos_m=PitchPoint(x=10.0, y=34.0))
    # Receiver running into space
    receiver = PressingActor(
        track_id=10,
        team_id="home",
        pos_m=PitchPoint(x=30.0, y=34.0),
        vel_ms=(4.0, 0.0),
    )
    # Opponent presser trailing wide
    opp = PressingActor(
        track_id=20,
        team_id="away",
        pos_m=PitchPoint(x=20.0, y=25.0),
        vel_ms=(2.0, 0.0),
    )

    # 15 frames at 25 Hz (0.6s)
    snapshots = [
        PressingSnapshot(
            passer=passer,
            receivers=[receiver],
            opponents=[opp],
            timestamp_s=round(i * 0.04, 2),
        )
        for i in range(15)
    ]

    eval_result = evaluate_temporal_sequence(snapshots)
    assert len(eval_result.timestamps_s) == 15
    assert 10 in eval_result.receiver_summaries
    summary = eval_result.receiver_summaries[10]
    assert len(summary.margin_series_s) == 15
    assert eval_result.best_window is not None
    assert eval_result.best_window.receiver_track_id == 10
    assert eval_result.best_window.is_actionable is True


def test_onside_through_ball_not_suppressed() -> None:
    """Verifies that a runner starting onside and sprinting behind the defense is not flagged offside."""
    passer = PressingActor(track_id=1, team_id="home", pos_m=PitchPoint(x=15.0, y=34.0))
    # Striker starts at x=65 (behind defender at x=70), sprinting forward at 6 m/s
    striker = PressingActor(
        track_id=10,
        team_id="home",
        pos_m=PitchPoint(x=65.0, y=34.0),
        vel_ms=(6.0, 0.0),
    )
    opp_gk = PressingActor(
        track_id=99, team_id="away", pos_m=PitchPoint(x=100.0, y=34.0)
    )
    opp_def = PressingActor(
        track_id=30, team_id="away", pos_m=PitchPoint(x=70.0, y=5.0)
    )

    snapshot = PressingSnapshot(
        passer=passer,
        receivers=[striker],
        opponents=[opp_gk, opp_def],
        timestamp_s=0.0,
    )
    result = evaluate_frame_cushions(snapshot)
    assert 10 not in result.offside_receivers
    assert result.cushions_s[10] != OFFSIDE_CUSHION_PENALTY_S


def test_dynamic_lateral_interceptor_detected() -> None:
    """Verifies that an opponent starting 2.5m off the lane sprinting to intercept is detected."""
    passer_pos = PitchPoint(x=10.0, y=34.0)
    target_pos = PitchPoint(x=40.0, y=34.0)

    # Opponent starts 2.5m lateral offset at (25.0, 36.5), sprinting down at -5 m/s toward the lane
    lateral_sprinter = PressingActor(
        track_id=30,
        team_id="away",
        pos_m=PitchPoint(x=25.0, y=36.5),
        vel_ms=(0.0, -5.0),
    )

    assert is_corridor_occluded(passer_pos, target_pos, [lateral_sprinter]) is True


def test_decoupled_smoothing_resilient_to_single_frame_glitch() -> None:
    """Verifies a single-frame occlusion does not ruin subsequent frames in the moving average."""
    passer = PressingActor(track_id=1, team_id="home", pos_m=PitchPoint(x=10.0, y=34.0))
    receiver = PressingActor(
        track_id=10,
        team_id="home",
        pos_m=PitchPoint(x=30.0, y=34.0),
        vel_ms=(4.0, 0.0),
    )
    # Opponent trailing wide normally
    opp_normal = PressingActor(
        track_id=20,
        team_id="away",
        pos_m=PitchPoint(x=20.0, y=25.0),
        vel_ms=(2.0, 0.0),
    )
    # Opponent momentarily blocking corridor at frame 7 only
    opp_blocking = PressingActor(
        track_id=20,
        team_id="away",
        pos_m=PitchPoint(x=20.0, y=34.0),
        vel_ms=(0.0, 0.0),
    )

    snapshots = [
        PressingSnapshot(
            passer=passer,
            receivers=[receiver],
            opponents=[opp_blocking if i == 7 else opp_normal],
            timestamp_s=round(i * 0.04, 2),
        )
        for i in range(25)
    ]

    eval_result = evaluate_temporal_sequence(snapshots)
    summary = eval_result.receiver_summaries[10]
    # Frame 8 (immediately after the glitch) should recover physical cushion without being dragged to -12.8s
    assert summary.margin_series_s[8] > 0.5
