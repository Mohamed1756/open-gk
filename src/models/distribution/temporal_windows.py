# Justification: Combines 25 Hz continuous temporal passing window simulation, multi-defender infimum tracking, and raycast occlusion.
"""Continuous Temporal Passing Window Evaluation (Module M7).

Evaluates outfield passing options across continuous tracking sequences at 25 Hz.
Extracts actionable temporal windows, optimal release epochs, and separation
kinematics while enforcing 8 mathematical edge cases:
1. Multi-defender infimum (defensive switches and covering sweepers)
2. Dynamic raycast corridor occlusion
3. Biomechanical perception-action latency floor (0.35s)
4. Touchline / pitch boundary buffer clipping
5. Passer harassment time-to-contact clock
6. Offside line coordinate suppression
7. Schmitt-trigger hysteresis sensor noise filtering
8. Stationary receiver decoupling (derivative != admissibility)
"""
# Note (§2.6): File length (~450 lines) encapsulates all 8 edge cases, multi-defender
# infimum logic, and temporal sequence dataclasses in a single cohesive module.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.config import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from src.core.geometry import PitchPoint
from src.models.pressing.arrival import (
    flight_time_s,
    lead_target,
    presser_arrival_s,
)
from src.models.pressing.types import PressingActor, PressingSnapshot
from src.physics.gk_constraints import (
    MIN_BIOMECHANICAL_WINDOW_S,
    PASSER_HARASSMENT_BUFFER_S,
    PASS_GROUND_INITIAL_SPEED_MS,
    PITCH_TOUCHLINE_BUFFER_M,
    PRESSER_SPRINT_MAX_SPEED_MS,
    SMOOTHING_WINDOW_FRAMES,
    WINDOW_CLOSE_THRESHOLD_S,
    WINDOW_OPEN_THRESHOLD_S,
)

OCCLUDED_CUSHION_PENALTY_S: float = -99.0
OFFSIDE_CUSHION_PENALTY_S: float = -99.0
OUT_OF_BOUNDS_PENALTY_S: float = -99.0


@dataclass(frozen=True)
class PassingWindow:
    """Represents an actionable or sub-perceptual temporal passing window."""

    receiver_track_id: int
    t_open_s: float
    t_close_s: float
    duration_s: float
    peak_epoch_s: float
    peak_cushion_s: float
    peak_separation_rate_ms: float
    is_actionable: bool
    rejection_reason: Optional[str]
    closest_presser_id: Optional[int]


@dataclass(frozen=True)
class ReceiverTrajectorySummary:
    """Temporal evolution summary for an individual receiver over a sequence."""

    receiver_track_id: int
    windows: List[PassingWindow]
    margin_series_s: List[float]
    timestamps_s: List[float]
    separation_rate_series_ms: List[float]
    is_active_at_release: bool


@dataclass(frozen=True)
class FrameCushionResult:
    """Instantaneous cushion evaluation for all receivers at a single frame."""

    timestamp_s: float
    cushions_s: Dict[int, float]
    physical_cushions_s: Dict[int, float]
    separation_rates_ms: Dict[int, float]
    closest_presser_ids: Dict[int, Optional[int]]
    occluded_receivers: List[int]
    offside_receivers: List[int]
    out_of_bounds_receivers: List[int]
    gk_tackle_time_s: float


@dataclass(frozen=True)
class SequenceTemporalEvaluation:
    """End-to-end multi-frame temporal window evaluation for distribution."""

    timestamps_s: List[float]
    receiver_summaries: Dict[int, ReceiverTrajectorySummary]
    actionable_windows: List[PassingWindow]
    gk_tackle_time_s: Optional[float]
    best_window: Optional[PassingWindow]


def separation_rate_ms(
    pos_a: PitchPoint,
    vel_a: Tuple[float, float],
    pos_b: PitchPoint,
    vel_b: Tuple[float, float],
) -> float:
    """Computes differential separation velocity d_dot = n . (v_a - v_b).

    Positive indicates player A is actively separating from player B.
    """
    dx = pos_a.x - pos_b.x
    dy = pos_a.y - pos_b.y
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return 0.0
    nx = dx / dist
    ny = dy / dist
    dvx = vel_a[0] - vel_b[0]
    dvy = vel_a[1] - vel_b[1]
    dot = nx * dvx + ny * dvy
    return round(dot, 3)


def is_corridor_occluded(
    passer_pos: PitchPoint,
    target_pos: PitchPoint,
    opponents: Sequence[PressingActor],
    pass_speed_ms: float = PASS_GROUND_INITIAL_SPEED_MS,
    intercept_radius_m: float = 1.2,
) -> bool:
    """Raycasts corridor from passer to target; returns True if an opponent intercepts."""
    dx = target_pos.x - passer_pos.x
    dy = target_pos.y - passer_pos.y
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return False

    for op in opponents:
        if not op.pitch_valid:
            continue
        op_dx = op.pos_m.x - passer_pos.x
        op_dy = op.pos_m.y - passer_pos.y
        s = max(0.0, min(1.0, (op_dx * dx + op_dy * dy) / (length * length)))
        qx = passer_pos.x + s * dx
        qy = passer_pos.y + s * dy
        dist_to_ray = math.hypot(op.pos_m.x - qx, op.pos_m.y - qy)

        ball_dist = s * length
        t_ball = flight_time_s(ball_dist, v0_ms=pass_speed_ms)
        max_reach_m = intercept_radius_m + PRESSER_SPRINT_MAX_SPEED_MS * t_ball
        if dist_to_ray <= max_reach_m:
            t_press, _ = presser_arrival_s(op, PitchPoint(x=qx, y=qy))
            if t_press <= t_ball + 0.10:
                return True
    return False


def calculate_offside_line_x(
    opponents: Sequence[PressingActor],
    attack_dir_x: float = 1.0,
) -> float:
    """Calculates the second-to-last defender offside threshold line."""
    halfway_x = PITCH_LENGTH_METERS / 2.0
    valid_xs = [op.pos_m.x for op in opponents if op.pitch_valid]
    if len(valid_xs) < 2:
        return halfway_x

    if attack_dir_x > 0:
        sorted_xs = sorted(valid_xs, reverse=True)
        return max(halfway_x, sorted_xs[1])
    sorted_xs = sorted(valid_xs)
    return min(halfway_x, sorted_xs[1])


def _clip_target_to_pitch(
    target: PitchPoint,
    buffer_m: float = PITCH_TOUCHLINE_BUFFER_M,
) -> Tuple[PitchPoint, bool]:
    """Clips lead target inside boundaries; returns clipped point and out-of-bounds flag."""
    min_x, max_x = buffer_m, PITCH_LENGTH_METERS - buffer_m
    min_y, max_y = buffer_m, PITCH_WIDTH_METERS - buffer_m
    is_oob = (
        target.x < min_x or target.x > max_x or target.y < min_y or target.y > max_y
    )
    cx = max(min_x, min(max_x, target.x))
    cy = max(min_y, min(max_y, target.y))
    return PitchPoint(x=round(cx, 2), y=round(cy, 2)), is_oob


def _evaluate_receiver_cushion(
    rec: PressingActor,
    passer_pos: PitchPoint,
    valid_opp: Sequence[PressingActor],
    offside_line_x: float,
    attack_dir_x: float,
    pass_speed_ms: float,
) -> Tuple[float, float, float, Optional[int], bool, bool, bool]:
    """Evaluates physical & effective cushion, separation rate, presser, and status flags."""
    dist = math.hypot(rec.pos_m.x - passer_pos.x, rec.pos_m.y - passer_pos.y)
    t_ball = flight_time_s(dist, v0_ms=pass_speed_ms)
    for _ in range(2):
        lead = lead_target(rec.pos_m, rec.vel_ms, t_ball)
        dist = math.hypot(lead.x - passer_pos.x, lead.y - passer_pos.y)
        t_ball = flight_time_s(dist, v0_ms=pass_speed_ms)
    lead = lead_target(rec.pos_m, rec.vel_ms, t_ball)
    clipped_lead, is_oob = _clip_target_to_pitch(lead)

    # Law 11: Offside evaluated at release epoch coordinate, not reception point
    is_off = (
        rec.pos_m.x > offside_line_x
        if attack_dir_x > 0
        else rec.pos_m.x < offside_line_x
    )
    is_occ = is_corridor_occluded(passer_pos, clipped_lead, valid_opp, pass_speed_ms)

    min_margin = 99.0
    best_op: Optional[PressingActor] = None
    for op in valid_opp:
        arr, _ = presser_arrival_s(op, clipped_lead)
        margin = arr - t_ball
        if margin < min_margin:
            min_margin = margin
            best_op = op

    rate = (
        separation_rate_ms(rec.pos_m, rec.vel_ms, best_op.pos_m, best_op.vel_ms)
        if best_op
        else 0.0
    )
    presser_id = best_op.track_id if best_op else None
    phys_cushion = round(min_margin, 3)
    eff_cushion = (
        OUT_OF_BOUNDS_PENALTY_S
        if is_oob
        else (
            OFFSIDE_CUSHION_PENALTY_S
            if is_off
            else OCCLUDED_CUSHION_PENALTY_S
            if is_occ
            else phys_cushion
        )
    )
    return phys_cushion, eff_cushion, rate, presser_id, is_occ, is_off, is_oob


def evaluate_frame_cushions(
    snapshot: PressingSnapshot,
    pass_speed_ms: float = PASS_GROUND_INITIAL_SPEED_MS,
) -> FrameCushionResult:
    """Evaluates multi-defender infimum margins, occlusion, and offside for one frame."""
    valid_opp = [o for o in snapshot.opponents if o.pitch_valid]
    passer_pos = snapshot.ball_pos_m or snapshot.passer.pos_m

    gk_tackle = min(
        (presser_arrival_s(op, passer_pos)[0] for op in valid_opp),
        default=99.0,
    )
    offside_line_x = calculate_offside_line_x(valid_opp, snapshot.attack_dir_x)

    cushions: Dict[int, float] = {}
    phys_cushions: Dict[int, float] = {}
    sep_rates: Dict[int, float] = {}
    closest_pressers: Dict[int, Optional[int]] = {}
    occluded_rec: List[int] = []
    offside_rec: List[int] = []
    oob_rec: List[int] = []

    for rec in snapshot.receivers:
        if not rec.pitch_valid:
            continue
        phys, eff, rate, press_id, is_occ, is_off, is_oob = _evaluate_receiver_cushion(
            rec,
            passer_pos,
            valid_opp,
            offside_line_x,
            snapshot.attack_dir_x,
            pass_speed_ms,
        )
        cushions[rec.track_id] = eff
        phys_cushions[rec.track_id] = phys
        sep_rates[rec.track_id] = rate
        closest_pressers[rec.track_id] = press_id
        if is_occ:
            occluded_rec.append(rec.track_id)
        if is_off:
            offside_rec.append(rec.track_id)
        if is_oob:
            oob_rec.append(rec.track_id)

    return FrameCushionResult(
        timestamp_s=snapshot.timestamp_s,
        cushions_s=cushions,
        physical_cushions_s=phys_cushions,
        separation_rates_ms=sep_rates,
        closest_presser_ids=closest_pressers,
        occluded_receivers=occluded_rec,
        offside_receivers=offside_rec,
        out_of_bounds_receivers=oob_rec,
        gk_tackle_time_s=round(gk_tackle, 3),
    )


def smooth_series(
    values: Sequence[float],
    window_size: int = SMOOTHING_WINDOW_FRAMES,
) -> List[float]:
    """Causal moving-average smoothing for 25 Hz tracking noise suppression."""
    n = len(values)
    if n == 0 or window_size <= 1:
        return list(values)
    smoothed: List[float] = []
    for i in range(n):
        start = max(0, i - window_size + 1)
        sub = values[start : i + 1]
        smoothed.append(round(sum(sub) / len(sub), 3))
    return smoothed


def _build_passing_window(
    margins: Sequence[float],
    timestamps: Sequence[float],
    separation_rates: Sequence[float],
    closest_presser_ids: Sequence[Optional[int]],
    receiver_track_id: int,
    start_idx: int,
    end_idx: int,
    min_biomech_s: float,
    gk_tackle_cutoff_s: Optional[float],
) -> PassingWindow:
    """Constructs and validates a single PassingWindow from an open index interval."""
    t_open, t_close = timestamps[start_idx], timestamps[end_idx]
    dur = round(max(0.0, t_close - t_open), 3)
    sub_margins = margins[start_idx : end_idx + 1]
    max_sub = max(sub_margins)
    peak_sub_idx = start_idx + sub_margins.index(max_sub)

    is_actionable = True
    reason: Optional[str] = None
    if gk_tackle_cutoff_s is not None and t_open >= gk_tackle_cutoff_s:
        is_actionable, reason = False, "PASSER_HARASSED"
    elif gk_tackle_cutoff_s is not None and t_close > gk_tackle_cutoff_s:
        t_close = gk_tackle_cutoff_s
        dur = round(max(0.0, t_close - t_open), 3)
        if dur < min_biomech_s:
            is_actionable, reason = False, "PASSER_HARASSED"
    elif dur < min_biomech_s:
        is_actionable, reason = False, "SUB_PERCEPTUAL_NOISE"

    return PassingWindow(
        receiver_track_id=receiver_track_id,
        t_open_s=round(t_open, 3),
        t_close_s=round(t_close, 3),
        duration_s=dur,
        peak_epoch_s=round(timestamps[peak_sub_idx], 3),
        peak_cushion_s=round(max_sub, 3),
        peak_separation_rate_ms=separation_rates[peak_sub_idx],
        is_actionable=is_actionable,
        rejection_reason=reason,
        closest_presser_id=closest_presser_ids[peak_sub_idx],
    )


def extract_passing_windows(
    margins: Sequence[float],
    timestamps: Sequence[float],
    separation_rates: Sequence[float],
    closest_presser_ids: Sequence[Optional[int]],
    receiver_track_id: int,
    min_biomech_s: float = MIN_BIOMECHANICAL_WINDOW_S,
    open_thresh_s: float = WINDOW_OPEN_THRESHOLD_S,
    close_thresh_s: float = WINDOW_CLOSE_THRESHOLD_S,
    gk_tackle_cutoff_s: Optional[float] = None,
) -> List[PassingWindow]:
    """Extracts continuous passing windows using Schmitt-trigger hysteresis state transitions."""
    n = len(margins)
    if n == 0:
        return []

    windows: List[PassingWindow] = []
    in_window = False
    start_idx = 0

    for i in range(n):
        cushion = margins[i]
        if not in_window and cushion >= open_thresh_s:
            in_window = True
            start_idx = i
        elif in_window and (cushion <= close_thresh_s or i == n - 1):
            in_window = False
            win = _build_passing_window(
                margins=margins,
                timestamps=timestamps,
                separation_rates=separation_rates,
                closest_presser_ids=closest_presser_ids,
                receiver_track_id=receiver_track_id,
                start_idx=start_idx,
                end_idx=i,
                min_biomech_s=min_biomech_s,
                gk_tackle_cutoff_s=gk_tackle_cutoff_s,
            )
            windows.append(win)

    return windows


def _build_receiver_summary(
    rid: int,
    frame_results: Sequence[FrameCushionResult],
    timestamps: Sequence[float],
    min_biomech_s: float,
    cutoff_s: float,
) -> ReceiverTrajectorySummary:
    """Builds trajectory summary and passing windows for one receiver across frames."""
    raw_phys = [fr.physical_cushions_s.get(rid, -99.0) for fr in frame_results]
    rates = [fr.separation_rates_ms.get(rid, 0.0) for fr in frame_results]
    pressers = [fr.closest_presser_ids.get(rid, None) for fr in frame_results]
    smoothed_phys = smooth_series(raw_phys)

    admissible = [
        (
            rid not in fr.occluded_receivers
            and rid not in fr.offside_receivers
            and rid not in fr.out_of_bounds_receivers
        )
        for fr in frame_results
    ]
    effective_margins = [
        sm if adm else OCCLUDED_CUSHION_PENALTY_S
        for sm, adm in zip(smoothed_phys, admissible)
    ]

    wins = extract_passing_windows(
        margins=effective_margins,
        timestamps=timestamps,
        separation_rates=rates,
        closest_presser_ids=pressers,
        receiver_track_id=rid,
        min_biomech_s=min_biomech_s,
        gk_tackle_cutoff_s=cutoff_s,
    )
    last_margin = effective_margins[-1] if effective_margins else -99.0
    return ReceiverTrajectorySummary(
        receiver_track_id=rid,
        windows=wins,
        margin_series_s=effective_margins,
        timestamps_s=list(timestamps),
        separation_rate_series_ms=rates,
        is_active_at_release=(last_margin >= WINDOW_OPEN_THRESHOLD_S),
    )


def evaluate_temporal_sequence(
    snapshots: Sequence[PressingSnapshot],
    pass_speed_ms: float = PASS_GROUND_INITIAL_SPEED_MS,
    min_biomech_s: float = MIN_BIOMECHANICAL_WINDOW_S,
) -> SequenceTemporalEvaluation:
    """Evaluates a multi-frame tracking sequence and extracts all passing windows."""
    if not snapshots:
        return SequenceTemporalEvaluation(
            timestamps_s=[],
            receiver_summaries={},
            actionable_windows=[],
            gk_tackle_time_s=None,
            best_window=None,
        )

    timestamps = [s.timestamp_s for s in snapshots]
    frame_results = [evaluate_frame_cushions(s, pass_speed_ms) for s in snapshots]
    all_rec_ids = sorted({rid for fr in frame_results for rid in fr.cushions_s.keys()})
    min_gk = min(fr.gk_tackle_time_s for fr in frame_results)
    earliest_tackle_epoch = min(
        fr.timestamp_s + fr.gk_tackle_time_s for fr in frame_results
    )
    cutoff_s = earliest_tackle_epoch - PASSER_HARASSMENT_BUFFER_S

    summaries = {
        rid: _build_receiver_summary(
            rid, frame_results, timestamps, min_biomech_s, cutoff_s
        )
        for rid in all_rec_ids
    }
    actionable_all = [
        w for s in summaries.values() for w in s.windows if w.is_actionable
    ]
    best_win = (
        max(actionable_all, key=lambda w: w.peak_cushion_s) if actionable_all else None
    )

    return SequenceTemporalEvaluation(
        timestamps_s=timestamps,
        receiver_summaries=summaries,
        actionable_windows=actionable_all,
        gk_tackle_time_s=round(min_gk, 3),
        best_window=best_win,
    )
