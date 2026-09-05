"""Duel resolution: observed post-release outcomes from tracking.

Classifies what actually happened after the ball leaves the passer: who
controlled it first, whether the team retained it, and how it was lost.
All functions are pure over plain sequences (positions in meters, teams as
strings); missing ball frames are None. Retention is team possession, so a
one-touch layoff under no pressure still counts as retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from src.config import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from src.physics.gk_constraints import (
    DUEL_CONTROL_RADIUS_M,
    RETENTION_MAX_GAP_FRAMES,
    RETENTION_WINDOW_S,
)

# Person frame entry: (track_id, team_id, x_m, y_m).
PersonRow = Tuple[int, str, float, float]


class DuelOutcome(str, Enum):
    RETAINED = "RETAINED"
    CONTESTED_RETAINED = "CONTESTED_RETAINED"
    TACKLED = "TACKLED"
    INTERCEPTED = "INTERCEPTED"
    BLOCKED = "BLOCKED"
    OUT_OF_PLAY = "OUT_OF_PLAY"
    OFFSIDE_INVALID = "OFFSIDE_INVALID"
    NOT_CONTROLLED = "NOT_CONTROLLED"
    LOST_TRACK = "LOST_TRACK"
    TRUNCATED_UNKNOWN = "TRUNCATED_UNKNOWN"
    NO_RELEASE = "NO_RELEASE"


@dataclass(frozen=True)
class DuelOutcomeResult:
    outcome: DuelOutcome
    controller_track_id: Optional[int] = None
    controller_team: Optional[str] = None
    control_frame_idx: Optional[int] = None
    release_frame_idx: Optional[int] = None
    contested: bool = False
    note: str = ""


def _gap(xy_a: Tuple[float, float], xy_b: Tuple[float, float]) -> float:
    return math.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1])


def release_frame_idx(
    ball_gaps_m: List[Optional[float]],
    touch_idx: int,
    radius_m: float = DUEL_CONTROL_RADIUS_M,
) -> Optional[int]:
    """First frame after touch where the ball leaves passer control radius."""
    for idx in range(touch_idx + 1, len(ball_gaps_m)):
        gap = ball_gaps_m[idx]
        if gap is not None and gap > radius_m:
            return idx
    return None


def first_controller(
    ball_xy: List[Optional[Tuple[float, float]]],
    person_frames: List[List[PersonRow]],
    start_idx: int,
    radius_m: float = DUEL_CONTROL_RADIUS_M,
) -> Optional[Tuple[int, int, str]]:
    """First (frame, track_id, team) with anyone inside control radius."""
    for idx in range(start_idx, len(ball_xy)):
        ball = ball_xy[idx]
        if ball is None or idx >= len(person_frames):
            continue
        best: Optional[Tuple[float, int, str]] = None
        for tid, team, x, y in person_frames[idx]:
            dist = _gap(ball, (x, y))
            if dist <= radius_m and (best is None or dist < best[0]):
                best = (dist, tid, team)
        if best is not None:
            return idx, best[1], best[2]
    return None


def _out_of_play(ball: Tuple[float, float]) -> bool:
    return not (
        0.0 <= ball[0] <= PITCH_LENGTH_METERS and 0.0 <= ball[1] <= PITCH_WIDTH_METERS
    )


def detect_deflection(
    ball_xy: List[Optional[Tuple[float, float]]],
    person_frames: List[List[PersonRow]],
    start_idx: int,
    end_idx: int,
    opponent_teams: List[str],
    radius_m: float = DUEL_CONTROL_RADIUS_M,
    min_speed_ms: float = 3.0,
    fps: float = 25.0,
) -> Optional[Tuple[int, int]]:
    """Mid-flight velocity flip beside an opponent: a block, with interceptor."""
    for idx in range(max(1, start_idx), min(end_idx, len(ball_xy) - 1)):
        prev, cur, nxt = ball_xy[idx - 1], ball_xy[idx], ball_xy[idx + 1]
        if prev is None or cur is None or nxt is None:
            continue
        v0 = ((cur[0] - prev[0]) * fps, (cur[1] - prev[1]) * fps)
        v1 = ((nxt[0] - cur[0]) * fps, (nxt[1] - cur[1]) * fps)
        s0, s1 = math.hypot(*v0), math.hypot(*v1)
        if s0 < min_speed_ms or s1 < min_speed_ms:
            continue
        cosang = (v0[0] * v1[0] + v0[1] * v1[1]) / (s0 * s1)
        if cosang > -0.17:  # less than ~100 degrees of direction change
            continue
        if idx < len(person_frames):
            for tid, team, x, y in person_frames[idx]:
                if team in opponent_teams and _gap(cur, (x, y)) <= radius_m * 1.25:
                    return idx, tid
    return None


def team_retained(
    ball_xy: List[Optional[Tuple[float, float]]],
    person_frames: List[List[PersonRow]],
    team: str,
    start_idx: int,
    n_frames: int,
    radius_m: float = DUEL_CONTROL_RADIUS_M,
    max_gap_frames: int = RETENTION_MAX_GAP_FRAMES,
) -> Tuple[bool, int]:
    """Team possession over a window. Returns (retained, frames_checked).

    Every observed ball frame must sit inside some teammate's control radius;
    a missing-ball run longer than max_gap_frames breaks retention loudly.
    """
    gap_run = 0
    checked = 0
    for idx in range(start_idx, min(start_idx + n_frames, len(ball_xy))):
        ball = ball_xy[idx]
        if ball is None:
            gap_run += 1
            if gap_run > max_gap_frames:
                return False, checked
            continue
        gap_run = 0
        if idx >= len(person_frames):
            return False, checked
        if not any(
            t == team and _gap(ball, (x, y)) <= radius_m
            for _, t, x, y in person_frames[idx]
        ):
            return False, checked
        checked += 1
    return True, checked


def first_opponent_touch(
    ball_xy: List[Optional[Tuple[float, float]]],
    person_frames: List[List[PersonRow]],
    opponent_teams: List[str],
    start_idx: int,
    radius_m: float = DUEL_CONTROL_RADIUS_M,
) -> Optional[Tuple[int, int]]:
    """First (frame, track_id) with an opponent inside control radius."""
    for idx in range(start_idx, len(ball_xy)):
        ball = ball_xy[idx]
        if ball is None or idx >= len(person_frames):
            continue
        best: Optional[Tuple[float, int]] = None
        for tid, team, x, y in person_frames[idx]:
            if team not in opponent_teams:
                continue
            dist = _gap(ball, (x, y))
            if dist <= radius_m and (best is None or dist < best[0]):
                best = (dist, tid)
        if best is not None:
            return idx, best[1]
    return None


def opponent_within_radius(
    ball_xy: List[Optional[Tuple[float, float]]],
    person_frames: List[List[PersonRow]],
    opponent_teams: List[str],
    start_idx: int,
    n_frames: int,
    radius_m: float = DUEL_CONTROL_RADIUS_M,
) -> bool:
    """Any opponent inside control radius during a window: contested control."""
    for idx in range(start_idx, min(start_idx + n_frames, len(ball_xy))):
        ball = ball_xy[idx]
        if ball is None or idx >= len(person_frames):
            continue
        if any(
            t in opponent_teams and _gap(ball, (x, y)) <= radius_m
            for _, t, x, y in person_frames[idx]
        ):
            return True
    return False


def resolve_duel_outcome(
    ball_xy: List[Optional[Tuple[float, float]]],
    ball_gaps_m: List[Optional[float]],
    person_frames: List[List[PersonRow]],
    touch_idx: int,
    receiver_team: str,
    opponent_teams: List[str],
    fps: float = 25.0,
    offside_invalid: bool = False,
    window_s: float = RETENTION_WINDOW_S,
) -> DuelOutcomeResult:
    """Full resolution for one released outlet. Predicted-vs-observed input."""
    window_frames = max(1, int(round(window_s * fps)))
    release = release_frame_idx(ball_gaps_m, touch_idx)
    if release is None:
        return DuelOutcomeResult(DuelOutcome.NO_RELEASE, note="ball never left")
    if release + 1 >= len(ball_xy):
        return DuelOutcomeResult(
            DuelOutcome.TRUNCATED_UNKNOWN,
            release_frame_idx=release,
            note="clip ends at release",
        )
    if offside_invalid:
        return DuelOutcomeResult(DuelOutcome.OFFSIDE_INVALID, release_frame_idx=release)
    control = first_controller(ball_xy, person_frames, release)
    if control is None:
        return DuelOutcomeResult(DuelOutcome.NOT_CONTROLLED, release_frame_idx=release)
    ctrl_idx, ctrl_id, ctrl_team = control
    if ctrl_team in opponent_teams:
        deflect = detect_deflection(
            ball_xy, person_frames, release, ctrl_idx + 1, opponent_teams, fps=fps
        )
        if deflect is not None:
            return DuelOutcomeResult(
                DuelOutcome.BLOCKED,
                controller_track_id=deflect[1],
                control_frame_idx=deflect[0],
                release_frame_idx=release,
                contested=True,
                note="mid-flight deflection",
            )
        return DuelOutcomeResult(
            DuelOutcome.INTERCEPTED,
            controller_track_id=ctrl_id,
            controller_team=ctrl_team,
            control_frame_idx=ctrl_idx,
            release_frame_idx=release,
            contested=True,
        )
    if ctrl_idx + window_frames > len(ball_xy):
        return DuelOutcomeResult(
            DuelOutcome.TRUNCATED_UNKNOWN,
            controller_track_id=ctrl_id,
            controller_team=ctrl_team,
            control_frame_idx=ctrl_idx,
            release_frame_idx=release,
            note="retention window exceeds clip",
        )
    for idx in range(ctrl_idx, min(ctrl_idx + window_frames, len(ball_xy))):
        ball = ball_xy[idx]
        if ball is not None and _out_of_play(ball):
            return DuelOutcomeResult(
                DuelOutcome.OUT_OF_PLAY,
                controller_track_id=ctrl_id,
                controller_team=ctrl_team,
                control_frame_idx=ctrl_idx,
                release_frame_idx=release,
            )
    retained, _ = team_retained(
        ball_xy, person_frames, ctrl_team, ctrl_idx, window_frames
    )
    if retained:
        contested = opponent_within_radius(
            ball_xy, person_frames, opponent_teams, ctrl_idx, max(1, int(fps))
        )
        return DuelOutcomeResult(
            DuelOutcome.CONTESTED_RETAINED if contested else DuelOutcome.RETAINED,
            controller_track_id=ctrl_id,
            controller_team=ctrl_team,
            control_frame_idx=ctrl_idx,
            release_frame_idx=release,
            contested=contested,
        )
    taker = first_opponent_touch(ball_xy, person_frames, opponent_teams, ctrl_idx + 1)
    taker_id = taker[1] if taker is not None else None
    return DuelOutcomeResult(
        DuelOutcome.TACKLED,
        controller_track_id=taker_id,
        control_frame_idx=ctrl_idx,
        release_frame_idx=release,
        contested=True,
        note="possession lost inside retention window",
    )
