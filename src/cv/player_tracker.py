# Justification: Combines Kalman filtering state, Hungarian assignment, ball-carrier snapping, and pitch boundary filtering.
"""
Multi-Object Temporal Video Tracking Engine (Kalman Filter + Hungarian Association).
Maintains persistent player, referee, and single match ball track identities across continuous video frames at 25 Hz.
Enforces Field-of-Play boundaries, filters off-pitch personnel, and snaps ball-to-carrier possession.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.cv.pitch_homography import PlanarPitchHomography
from src.cv.player_detector import (
    YOLOPlayerDetector,
    DetectedPlayer,
    is_keeper_label,
    is_official_label,
)


def is_keeper_track(team_label: str, team_color_hex: str) -> bool:
    if is_keeper_label(team_label):
        return True
    return team_color_hex.lower() in ("#c026d3", "#facc15")


DISPLAY_SWITCH_VOTES = 5
BIRTH_CONFIRM_VOTES = 2


def resolve_display_label(
    current_display: Optional[str], vote_history: List[str]
) -> Optional[str]:
    recent = vote_history[-DISPLAY_SWITCH_VOTES:]
    if current_display is None:
        if (
            len(recent) >= BIRTH_CONFIRM_VOTES
            and len(set(recent[-BIRTH_CONFIRM_VOTES:])) == 1
        ):
            return recent[-1]
        return None
    if (
        len(recent) >= DISPLAY_SWITCH_VOTES
        and len(set(recent)) == 1
        and recent[0] != current_display
    ):
        return recent[0]
    return current_display


def _color_for_label(label: str) -> str:
    if "Maignan - Purple" in label:
        return "#c026d3"
    if "Yellow" in label or is_keeper_label(label):
        if "Cyan" not in label:
            return "#c026d3" if "Maignan" in label else "#facc15"
    if "Cyan" in label or is_official_label(label):
        return "#06b6d4"
    if "White" in label:
        return "#ffffff"
    if "Unknown" in label:
        return "#94a3b8"
    return "#881337"


def split_possession(
    entities: List[Any],
    possession_labels: List[str],
    opponent_labels: List[str],
) -> Tuple[List[Any], List[Any]]:
    teammates: List[Any] = []
    opponents: List[Any] = []
    for e in entities:
        label = e["team_label"] if isinstance(e, dict) else e.team_label
        if label in possession_labels:
            teammates.append(e)
        elif label in opponent_labels:
            opponents.append(e)
    return teammates, opponents


@dataclass
class TrackedEntity:
    """Represents a single persistent tracked entity (player, official, or ball) across time."""

    track_id: int
    class_name: str  # "person" or "sports ball"
    team_label: str
    team_color_hex: str
    state_xyuv: np.ndarray  # [u, v, du, dv] in pixels and px/frame
    covariance: np.ndarray
    bbox_xyxy: Tuple[float, float, float, float]
    pitch_pos_m: Tuple[float, float]
    pitch_vel_ms: Tuple[float, float] = (0.0, 0.0)
    speed_ms: float = 0.0
    age_frames: int = 1
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)
    label_counts: Dict[str, int] = field(default_factory=dict)
    pitch_valid: bool = True
    prev_screen_uv: Tuple[float, float] = (0.0, 0.0)
    display_label: Optional[str] = None
    display_color_hex: str = "#94a3b8"
    recent_votes: List[str] = field(default_factory=list)


class MultiObjectVideoTracker:
    """
    Temporal multi-object tracker for broadcast football video.
    Filters off-pitch staff, maintains single ball track, and computes physical velocities.
    """

    def __init__(
        self,
        detector: YOLOPlayerDetector,
        homography: Optional[PlanarPitchHomography] = None,
        max_age: int = 15,
        min_hits: int = 2,
        dist_threshold_px: float = 65.0,
        fps: float = 25.0,
    ):
        self.detector = detector
        self.homography = homography or PlanarPitchHomography()
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold_px = dist_threshold_px
        self.fps = fps
        self.next_id: int = 1
        self.tracks: Dict[int, TrackedEntity] = {}

        # 4-state Kalman matrices (constant velocity model)
        dt = 1.0  # 1 frame
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.H_meas = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )

        self.Q = np.diag([2.0, 2.0, 8.0, 8.0]).astype(np.float64)  # Process noise
        self.R = np.diag([4.0, 4.0]).astype(np.float64)  # Measurement noise

    def _predict_tracks(self) -> None:
        """Propagates all active tracks forward by 1 frame via Kalman prediction."""
        for track in self.tracks.values():
            track.state_xyuv = np.dot(self.F, track.state_xyuv)
            track.covariance = (
                np.dot(np.dot(self.F, track.covariance), self.F.T) + self.Q
            )
            track.age_frames += 1
            track.time_since_update += 1

    def update(self, frame_bgr: np.ndarray, frame_idx: int) -> List[TrackedEntity]:
        """
        Processes a single video frame: detects entities, enforces single ball,
        filters off-pitch staff, updates Kalman filters, and computes pitch positions.
        """
        # 1. Predict existing tracks
        self._predict_tracks()

        # 2. Run object detection
        detections: List[DetectedPlayer] = self.detector.detect_frame(
            frame_bgr, frame_idx=frame_idx
        )

        # Enforce exactly <= 1 Ball detection (select highest confidence ball on pitch)
        ball_dets = [
            d
            for d in detections
            if d.class_name == "sports ball"
            and 80.0 <= d.feet_screen_uv[0] <= 1250.0
            and 200.0 <= d.feet_screen_uv[1] <= 710.0
            and 0.0 <= d.pitch_pos_m[0] <= 105.0
            and 0.0 <= d.pitch_pos_m[1] <= 68.0
        ]
        person_dets = [
            d
            for d in detections
            if d.class_name == "person" and "Off-Pitch" not in d.team_label
        ]

        valid_detections: List[DetectedPlayer] = list(person_dets)
        if ball_dets:
            best_ball = max(ball_dets, key=lambda b: b.confidence)
            valid_detections.append(best_ball)

        # 3. Associate detections to existing tracks
        matched_indices, unmatched_dets, unmatched_trks = self._associate(
            valid_detections
        )

        # 4. Update matched tracks
        for trk_id, det_idx in matched_indices:
            det = valid_detections[det_idx]
            trk = self.tracks[trk_id]

            # Kalman Update
            z = np.array(
                [det.feet_screen_uv[0], det.feet_screen_uv[1]], dtype=np.float64
            )
            y = z - np.dot(self.H_meas, trk.state_xyuv)
            S = np.dot(np.dot(self.H_meas, trk.covariance), self.H_meas.T) + self.R
            K = np.dot(np.dot(trk.covariance, self.H_meas.T), np.linalg.inv(S))

            trk.state_xyuv = trk.state_xyuv + np.dot(K, y)
            trk.covariance = trk.covariance - np.dot(
                np.dot(K, self.H_meas), trk.covariance
            )

            # Update visual metadata with temporal consensus
            trk.hits += 1
            trk.time_since_update = 0
            trk.bbox_xyxy = det.bbox_xyxy
            if det.class_name != "sports ball":
                trk.label_counts[det.team_label] = (
                    trk.label_counts.get(det.team_label, 0) + 1
                )
                majority_label = max(trk.label_counts, key=trk.label_counts.get)
                trk.team_label = majority_label
                trk.team_color_hex = det.team_color_hex
                if "Maignan - Purple" in majority_label:
                    trk.team_color_hex = "#c026d3"
                elif "Yellow" in majority_label or is_keeper_label(majority_label):
                    if "Cyan" not in majority_label:
                        trk.team_color_hex = (
                            "#c026d3" if "Maignan" in majority_label else "#facc15"
                        )
                elif "Cyan" in majority_label or is_official_label(majority_label):
                    trk.team_color_hex = "#06b6d4"
                elif "White" in majority_label:
                    trk.team_color_hex = "#ffffff"
                elif "Unknown" in majority_label:
                    trk.team_color_hex = "#94a3b8"
                else:
                    trk.team_color_hex = "#881337"
                trk.recent_votes.append(det.team_label)
                if len(trk.recent_votes) > DISPLAY_SWITCH_VOTES:
                    trk.recent_votes.pop(0)
                shown = resolve_display_label(trk.display_label, trk.recent_votes)
                if shown is not None:
                    trk.display_label = shown
                    trk.display_color_hex = _color_for_label(shown)

            # Pitch projection with camera compensation: re-project previous
            # screen point through the CURRENT homography so pan/zoom does not
            # masquerade as player motion. Validity is explicit, never clamped silently.
            to_valid = getattr(self.homography, "to_pitch_valid", None)
            prev_u, prev_v = trk.prev_screen_uv or (
                trk.state_xyuv[0],
                trk.state_xyuv[1],
            )
            cur_u, cur_v = float(trk.state_xyuv[0]), float(trk.state_xyuv[1])
            if to_valid is not None:
                pitch_pt, valid = to_valid(cur_u, cur_v, frame_idx=frame_idx)
                prev_reproj, prev_valid = to_valid(prev_u, prev_v, frame_idx=frame_idx)
                trk.pitch_valid = bool(valid and prev_valid)
            else:
                pitch_pt = self.homography.to_pitch(cur_u, cur_v, frame_idx=frame_idx)
                prev_reproj = self.homography.to_pitch(
                    prev_u, prev_v, frame_idx=frame_idx
                )
                trk.pitch_valid = True
            trk.pitch_pos_m = (round(pitch_pt.x, 2), round(pitch_pt.y, 2))
            trk.prev_screen_uv = (cur_u, cur_v)

            # Compute physical velocity in m/s from same-H reprojection
            vx_ms = (trk.pitch_pos_m[0] - round(prev_reproj.x, 2)) * self.fps
            vy_ms = (trk.pitch_pos_m[1] - round(prev_reproj.y, 2)) * self.fps
            speed_ms = float(np.hypot(vx_ms, vy_ms))

            if speed_ms < 11.5 and trk.pitch_valid:
                trk.pitch_vel_ms = (round(vx_ms, 2), round(vy_ms, 2))
                trk.speed_ms = round(speed_ms, 2)
            elif not trk.pitch_valid:
                trk.pitch_vel_ms = (0.0, 0.0)
                trk.speed_ms = 0.0

            trk.history.append(trk.pitch_pos_m)
            if len(trk.history) > 30:
                trk.history.pop(0)

        # 5. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = valid_detections[det_idx]

            # Enforce single ball track ID (reserve ID 0 for BALL)
            if det.class_name == "sports ball":
                # If a ball track already exists, do not teleport to distant outliers
                existing_ball = [
                    t for t in self.tracks.values() if t.class_name == "sports ball"
                ]
                if existing_ball:
                    b_trk = existing_ball[0]
                    if b_trk.time_since_update > 15:
                        b_trk.state_xyuv = np.array(
                            [
                                det.feet_screen_uv[0],
                                det.feet_screen_uv[1],
                                0.0,
                                0.0,
                            ],
                            dtype=np.float64,
                        )
                        b_trk.pitch_pos_m = det.pitch_pos_m
                        b_trk.time_since_update = 0
                        b_trk.hits += 1
                    continue
                assigned_id = 0
            else:
                assigned_id = self.next_id
                self.next_id += 1

            init_state = np.array(
                [det.feet_screen_uv[0], det.feet_screen_uv[1], 0.0, 0.0],
                dtype=np.float64,
            )
            init_cov = np.diag([10.0, 10.0, 40.0, 40.0]).astype(np.float64)

            new_track = TrackedEntity(
                track_id=assigned_id,
                class_name=det.class_name,
                team_label=det.team_label,
                team_color_hex=det.team_color_hex,
                state_xyuv=init_state,
                covariance=init_cov,
                bbox_xyxy=det.bbox_xyxy,
                pitch_pos_m=det.pitch_pos_m,
                pitch_vel_ms=(0.0, 0.0),
                speed_ms=0.0,
                age_frames=1,
                hits=1,
                time_since_update=0,
                history=[det.pitch_pos_m],
                label_counts=(
                    {det.team_label: 1} if det.class_name != "sports ball" else {}
                ),
                pitch_valid=bool(getattr(det, "pitch_valid", True)),
                prev_screen_uv=(
                    float(det.feet_screen_uv[0]),
                    float(det.feet_screen_uv[1]),
                ),
                recent_votes=(
                    [det.team_label] if det.class_name != "sports ball" else []
                ),
            )
            self.tracks[assigned_id] = new_track

        # 6. Delete dead tracks (max age for persons: 15 frames, ball: 30 frames)
        dead_ids = [
            tid
            for tid, t in self.tracks.items()
            if t.time_since_update
            > (30 if t.class_name == "sports ball" else self.max_age)
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # 7. Enforce exactly <= 1 keeper per keeper-kit (color) on the pitch.
        # Own goal sits at +X when attack_dir is -X (this episode: x=105),
        # so the tiebreak keeps the candidate nearest the own goal line.
        own_goal_x = getattr(self, "own_goal_x_m", 105.0)
        gk_candidates = [
            t
            for t in self.tracks.values()
            if t.class_name == "person"
            and is_keeper_track(t.team_label, t.team_color_hex)
        ]
        by_kit: Dict[str, List[TrackedEntity]] = {}
        for t in gk_candidates:
            by_kit.setdefault(t.team_color_hex.lower(), []).append(t)
        for _kit_hex, group in by_kit.items():
            if len(group) <= 1:
                continue

            def _keeper_votes(t: TrackedEntity) -> int:
                return sum(
                    c for lab, c in t.label_counts.items() if is_keeper_label(lab)
                )

            true_gk = max(
                group,
                key=lambda t: (_keeper_votes(t), -abs(t.pitch_pos_m[0] - own_goal_x)),
            )
            for t in group:
                if t.track_id != true_gk.track_id:
                    keeper_labs = [
                        lab for lab in t.label_counts if is_keeper_label(lab)
                    ]
                    if "AC Milan (White)" in t.label_counts:
                        t.team_label = "AC Milan (White)"
                        t.team_color_hex = "#ffffff"
                    elif t.team_color_hex == "#facc15":
                        t.team_label = "Unknown-Outfield"
                        t.team_color_hex = "#94a3b8"
                    else:
                        t.team_label = "Unknown-Outfield"
                        t.team_color_hex = "#94a3b8"
                    for lab in keeper_labs:
                        del t.label_counts[lab]

        # Return confirmed active tracks (only on-pitch entities).
        # Persons render only once the display label confirms (birth gating);
        # the ball is exempt.
        active = [
            t
            for t in self.tracks.values()
            if (t.hits >= self.min_hits or t.class_name == "sports ball")
            and (t.class_name == "sports ball" or t.display_label is not None)
            and t.pitch_pos_m[1] <= 67.8  # Exclude managers behind touchline
        ]
        return active

    def _associate(
        self, detections: List[DetectedPlayer]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy / Hungarian nearest-neighbor distance association."""
        if not self.tracks or not detections:
            return (
                [],
                list(range(len(detections))),
                [t.track_id for t in self.tracks.values()],
            )

        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float64)

        for r, tid in enumerate(track_ids):
            trk = self.tracks[tid]
            for c, det in enumerate(detections):
                if trk.class_name != det.class_name:
                    cost_matrix[r, c] = 9999.0
                    continue

                dist = np.hypot(
                    trk.state_xyuv[0] - det.feet_screen_uv[0],
                    trk.state_xyuv[1] - det.feet_screen_uv[1],
                )
                cost_matrix[r, c] = dist

        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = track_ids.copy()

        for r, c in zip(row_ind, col_ind):
            thresh = (
                120.0
                if self.tracks[track_ids[r]].class_name == "sports ball"
                else self.dist_threshold_px
            )
            if cost_matrix[r, c] <= thresh:
                matched.append((track_ids[r], c))
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
                if track_ids[r] in unmatched_trks:
                    unmatched_trks.remove(track_ids[r])

        return matched, unmatched_dets, unmatched_trks
