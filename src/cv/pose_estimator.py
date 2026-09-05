# Justification: Combines 17-keypoint skeleton extraction, torso/head decoupled gaze projection, and temporal smoothing.
"""
Player Pose, Body Orientation, and Gaze Estimation Engine.
Extracts 17-keypoint skeletons and computes decoupled sagittal torso and head-gaze angles
projected into canonical pitch coordinates (meters, radians).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from src.cv.pitch_homography import PlanarPitchHomography


@dataclass(frozen=True)
class PlayerPoseResult:
    """Estimated body pose keypoints, torso orientation, and gaze direction."""

    keypoints_xyc: np.ndarray  # Shape (17, 3): [u, v, confidence]
    facing_angle_pitch_rad: (
        float  # Legacy alias: torso angle in canonical pitch [-pi, pi]
    )
    confidence: float
    is_valid: bool
    torso_facing_rad: float = 0.0  # Biomechanical chest/hip orientation [-pi, pi]
    gaze_facing_rad: float = 0.0  # Perception visual gaze orientation [-pi, pi]
    torso_confidence: float = 0.0
    gaze_confidence: float = 0.0
    gaze_valid: bool = False  # True only with facial evidence; never torso-as-gaze


# Cervical spine physiological range of motion (active neck rotation limit is ~70-80 deg, AAOS)
MAX_CERVICAL_ROTATION_RAD: float = math.radians(75.0)


def wrap_angle_rad(angle_rad: float) -> float:
    """Wraps an angle into [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def circular_weighted_smooth(angles_rad: List[float], weights: List[float]) -> float:
    """
    Computes confidence-weighted circular mean across a sequence of angles in [-pi, pi].
    Formula: atan2(sum(w_i * sin(theta_i)), sum(w_i * cos(theta_i))).
    Derivation: Fisher (1995), Statistical Analysis of Circular Data.
    """
    if not angles_rad or not weights or len(angles_rad) != len(weights):
        return 0.0
    sum_sin = sum(w * math.sin(a) for a, w in zip(angles_rad, weights))
    sum_cos = sum(w * math.cos(a) for a, w in zip(angles_rad, weights))
    if abs(sum_sin) < 1e-6 and abs(sum_cos) < 1e-6:
        return angles_rad[-1]
    return math.atan2(sum_sin, sum_cos)


def _circular_resultant_length(angles_rad: List[float], weights: List[float]) -> float:
    sum_w = sum(weights)
    if sum_w <= 1e-9:
        return 0.0
    sum_sin = sum(w * math.sin(a) for a, w in zip(angles_rad, weights))
    sum_cos = sum(w * math.cos(a) for a, w in zip(angles_rad, weights))
    return math.hypot(sum_sin, sum_cos) / sum_w


def gaze_yaw_abs_error_deg(label_deg: float, pred_rad: float) -> float:
    """Absolute circular error between a rater yaw label and a predicted angle."""
    return abs(math.degrees(wrap_angle_rad(math.radians(label_deg) - pred_rad)))


def head_size_bin(head_px: float) -> str:
    """Resolution bin for gaze accuracy reporting: <8px, 8-15px, >15px."""
    if head_px < 8.0:
        return "<8px"
    if head_px <= 15.0:
        return "8-15px"
    return ">15px"


def binned_mae_deg(records: List[Tuple[str, float]]) -> dict[str, float]:
    """Mean absolute error per bin from (bin, error_deg) records."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for bin_name, err in records:
        totals[bin_name] = totals.get(bin_name, 0.0) + err
        counts[bin_name] = counts.get(bin_name, 0) + 1
    return {b: totals[b] / counts[b] for b in totals}


def inter_rater_mae_deg(pairs: List[Tuple[float, float]]) -> float:
    """Mean absolute circular disagreement between two raters' yaw labels."""
    if not pairs:
        return 0.0
    errs = [
        abs(math.degrees(wrap_angle_rad(math.radians(a) - math.radians(b))))
        for a, b in pairs
    ]
    return sum(errs) / len(errs)


def project_screen_direction_to_pitch(
    screen_dir: np.ndarray,
    ground_uv: Tuple[float, float],
    homography: PlanarPitchHomography,
    frame_idx: Optional[int] = None,
    step_px: float = 5.0,
) -> float:
    """
    Projects a 2D screen direction vector onto canonical pitch coordinates using local differential
    stepping anchored at the player's ground contact point (Z = 0).
    Avoids off-ground chest parallax error inherent in planar homographies.
    """
    u0, v0 = ground_uv
    p_ground = homography.to_pitch(u0, v0, frame_idx=frame_idx)
    # Local epsilon perturbation in screen direction at ground level
    u1 = u0 + float(screen_dir[0]) * step_px
    v1 = v0 + float(screen_dir[1]) * step_px
    p_step = homography.to_pitch(u1, v1, frame_idx=frame_idx)

    dx_m = p_step.x - p_ground.x
    dy_m = p_step.y - p_ground.y
    return math.atan2(dy_m, dx_m)


def disambiguate_torso_normal(
    torso_normal_screen: np.ndarray,
    keypoints_xyc: np.ndarray,
    velocity_screen_uv: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Resolves 180-degree front/back ambiguity of the 2D torso normal and extracts gaze vector.
    Uses facial landmarks (nose relative to ears/shoulders) or motion velocity.

    Returns:
        (disambiguated_torso_normal, gaze_dir_screen, gaze_conf, head_valid)
    """
    nose = keypoints_xyc[0]
    l_eye = keypoints_xyc[1]
    r_eye = keypoints_xyc[2]
    l_ear = keypoints_xyc[3]
    r_ear = keypoints_xyc[4]
    l_sh = keypoints_xyc[5]
    r_sh = keypoints_xyc[6]

    head_valid = False
    gaze_conf = 0.0
    face_dir = np.zeros(2, dtype=np.float32)

    # Strategy 1: Nose vs ears (most direct facial normal)
    ear_valid = l_ear[2] >= 0.25 and r_ear[2] >= 0.25
    if nose[2] >= 0.25 and ear_valid:
        ear_mid = 0.5 * (l_ear[:2] + r_ear[:2])
        face_dir = nose[:2] - ear_mid
        norm = float(np.linalg.norm(face_dir))
        if norm > 1.0:
            face_dir /= norm
            head_valid = True
            gaze_conf = float((nose[2] + l_ear[2] + r_ear[2]) / 3.0)

    # Strategy 2: Nose vs shoulders (fallback if ears partially occluded)
    if not head_valid and nose[2] >= 0.30 and (l_sh[2] >= 0.30 or r_sh[2] >= 0.30):
        if l_sh[2] >= 0.30 and r_sh[2] >= 0.30:
            sh_mid = 0.5 * (l_sh[:2] + r_sh[:2])
        elif l_sh[2] >= 0.30:
            sh_mid = l_sh[:2]
        else:
            sh_mid = r_sh[:2]
        eye_conf = max(l_eye[2], r_eye[2])
        if eye_conf >= 0.25:
            eye_mid = (
                0.5 * (l_eye[:2] + r_eye[:2])
                if (l_eye[2] >= 0.25 and r_eye[2] >= 0.25)
                else (l_eye[:2] if l_eye[2] >= 0.25 else r_eye[:2])
            )
            face_dir = eye_mid - sh_mid
            norm = float(np.linalg.norm(face_dir))
            if norm > 1.0:
                face_dir /= norm
                head_valid = True
                gaze_conf = float((nose[2] + eye_conf) / 2.0)

    normal = np.copy(torso_normal_screen)

    # Disambiguate torso normal with face vector if valid
    if head_valid and float(np.linalg.norm(face_dir)) > 0.5:
        dot = float(np.dot(normal, face_dir))
        if dot < -0.1:
            normal = -normal
        gaze_dir = face_dir
    else:
        # Check velocity disambiguation if player is in motion
        if velocity_screen_uv is not None:
            vx, vy = velocity_screen_uv
            v_norm = math.hypot(vx, vy)
            if v_norm >= 3.0:
                v_dir = np.array([vx / v_norm, vy / v_norm], dtype=np.float32)
                dot = float(np.dot(normal, v_dir))
                if dot < -0.2:
                    normal = -normal
        gaze_dir = normal
        gaze_conf = float(min(1.0, max(nose[2], 0.1)))

    return normal, gaze_dir, gaze_conf, head_valid


def compute_detailed_pose_orientation(
    keypoints_xyc: np.ndarray,
    homography: Optional[PlanarPitchHomography] = None,
    frame_idx: Optional[int] = None,
    ground_uv: Optional[Tuple[float, float]] = None,
    velocity_screen_uv: Optional[Tuple[float, float]] = None,
) -> PlayerPoseResult:
    """
    Computes decoupled torso orientation and gaze direction in canonical pitch coordinates.
    """
    if keypoints_xyc.shape[0] < 13:
        return PlayerPoseResult(
            keypoints_xyc=keypoints_xyc,
            facing_angle_pitch_rad=0.0,
            confidence=0.0,
            is_valid=False,
            torso_facing_rad=0.0,
            gaze_facing_rad=0.0,
            torso_confidence=0.0,
            gaze_confidence=0.0,
        )

    l_sh = keypoints_xyc[5]
    r_sh = keypoints_xyc[6]
    l_hip = keypoints_xyc[11]
    r_hip = keypoints_xyc[12]

    # Check keypoint confidence
    sh_valid = l_sh[2] >= 0.30 and r_sh[2] >= 0.30
    hip_valid = l_hip[2] >= 0.30 and r_hip[2] >= 0.30

    if not sh_valid and not hip_valid:
        return PlayerPoseResult(
            keypoints_xyc=keypoints_xyc,
            facing_angle_pitch_rad=0.0,
            confidence=0.0,
            is_valid=False,
            torso_facing_rad=0.0,
            gaze_facing_rad=0.0,
            torso_confidence=0.0,
            gaze_confidence=0.0,
        )

    # Choose primary torso vector (shoulders preferred, fallback to hips)
    if sh_valid and hip_valid:
        torso_vec = 0.6 * (l_sh[:2] - r_sh[:2]) + 0.4 * (l_hip[:2] - r_hip[:2])
        mean_sh_hip_conf = float((l_sh[2] + r_sh[2] + l_hip[2] + r_hip[2]) / 4.0)
    elif sh_valid:
        torso_vec = l_sh[:2] - r_sh[:2]
        mean_sh_hip_conf = float((l_sh[2] + r_sh[2]) / 2.0)
    else:
        torso_vec = l_hip[:2] - r_hip[:2]
        mean_sh_hip_conf = float((l_hip[2] + r_hip[2]) / 2.0)

    # In screen coordinates, normal perpendicular to (dx, dy)
    raw_screen_normal = np.array([-torso_vec[1], torso_vec[0]], dtype=np.float32)
    norm = float(np.linalg.norm(raw_screen_normal)) + 1e-6
    raw_screen_normal /= norm

    # Disambiguate torso normal and extract gaze vector
    screen_torso, screen_gaze, gaze_conf, head_valid = disambiguate_torso_normal(
        torso_normal_screen=raw_screen_normal,
        keypoints_xyc=keypoints_xyc,
        velocity_screen_uv=velocity_screen_uv,
    )

    # Ground anchor must be at Z=0 to avoid chest-height parallax.
    # Off-ground shoulder/hip midpoints are never valid homography anchors.
    anchor_uv: Optional[Tuple[float, float]] = ground_uv
    if anchor_uv is None and keypoints_xyc.shape[0] >= 17:
        l_ank = keypoints_xyc[15]
        r_ank = keypoints_xyc[16]
        if l_ank[2] >= 0.20 and r_ank[2] >= 0.20:
            anchor_uv = (
                float((l_ank[0] + r_ank[0]) / 2.0),
                float((l_ank[1] + r_ank[1]) / 2.0),
            )

    # Cervical clamp lives in screen space (observation space): clamping after
    # pitch projection would mistake perspective distortion for neck rotation.
    screen_torso_angle = math.atan2(float(screen_torso[1]), float(screen_torso[0]))
    screen_gaze_angle = math.atan2(float(screen_gaze[1]), float(screen_gaze[0]))
    yaw_diff_screen = wrap_angle_rad(screen_gaze_angle - screen_torso_angle)
    if abs(yaw_diff_screen) > MAX_CERVICAL_ROTATION_RAD:
        clamped_yaw = math.copysign(MAX_CERVICAL_ROTATION_RAD, yaw_diff_screen)
        screen_gaze_angle = wrap_angle_rad(screen_torso_angle + clamped_yaw)
        screen_gaze = np.array(
            [math.cos(screen_gaze_angle), math.sin(screen_gaze_angle)],
            dtype=np.float32,
        )

    if homography is None or anchor_uv is None:
        torso_angle = screen_torso_angle
        gaze_angle = screen_gaze_angle
    else:
        torso_angle = project_screen_direction_to_pitch(
            screen_dir=screen_torso,
            ground_uv=anchor_uv,
            homography=homography,
            frame_idx=frame_idx,
        )
        gaze_angle = project_screen_direction_to_pitch(
            screen_dir=screen_gaze,
            ground_uv=anchor_uv,
            homography=homography,
            frame_idx=frame_idx,
        )

    return PlayerPoseResult(
        keypoints_xyc=keypoints_xyc,
        facing_angle_pitch_rad=torso_angle,
        confidence=mean_sh_hip_conf,
        is_valid=True,
        torso_facing_rad=torso_angle,
        gaze_facing_rad=gaze_angle,
        torso_confidence=mean_sh_hip_conf,
        gaze_confidence=gaze_conf,
        gaze_valid=head_valid,
    )


def compute_facing_angle_from_keypoints(
    keypoints_xyc: np.ndarray,
    homography: Optional[PlanarPitchHomography] = None,
    frame_idx: Optional[int] = None,
    ground_uv: Optional[Tuple[float, float]] = None,
    velocity_screen_uv: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, bool]:
    """
    Computes body facing angle in radians relative to canonical pitch +X axis.
    Uses shoulder (5, 6) and hip (11, 12) vectors with facial landmark disambiguation.
    Retains backward-compatible 3-tuple return signature (facing_angle_rad, confidence, is_valid).
    """
    res = compute_detailed_pose_orientation(
        keypoints_xyc=keypoints_xyc,
        homography=homography,
        frame_idx=frame_idx,
        ground_uv=ground_uv,
        velocity_screen_uv=velocity_screen_uv,
    )
    return res.torso_facing_rad, res.torso_confidence, res.is_valid


class PlayerPoseEstimator:
    """Estimates player skeletal pose, torso orientation, and gaze direction from video frames."""

    def __init__(
        self,
        weights_path: str = "yolov8n-pose.pt",
        homography: Optional[PlanarPitchHomography] = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.homography = homography

    def estimate_pose_in_crop(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
        frame_idx: Optional[int] = None,
        velocity_screen_uv: Optional[Tuple[float, float]] = None,
    ) -> PlayerPoseResult:
        """Runs pose estimation on a cropped bounding box region."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        # Add generous margin for full skeletal context
        w_box = x2 - x1
        h_box = y2 - y1
        margin_x = max(30.0, w_box * 0.6)
        margin_y = max(30.0, h_box * 0.4)
        cx1 = max(0, int(x1 - margin_x))
        cy1 = max(0, int(y1 - margin_y))
        cx2 = min(w, int(x2 + margin_x))
        cy2 = min(h, int(y2 + margin_y))

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0 or (cx2 - cx1) < 8 or (cy2 - cy1) < 8:
            return PlayerPoseResult(
                keypoints_xyc=np.zeros((17, 3), dtype=np.float32),
                facing_angle_pitch_rad=0.0,
                confidence=0.0,
                is_valid=False,
            )

        results = self.model(crop, conf=0.15, verbose=False)
        if not results or len(results[0].keypoints) == 0:
            return PlayerPoseResult(
                keypoints_xyc=np.zeros((17, 3), dtype=np.float32),
                facing_angle_pitch_rad=0.0,
                confidence=0.0,
                is_valid=False,
            )

        # Get keypoints in full frame coordinates
        kp = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
        kp[:, 0] += cx1
        kp[:, 1] += cy1

        # Ground contact point at bottom center of bounding box
        ground_uv = (float((x1 + x2) / 2.0), float(y2))

        return compute_detailed_pose_orientation(
            keypoints_xyc=kp,
            homography=self.homography,
            frame_idx=frame_idx,
            ground_uv=ground_uv,
            velocity_screen_uv=velocity_screen_uv,
        )

    def smooth_pose_sequence(
        self,
        pose_results: List[PlayerPoseResult],
        half_life_frames: float = 4.0,
    ) -> PlayerPoseResult:
        """
        Applies confidence-weighted temporal circular smoothing over a sequence of pose results.
        Uses exponential recency weights so recent frames carry higher weight.
        """
        valid_poses = [p for p in pose_results if p.is_valid]
        if not valid_poses:
            return PlayerPoseResult(
                keypoints_xyc=np.zeros((17, 3), dtype=np.float32),
                facing_angle_pitch_rad=0.0,
                confidence=0.0,
                is_valid=False,
            )
        if len(valid_poses) == 1:
            return valid_poses[0]

        n = len(valid_poses)
        # Recency weights: w_i = conf_i * exp(-(n - 1 - i) / half_life_frames)
        torso_angles = [p.torso_facing_rad for p in valid_poses]
        torso_weights = [
            p.torso_confidence * math.exp(-(n - 1 - i) / half_life_frames)
            for i, p in enumerate(valid_poses)
        ]
        smoothed_torso = circular_weighted_smooth(torso_angles, torso_weights)
        if _circular_resultant_length(torso_angles, torso_weights) < 0.3:
            smoothed_torso = valid_poses[-1].torso_facing_rad

        gaze_sources = [p for p in valid_poses if p.gaze_valid]
        if gaze_sources:
            m = len(gaze_sources)
            gaze_angles = [p.gaze_facing_rad for p in gaze_sources]
            gaze_weights = [
                p.gaze_confidence * math.exp(-(m - 1 - i) / half_life_frames)
                for i, p in enumerate(gaze_sources)
            ]
            smoothed_gaze = circular_weighted_smooth(gaze_angles, gaze_weights)
            if _circular_resultant_length(gaze_angles, gaze_weights) < 0.3:
                smoothed_gaze = gaze_sources[-1].gaze_facing_rad
            smoothed_gaze_conf = float(
                np.mean([p.gaze_confidence for p in gaze_sources])
            )
            smoothed_gaze_valid = True
        else:
            smoothed_gaze = valid_poses[-1].torso_facing_rad
            smoothed_gaze_conf = 0.0
            smoothed_gaze_valid = False

        last = valid_poses[-1]
        mean_conf = float(np.mean([p.confidence for p in valid_poses]))

        return PlayerPoseResult(
            keypoints_xyc=last.keypoints_xyc,
            facing_angle_pitch_rad=smoothed_torso,
            confidence=mean_conf,
            is_valid=True,
            torso_facing_rad=smoothed_torso,
            gaze_facing_rad=smoothed_gaze,
            torso_confidence=float(np.mean([p.torso_confidence for p in valid_poses])),
            gaze_confidence=smoothed_gaze_conf,
            gaze_valid=smoothed_gaze_valid,
        )
