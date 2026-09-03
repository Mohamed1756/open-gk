"""
Player Pose and Body Orientation Estimation Engine.
Extracts 17-keypoint skeletons and computes sagittal facing angles
projected into canonical pitch coordinates (meters, radians).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.cv.pitch_homography import PlanarPitchHomography


@dataclass(frozen=True)
class PlayerPoseResult:
    """Estimated body pose keypoints and facing direction."""

    keypoints_xyc: np.ndarray  # Shape (17, 3): [u, v, confidence]
    facing_angle_pitch_rad: float  # Angle relative to pitch +X axis in [-pi, pi]
    confidence: float
    is_valid: bool


def compute_facing_angle_from_keypoints(
    keypoints_xyc: np.ndarray,
    homography: Optional[PlanarPitchHomography] = None,
    frame_idx: Optional[int] = None,
) -> Tuple[float, float, bool]:
    """
    Computes body facing angle in radians relative to canonical pitch +X axis.
    Uses shoulder (5, 6) and hip (11, 12) vectors to derive normal facing direction.
    """
    if keypoints_xyc.shape[0] < 13:
        return 0.0, 0.0, False

    l_sh = keypoints_xyc[5]
    r_sh = keypoints_xyc[6]
    l_hip = keypoints_xyc[11]
    r_hip = keypoints_xyc[12]

    # Check keypoint confidence
    sh_valid = l_sh[2] >= 0.30 and r_sh[2] >= 0.30
    hip_valid = l_hip[2] >= 0.30 and r_hip[2] >= 0.30

    if not sh_valid and not hip_valid:
        return 0.0, 0.0, False

    # Choose primary torso vector (shoulders preferred, fallback to hips)
    if sh_valid and hip_valid:
        torso_vec = 0.6 * (l_sh[:2] - r_sh[:2]) + 0.4 * (l_hip[:2] - r_hip[:2])
        mean_conf = float((l_sh[2] + r_sh[2] + l_hip[2] + r_hip[2]) / 4.0)
    elif sh_valid:
        torso_vec = l_sh[:2] - r_sh[:2]
        mean_conf = float((l_sh[2] + r_sh[2]) / 2.0)
    else:
        torso_vec = l_hip[:2] - r_hip[:2]
        mean_conf = float((l_hip[2] + r_hip[2]) / 2.0)

    # In screen coordinates, normal perpendicular to (dx, dy) pointing forward/down
    screen_facing = np.array([-torso_vec[1], torso_vec[0]], dtype=np.float32)
    norm = float(np.linalg.norm(screen_facing)) + 1e-6
    screen_facing /= norm

    # Anchor point at torso center
    if sh_valid:
        cx = float((l_sh[0] + r_sh[0]) / 2.0)
        cy = float((l_sh[1] + r_sh[1]) / 2.0)
    else:
        cx = float((l_hip[0] + r_hip[0]) / 2.0)
        cy = float((l_hip[1] + r_hip[1]) / 2.0)

    if homography is None:
        raw_angle = math.atan2(float(screen_facing[1]), float(screen_facing[0]))
        return raw_angle, mean_conf, True

    # Project center and front step onto pitch to get true pitch orientation
    p_center = homography.to_pitch(cx, cy, frame_idx=frame_idx)
    step_px = 15.0
    p_front = homography.to_pitch(
        cx + float(screen_facing[0]) * step_px,
        cy + float(screen_facing[1]) * step_px,
        frame_idx=frame_idx,
    )

    dx_m = p_front.x - p_center.x
    dy_m = p_front.y - p_center.y
    facing_angle_rad = math.atan2(dy_m, dx_m)

    return facing_angle_rad, mean_conf, True


class PlayerPoseEstimator:
    """Estimates player skeletal pose and orientation from video frames."""

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

        # Get keypoints in frame coordinates
        kp = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
        kp[:, 0] += cx1
        kp[:, 1] += cy1

        angle_rad, conf, valid = compute_facing_angle_from_keypoints(
            kp, homography=self.homography, frame_idx=frame_idx
        )

        return PlayerPoseResult(
            keypoints_xyc=kp,
            facing_angle_pitch_rad=angle_rad,
            confidence=conf,
            is_valid=valid,
        )
