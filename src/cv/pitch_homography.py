# Justification: Combines 3x3 homography matrix solving, pitch landmark reprojection, dynamic smoothing, and tactical AR polygon transforms.
"""
Planar Pitch Homography Engine for Monocular Video Broadcast Ingestion.
Computes the 3x3 Projective Homography Matrix H mapping screen pixels (u, v) -> Canonical Pitch Meters (X, Y).
Provides inverse projection H^-1 for Augmented Reality tactical trajectory overlays.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.core.geometry import PitchPoint


@dataclass(frozen=True)
class CalibrationKeyframe:
    timestamp_s: float
    screen_points: List[Tuple[float, float]]
    pitch_points: List[Tuple[float, float]]


@dataclass(frozen=True)
class PitchKeypoint:
    """A known spatial landmark on a football pitch."""

    name: str
    pitch_x_m: float  # Canonical X (0 to 105m)
    pitch_y_m: float  # Canonical Y (0 to 68m)


# Standard FIFA 105m x 68m Keypoints
STANDARD_PITCH_KEYPOINTS: Dict[str, PitchKeypoint] = {
    # Own Penalty Box (Left Side)
    "OWN_GOAL_CENTER": PitchKeypoint("OWN_GOAL_CENTER", 0.0, 34.0),
    "OWN_PENALTY_SPOT": PitchKeypoint("OWN_PENALTY_SPOT", 11.0, 34.0),
    "OWN_BOX_TOP_LEFT": PitchKeypoint("OWN_BOX_TOP_LEFT", 16.5, 54.16),
    "OWN_BOX_BOTTOM_LEFT": PitchKeypoint("OWN_BOX_BOTTOM_LEFT", 16.5, 13.84),
    "OWN_SIX_YARD_TOP": PitchKeypoint("OWN_SIX_YARD_TOP", 5.5, 43.16),
    "OWN_SIX_YARD_BOTTOM": PitchKeypoint("OWN_SIX_YARD_BOTTOM", 5.5, 24.84),
    "OWN_CORNER_TOP": PitchKeypoint("OWN_CORNER_TOP", 0.0, 68.0),
    "OWN_CORNER_BOTTOM": PitchKeypoint("OWN_CORNER_BOTTOM", 0.0, 0.0),
    # Center Pitch
    "CENTER_CIRCLE_CENTER": PitchKeypoint("CENTER_CIRCLE_CENTER", 52.5, 34.0),
    "CENTER_LINE_TOP": PitchKeypoint("CENTER_LINE_TOP", 52.5, 68.0),
    "CENTER_LINE_BOTTOM": PitchKeypoint("CENTER_LINE_BOTTOM", 52.5, 0.0),
}


def compute_homography_matrix(
    screen_points: List[Tuple[float, float]],
    pitch_points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Computes the 3x3 Planar Homography matrix H using Direct Linear Transformation (DLT)
    with OpenCV RANSAC optimization for robust multi-point calibration.
    s * [X, Y, 1]^T = H * [u, v, 1]^T
    """
    if len(screen_points) < 4 or len(pitch_points) < 4:
        raise ValueError(
            "At least 4 point correspondences required for Homography computation."
        )

    num_pts = min(len(screen_points), len(pitch_points))
    try:
        import cv2

        src = np.array(screen_points[:num_pts], dtype=np.float32)
        dst = np.array(pitch_points[:num_pts], dtype=np.float32)
        method = cv2.RANSAC if num_pts > 4 else 0
        H_cv, _ = cv2.findHomography(src, dst, method, 2.5)
        if H_cv is not None and not np.isnan(H_cv).any():
            if abs(H_cv[2, 2]) > 1e-8:
                H_cv = H_cv / H_cv[2, 2]
            return H_cv.astype(np.float64)
    except Exception:
        pass

    A: List[List[float]] = []

    for i in range(num_pts):
        u, v = screen_points[i]
        X, Y = pitch_points[i]

        # Row 1: -u, -v, -1, 0, 0, 0, u*X, v*X, X
        A.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * X, v * X, X])
        # Row 2: 0, 0, 0, -u, -v, -1, u*Y, v*Y, Y
        A.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * Y, v * Y, Y])

    A_mat = np.array(A, dtype=np.float64)

    # Solve via Singular Value Decomposition (SVD)
    _, _, Vt = np.linalg.svd(A_mat)
    H = Vt[-1].reshape((3, 3))

    if abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]

    return H.astype(np.float64)


FULL_PITCH_ANCHORS_M: List[List[float]] = [
    [0.0, 0.0],
    [0.0, 68.0],
    [52.5, 68.0],
    [52.5, 0.0],
    [105.0, 68.0],
    [105.0, 0.0],
    [16.5, 13.84],
    [88.5, 54.16],
]

CALIBRATION_HULL_MARGIN_M = 8.0


def project_screen_to_pitch_raw(
    u: float, v: float, H: np.ndarray
) -> Tuple[float, float]:
    screen_vec = np.array([u, v, 1.0], dtype=np.float64)
    pitch_vec = np.dot(H, screen_vec)
    w = 1e-8 if abs(pitch_vec[2]) < 1e-8 else pitch_vec[2]
    return float(pitch_vec[0] / w), float(pitch_vec[1] / w)


def is_inside_pitch_hull(x_m: float, y_m: float, margin_m: float = 0.0) -> bool:
    return -margin_m <= x_m <= 105.0 + margin_m and -margin_m <= y_m <= 68.0 + margin_m


def project_screen_to_pitch_valid(
    u: float, v: float, H: np.ndarray, margin_m: float = CALIBRATION_HULL_MARGIN_M
) -> Tuple[PitchPoint, bool]:
    x_m, y_m = project_screen_to_pitch_raw(u, v, H)
    valid = is_inside_pitch_hull(x_m, y_m, margin_m=margin_m)
    x_clamped = max(0.0, min(105.0, x_m))
    y_clamped = max(0.0, min(68.0, y_m))
    return PitchPoint(x=round(x_clamped, 2), y=round(y_clamped, 2)), valid


def project_screen_to_pitch(u: float, v: float, H: np.ndarray) -> PitchPoint:
    """
    Projects a 2D screen pixel coordinate (u, v) -> Real-world pitch coordinate (X, Y) in meters.
    """
    pt, _ = project_screen_to_pitch_valid(u, v, H)
    return pt


def project_pitch_to_screen(
    x_m: float, y_m: float, H: np.ndarray, z_m: float = 0.0
) -> Tuple[int, int]:
    """
    Projects a real-world pitch coordinate (X, Y, Z) -> Screen pixel (u, v) using H^-1.
    """
    H_inv = np.linalg.inv(H)
    pitch_vec = np.array([x_m, y_m, 1.0], dtype=np.float64)
    screen_vec = np.dot(H_inv, pitch_vec)

    w = 1e-8 if abs(screen_vec[2]) < 1e-8 else screen_vec[2]
    u = screen_vec[0] / w
    v = (screen_vec[1] / w) - (z_m * 18.0)

    return int(round(u)), int(round(v))


class PlanarPitchHomography:
    """
    Manages pitch calibration and Augmented Reality tactical trajectory overlays.
    """

    def __init__(self, H: Optional[np.ndarray] = None):
        if H is not None:
            self.H = H
        else:
            # Canonical 720p/1080p Broadcast Camera Homography
            screen_pts = [
                (120.0, 680.0),  # Own Corner Bottom
                (320.0, 180.0),  # Own Corner Top
                (1220.0, 190.0),  # Center Line Top
                (1180.0, 690.0),  # Center Line Bottom
            ]
            pitch_pts = [
                (0.0, 0.0),
                (0.0, 68.0),
                (52.5, 68.0),
                (52.5, 0.0),
            ]
            self.H = compute_homography_matrix(screen_pts, pitch_pts)

    def to_pitch(
        self, u: float, v: float, frame_idx: Optional[int] = None
    ) -> PitchPoint:
        return project_screen_to_pitch(u, v, self.H)

    def to_screen(
        self, x_m: float, y_m: float, z_m: float = 0.0, frame_idx: Optional[int] = None
    ) -> Tuple[int, int]:
        return project_pitch_to_screen(x_m, y_m, self.H, z_m=z_m)

    def generate_ar_trajectory_overlay(
        self,
        gk_pt: PitchPoint,
        target_pt: PitchPoint,
        striker_pt: PitchPoint,
        swerve_lateral_m: float = 2.10,
        flight_time_s: float = 1.8,
        num_points: int = 25,
    ) -> Dict[str, Any]:
        """
        Generates 3D Augmented Reality trajectory curve coordinates for video overlay.
        """
        trajectory_screen_pts: List[Tuple[int, int]] = []
        danger_zone_screen_pts: List[Tuple[int, int]] = []

        # 1. 3D Parabolic Flight Curve with Magnus Lateral Swerve
        for i in range(num_points):
            t_frac = i / (num_points - 1)
            h_max = 3.2 if flight_time_s > 1.2 else 0.4
            z = 4.0 * h_max * t_frac * (1.0 - t_frac)

            perp_x = -(target_pt.y - gk_pt.y)
            perp_y = target_pt.x - gk_pt.x
            norm_perp = math.hypot(perp_x, perp_y) + 1e-6
            perp_unit_x = perp_x / norm_perp
            perp_unit_y = perp_y / norm_perp

            lat_disp = 4.0 * swerve_lateral_m * t_frac * (1.0 - t_frac)

            x = gk_pt.x + (target_pt.x - gk_pt.x) * t_frac + perp_unit_x * lat_disp
            y = gk_pt.y + (target_pt.y - gk_pt.y) * t_frac + perp_unit_y * lat_disp

            u, v = self.to_screen(x, y, z_m=z)
            trajectory_screen_pts.append((u, v))

        # 2. Striker Danger Reach Circle (1.25m ground radius)
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            dx = striker_pt.x + 1.25 * math.cos(rad)
            dy = striker_pt.y + 1.25 * math.sin(rad)
            u_d, v_d = self.to_screen(dx, dy, z_m=0.0)
            danger_zone_screen_pts.append((u_d, v_d))

        return {
            "flight_trajectory_pixels": trajectory_screen_pts,
            "striker_danger_pixels": danger_zone_screen_pts,
            "gk_screen_pixel": self.to_screen(gk_pt.x, gk_pt.y, z_m=0.0),
            "target_screen_pixel": self.to_screen(target_pt.x, target_pt.y, z_m=0.0),
            "striker_screen_pixel": self.to_screen(striker_pt.x, striker_pt.y, z_m=0.0),
            "magnus_swerve_m": swerve_lateral_m,
            "flight_time_s": flight_time_s,
        }


def save_calibration_json(
    path: str,
    keyframes: List[CalibrationKeyframe],
    match_id: str = "",
    fps: float = 25.0,
) -> None:
    import json

    payload = {
        "match_id": match_id,
        "fps": fps,
        "keyframes": [
            {
                "timestamp_s": k.timestamp_s,
                "screen_points": [list(p) for p in k.screen_points],
                "pitch_points": [list(p) for p in k.pitch_points],
            }
            for k in keyframes
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_calibration_json(
    path: str,
) -> Tuple[List[CalibrationKeyframe], Dict[str, Any]]:
    import json

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    keyframes = [
        CalibrationKeyframe(
            timestamp_s=float(k["timestamp_s"]),
            screen_points=[tuple(p) for p in k["screen_points"]],
            pitch_points=[tuple(p) for p in k["pitch_points"]],
        )
        for k in payload.get("keyframes", [])
    ]
    if not keyframes:
        raise ValueError(f"No keyframes in calibration file: {path}")
    return keyframes, payload


def homographies_from_keyframes(
    keyframes: List[CalibrationKeyframe],
) -> Dict[float, np.ndarray]:
    out: Dict[float, np.ndarray] = {}
    for k in sorted(keyframes, key=lambda e: e.timestamp_s):
        out[float(k.timestamp_s)] = compute_homography_matrix(
            list(k.screen_points), list(k.pitch_points)
        )
    return out


class DynamicPitchHomography(PlanarPitchHomography):
    """
    Manages frame-varying homography matrices to compensate for moving broadcast cameras (pan/tilt/zoom).
    Interpolates smoothly between keyframe homographies using cosine smoothstep transitions.
    Keyframes are keyed by timestamp seconds (fps-independent); frame indices are
    converted via fps when provided.
    """

    def __init__(
        self,
        keyframe_homographies: Dict[float, np.ndarray],
        default_H: Optional[np.ndarray] = None,
        fps: float = 25.0,
    ):
        if not keyframe_homographies:
            raise ValueError("keyframe_homographies cannot be empty.")
        self.keyframe_homographies: Dict[float, np.ndarray] = dict(
            sorted(keyframe_homographies.items())
        )
        first_h = next(iter(self.keyframe_homographies.values()))
        super().__init__(H=default_H if default_H is not None else first_h)
        self._sorted_keys = sorted(self.keyframe_homographies.keys())
        self.fps = float(fps)

    @classmethod
    def from_calibration_file(
        cls, path: str, default_timestamp_s: Optional[float] = None
    ) -> DynamicPitchHomography:
        keyframes, payload = load_calibration_json(path)
        fps = float(payload.get("fps", 25.0))
        homos = homographies_from_keyframes(keyframes)
        default_H = None
        if default_timestamp_s is not None:
            tmp = cls(keyframe_homographies=homos, fps=fps)
            default_H = tmp.get_H_by_time(default_timestamp_s)
        return cls(keyframe_homographies=homos, default_H=default_H, fps=fps)

    def get_H_by_time(self, timestamp_s: Optional[float] = None) -> np.ndarray:
        if timestamp_s is None:
            return self.H
        if timestamp_s <= self._sorted_keys[0]:
            return self.keyframe_homographies[self._sorted_keys[0]]
        if timestamp_s >= self._sorted_keys[-1]:
            return self.keyframe_homographies[self._sorted_keys[-1]]

        k_prev = self._sorted_keys[0]
        k_next = self._sorted_keys[-1]
        for i in range(len(self._sorted_keys) - 1):
            if self._sorted_keys[i] <= timestamp_s <= self._sorted_keys[i + 1]:
                k_prev = self._sorted_keys[i]
                k_next = self._sorted_keys[i + 1]
                break

        if k_prev == k_next:
            return self.keyframe_homographies[k_prev]

        alpha = (timestamp_s - k_prev) / (k_next - k_prev)
        s = (1.0 - math.cos(alpha * math.pi)) / 2.0

        H_prev = self.keyframe_homographies[k_prev]
        H_next = self.keyframe_homographies[k_next]

        # Non-singular projective interpolation via full-pitch anchor points
        try:
            import cv2

            H_inv_prev = np.linalg.inv(H_prev)
            H_inv_next = np.linalg.inv(H_next)
            anchors_pitch = np.array(FULL_PITCH_ANCHORS_M, dtype=np.float32)
            scr_prev = []
            scr_next = []
            for p in anchors_pitch:
                vp = np.dot(H_inv_prev, np.array([p[0], p[1], 1.0]))
                vn = np.dot(H_inv_next, np.array([p[0], p[1], 1.0]))
                scr_prev.append([vp[0] / vp[2], vp[1] / vp[2]])
                scr_next.append([vn[0] / vn[2], vn[1] / vn[2]])
            scr_interp = (1.0 - s) * np.array(
                scr_prev, dtype=np.float32
            ) + s * np.array(scr_next, dtype=np.float32)
            H_interp, _ = cv2.findHomography(scr_interp, anchors_pitch, 0)
            if H_interp is not None and not np.isnan(H_interp).any():
                if abs(H_interp[2, 2]) > 1e-8:
                    H_interp = H_interp / H_interp[2, 2]
                return H_interp.astype(np.float64)
        except Exception:
            pass

        H_interp = (1.0 - s) * H_prev + s * H_next
        if abs(H_interp[2, 2]) > 1e-8:
            H_interp = H_interp / H_interp[2, 2]
        return H_interp.astype(np.float64)

    def get_H(
        self, frame_idx: Optional[int] = None, timestamp_s: Optional[float] = None
    ) -> np.ndarray:
        if timestamp_s is not None:
            return self.get_H_by_time(timestamp_s)
        if frame_idx is None:
            return self.H
        return self.get_H_by_time(frame_idx / self.fps if self.fps else 0.0)

    def to_pitch(
        self,
        u: float,
        v: float,
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> PitchPoint:
        H = self.get_H(frame_idx, timestamp_s=timestamp_s)
        return project_screen_to_pitch(u, v, H)

    def to_pitch_valid(
        self,
        u: float,
        v: float,
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> Tuple[PitchPoint, bool]:
        H = self.get_H(frame_idx, timestamp_s=timestamp_s)
        return project_screen_to_pitch_valid(u, v, H)

    def to_screen(
        self,
        x_m: float,
        y_m: float,
        z_m: float = 0.0,
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> Tuple[int, int]:
        H = self.get_H(frame_idx, timestamp_s=timestamp_s)
        return project_pitch_to_screen(x_m, y_m, H, z_m=z_m)
