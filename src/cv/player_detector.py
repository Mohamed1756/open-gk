"""
Automated Deep Learning Object Detector for Football Broadcast Frames.
Includes Pitch-Boundary Field of Play Filtering (filtering out stewards/bench staff)
and Edge-Robust Color Sampling (classifying border-clipped referees).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import numpy as np

from src.cv.pitch_homography import PlanarPitchHomography


@dataclass(frozen=True)
class DetectedPlayer:
    """Represents an actual player, goalkeeper, official, or ball detected from video frame pixels with canonical pitch mapping."""

    track_id: int
    class_name: str  # "person" or "sports ball"
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    feet_screen_uv: Tuple[float, float]
    pitch_pos_m: Tuple[float, float]
    team_label: str  # "Torino (Maroon)", "AC Milan (White)", "Goalkeeper (Yellow)", "Match Official (Cyan)", "Match Ball", or "Off-Pitch Staff"
    team_color_hex: str
    pitch_valid: bool = True


def apply_adaptive_contrast_enhancement(img_bgr: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE on the L channel in LAB space to enhance low-contrast silhouettes
    against dark or flashing perimeter LED advertising boards.
    """
    import cv2

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_enhanced, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def is_keeper_label(label: str) -> bool:
    low = label.lower()
    return "gk" in low or "keeper" in low or "maignan" in low


def is_official_label(label: str) -> bool:
    low = label.lower()
    return "official" in low or "referee" in low or "cyan" in low


def classify_jersey_precision(
    frame_rgb: np.ndarray,
    frame_hsv: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    kits: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, str]:
    """
    Extracts torso fabric color with edge-robust sampling for border-clipped figures.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w, _ = frame_rgb.shape

    # Clamp bbox within frame
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    box_h = y2 - y1
    box_w = x2 - x1
    if box_h < 10 or box_w < 5:
        return "Unknown", "#94a3b8"

    # Edge-robust sampling: if box touches screen border, sample entire box; else center 70% torso
    is_border_clipped = x1 <= 5 or x2 >= w - 5 or y1 <= 5 or y2 >= h - 5
    if is_border_clipped:
        torso_rgb = frame_rgb[y1:y2, x1:x2]
        torso_hsv = frame_hsv[y1:y2, x1:x2]
    else:
        tx1 = x1 + int(box_w * 0.15)
        tx2 = x2 - int(box_w * 0.15)
        ty1 = y1 + int(box_h * 0.15)
        ty2 = y1 + int(box_h * 0.55)
        torso_rgb = frame_rgb[ty1:ty2, tx1:tx2]
        torso_hsv = frame_hsv[ty1:ty2, tx1:tx2]

    if torso_rgb.size == 0:
        return "Unknown", "#94a3b8"

    # Mask out green pitch grass (Hue 35 to 85 in OpenCV HSV, Saturation > 35)
    is_grass = (
        (torso_hsv[:, :, 0] >= 35)
        & (torso_hsv[:, :, 0] <= 85)
        & (torso_hsv[:, :, 1] >= 35)
    )
    jersey_rgb = torso_rgb[~is_grass]
    jersey_hsv = torso_hsv[~is_grass]
    jersey_pixels = int(len(jersey_rgb))
    used_grass_fallback = jersey_pixels < 5

    if used_grass_fallback:
        jersey_rgb = torso_rgb.reshape(-1, 3)
        jersey_hsv = torso_hsv.reshape(-1, 3)

    r = float(np.median(jersey_rgb[:, 0]))
    g = float(np.median(jersey_rgb[:, 1]))
    b_val = float(np.median(jersey_rgb[:, 2]))

    h_val = float(np.median(jersey_hsv[:, 0]))
    s_val = float(np.median(jersey_hsv[:, 1]))

    # 1. AC Milan Goalkeeper (Mike Maignan - Vibrant Purple/Fuchsia Kit):
    # Distinct Purple/Magenta: High Blue & Red, Blue > Green + 25, Blue > 90, Saturation >= 90
    if (b_val > g + 25 and r > g + 25 and b_val > 90 and s_val >= 90) or (
        130 <= h_val <= 170 and b_val > g + 30 and s_val >= 100
    ):
        return "AC Milan GK (Maignan - Purple)", "#c026d3"

    # 2. Goalkeeper Kit (Bright Yellow / Neon Orange):
    if (18 <= h_val <= 40) and s_val > 65 and r > 130 and g > 130:
        return "Goalkeeper (Yellow)", "#facc15"

    # 2b. Ambiguous fabric signal: grass-mask fallback or a handful of jersey
    # pixels inside the grass/cyan overlap band (H 65-115) is LED spill, shadow,
    # or occlusion — not a kit. Emit Unknown instead of guessing cyan/maroon.
    if (used_grass_fallback or jersey_pixels < 15) and 65 <= h_val <= 115:
        return "Unknown", "#94a3b8"
    if is_border_clipped and jersey_pixels < 10:
        return "Unknown", "#94a3b8"

    # 3. Match Official (Referee / Linesman in Cyan kit):
    # Cyan hue (75-112 in OpenCV), Green and Blue higher than Red
    if (75 <= h_val <= 112) and (b_val > r + 15 or g > r + 15):
        return "Match Official (Cyan)", "#06b6d4"

    # 4. AC Milan (White Away Kit - High brightness, low-to-moderate saturation):
    if (r + g + b_val >= 360 and s_val <= 75) or (
        r >= 115 and g >= 115 and b_val >= 115 and s_val <= 70
    ):
        return "AC Milan (White)", "#ffffff"

    # 5. Torino (Maroon Home Kit):
    return "Torino (Maroon)", "#881337"


class YOLOPlayerDetector:
    """
    Automated Deep Learning Object Detector for football broadcast frames.
    Detects players, goalkeepers, officials, and ball with Field-of-Play perimeter filtering.
    """

    def __init__(
        self,
        weights_path: str = "yolov8m.pt",
        homography: Optional[PlanarPitchHomography] = None,
        conf_threshold: float = 0.20,
        ball_conf_threshold: float = 0.08,
        iou_threshold: float = 0.35,
        kits: Optional[List[Any]] = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.homography = homography or PlanarPitchHomography()
        self.conf_threshold = conf_threshold
        self.ball_conf_threshold = ball_conf_threshold
        self.iou_threshold = iou_threshold
        self.kits = list(kits) if kits else []

    def _remap_to_kits(self, team_label: str, team_hex: str) -> Tuple[str, str]:
        if is_keeper_label(team_label):
            for k in self.kits:
                if getattr(k, "is_keeper", False) and k.label == team_label:
                    return k.label, k.color_hex
            for k in self.kits:
                if (
                    getattr(k, "is_keeper", False)
                    and k.label.split("(")[0].strip().lower() in team_label.lower()
                ):
                    return k.label, k.color_hex
            return team_label, team_hex
        if is_official_label(team_label):
            for k in self.kits:
                if getattr(k, "is_official", False):
                    return k.label, k.color_hex
        for k in self.kits:
            if getattr(k, "is_keeper", False) or getattr(k, "is_official", False):
                continue
            if k.label == team_label:
                return k.label, k.color_hex
        if team_label == "Unknown":
            for k in self.kits:
                if getattr(k, "team_id", "") == "unknown":
                    return k.label, k.color_hex
        return team_label, team_hex

    def detect_frame(
        self, frame_path_or_array: Any, frame_idx: Optional[int] = None
    ) -> List[DetectedPlayer]:
        """
        Runs object detection on a single image file or numpy array.
        Filters off-pitch personnel (stewards, bench staff) and resolves border-clipped officials.
        """
        import cv2

        if isinstance(frame_path_or_array, str):
            img_bgr = cv2.imread(frame_path_or_array)
        elif isinstance(frame_path_or_array, np.ndarray):
            img_bgr = frame_path_or_array
        else:
            img_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

        if img_bgr is None:
            return []

        # 1. Dynamic CLAHE Contrast Enhancement
        enhanced_bgr = apply_adaptive_contrast_enhancement(img_bgr)
        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 2. Run Deep Learning Detection with dual threshold
        results = self.model(
            enhanced_bgr,
            conf=self.ball_conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        raw_boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]

                if cls_name == "person" and conf >= self.conf_threshold:
                    xyxy = tuple(float(x) for x in box.xyxy[0])
                    raw_boxes.append((conf, xyxy, "person"))
                elif cls_name == "sports ball" and conf >= self.ball_conf_threshold:
                    xyxy = tuple(float(x) for x in box.xyxy[0])
                    raw_boxes.append((conf, xyxy, "sports ball"))

        if not raw_boxes:
            return []

        # Sort detections left to right by screen x
        raw_boxes = sorted(raw_boxes, key=lambda b: b[1][0])

        detections: List[DetectedPlayer] = []
        to_valid = getattr(self.homography, "to_pitch_valid", None)
        for idx, (conf, xyxy, cls_name) in enumerate(raw_boxes, 1):
            if cls_name == "sports ball":
                center_u = (xyxy[0] + xyxy[2]) / 2.0
                center_v = (xyxy[1] + xyxy[3]) / 2.0
                if to_valid is not None:
                    pitch_pt, ball_valid = to_valid(
                        center_u, center_v, frame_idx=frame_idx
                    )
                else:
                    pitch_pt = self.homography.to_pitch(
                        center_u, center_v, frame_idx=frame_idx
                    )
                    ball_valid = True
                pitch_coords = (round(pitch_pt.x, 2), round(pitch_pt.y, 2))

                detections.append(
                    DetectedPlayer(
                        track_id=idx,
                        class_name="sports ball",
                        confidence=round(conf, 3),
                        bbox_xyxy=xyxy,
                        feet_screen_uv=(round(center_u, 1), round(center_v, 1)),
                        pitch_pos_m=pitch_coords,
                        team_label="Match Ball",
                        team_color_hex="#eab308",
                        pitch_valid=bool(ball_valid),
                    )
                )
            else:
                feet_u = (xyxy[0] + xyxy[2]) / 2.0
                feet_v = xyxy[3]

                if to_valid is not None:
                    pitch_pt, pitch_valid = to_valid(
                        feet_u, feet_v, frame_idx=frame_idx
                    )
                else:
                    pitch_pt = self.homography.to_pitch(
                        feet_u, feet_v, frame_idx=frame_idx
                    )
                    pitch_valid = True
                pitch_coords = (round(pitch_pt.x, 2), round(pitch_pt.y, 2))

                # Field of Play Filter: check if figure is off-pitch (behind advertising boards)
                # Sample 10px below feet to check for turf grass
                ground_y2 = min(719, int(feet_v) + 8)
                ground_patch_hsv = frame_hsv[
                    int(feet_v) : ground_y2,
                    max(0, int(feet_u) - 10) : min(1279, int(feet_u) + 10),
                ]
                is_on_turf = False
                if ground_patch_hsv.size > 0:
                    grass_ratio = np.mean(
                        (ground_patch_hsv[:, :, 0] >= 35)
                        & (ground_patch_hsv[:, :, 0] <= 85)
                        & (ground_patch_hsv[:, :, 1] >= 35)
                    )
                    is_on_turf = grass_ratio > 0.20

                if not is_on_turf and (feet_v < 220 or pitch_coords[0] <= 0.0):
                    team_label = "Off-Pitch Staff (Steward)"
                    team_hex = "#64748b"
                else:
                    team_label, team_hex = classify_jersey_precision(
                        frame_rgb, frame_hsv, xyxy
                    )
                    if self.kits:
                        team_label, team_hex = self._remap_to_kits(team_label, team_hex)

                detections.append(
                    DetectedPlayer(
                        track_id=idx,
                        class_name="person",
                        confidence=round(conf, 3),
                        bbox_xyxy=xyxy,
                        feet_screen_uv=(round(feet_u, 1), round(feet_v, 1)),
                        pitch_pos_m=pitch_coords,
                        team_label=team_label,
                        team_color_hex=team_hex,
                        pitch_valid=bool(pitch_valid),
                    )
                )

        return detections
