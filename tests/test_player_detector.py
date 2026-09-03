"""
Unit tests for YOLO Player & Ball Detector with Pitch Homography Mapping and Jersey Classification.
"""

import numpy as np

from pathlib import Path
from src.cv.player_detector import (
    YOLOPlayerDetector,
    DetectedPlayer,
    classify_jersey_precision,
)
from src.cv.pitch_homography import PlanarPitchHomography
from src.cv.player_tracker import resolve_display_label


def test_yolo_player_detector_initialization():
    """Verifies that YOLO detector initializes with homography and valid configuration."""
    detector = YOLOPlayerDetector(conf_threshold=0.30)
    assert detector.conf_threshold == 0.30
    assert isinstance(detector.homography, PlanarPitchHomography)


def _solid_frames(rgb, hsv, size=60):
    return (
        np.full((size, size, 3), rgb, dtype=np.uint8),
        np.full((size, size, 3), hsv, dtype=np.uint8),
    )


def test_display_birth_needs_two_agreeing_votes():
    assert resolve_display_label(None, ["AC Milan (White)"]) is None
    assert (
        resolve_display_label(None, ["AC Milan (White)", "AC Milan (White)"])
        == "AC Milan (White)"
    )
    assert resolve_display_label(None, ["AC Milan (White)", "Torino (Maroon)"]) is None


def test_display_tolerates_short_blips():
    base = ["AC Milan (White)"] * 5
    assert (
        resolve_display_label("AC Milan (White)", base + ["Torino (Maroon)"] * 4)
        == "AC Milan (White)"
    )
    assert (
        resolve_display_label("AC Milan (White)", base + ["Torino (Maroon)"] * 5)
        == "Torino (Maroon)"
    )


def test_display_eventually_accepts_corrections():
    assert (
        resolve_display_label("Match Official (Cyan)", ["AC Milan (White)"] * 5)
        == "AC Milan (White)"
    )


def test_ambiguous_grass_overlap_yields_unknown():
    rgb, hsv = _solid_frames([120, 140, 150], [80, 200, 150])
    label, _ = classify_jersey_precision(rgb, hsv, (10, 10, 40, 40))
    assert label == "Unknown"


def test_clear_white_still_classifies_milan():
    rgb, hsv = _solid_frames([230, 230, 230], [0, 10, 230])
    label, _ = classify_jersey_precision(rgb, hsv, (10, 10, 40, 40))
    assert label == "AC Milan (White)"


def test_yolo_detection_on_real_frame():
    """Verifies that real YOLO inference detects players on a match frame without hardcoding."""
    frame_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "video_raw"
        / "debug_frames"
        / "frame_0540s.png"
    )
    if not frame_path.exists():
        return

    detector = YOLOPlayerDetector(conf_threshold=0.20)
    detections = detector.detect_frame(str(frame_path))

    # Real match frame must contain multiple detected players and officials
    assert len(detections) >= 8

    labels = [d.team_label for d in detections]
    assert "AC Milan (White)" in labels
    assert "Torino (Maroon)" in labels
    assert "Match Official (Cyan)" in labels

    for d in detections:
        assert isinstance(d, DetectedPlayer)
        assert d.class_name in ["person", "sports ball"]
        assert d.confidence >= 0.20
        # Coordinates must be projected onto pitch coordinates
        assert len(d.pitch_pos_m) == 2
        # Screen feet coordinates must be within 1080p / 720p bounds
        assert 0 <= d.feet_screen_uv[0] <= 1920
        assert 0 <= d.feet_screen_uv[1] <= 1080
