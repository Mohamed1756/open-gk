"""
Computer Vision module for Planar Pitch Homography, YOLO Player Detection, Tracking, and Pose Estimation.
"""

from src.cv.match_segmenter import (
    CandidateSequence,
    cluster_possession_triggers,
    detect_scene_cut,
    is_tactical_pitch_frame,
    scan_match_video_coarse,
    slice_match_sequences,
)
from src.cv.pitch_homography import (
    PitchKeypoint,
    PlanarPitchHomography,
    compute_homography_matrix,
    project_screen_to_pitch,
    project_pitch_to_screen,
)
from src.cv.player_detector import (
    DetectedPlayer,
    YOLOPlayerDetector,
)

__all__ = [
    "CandidateSequence",
    "cluster_possession_triggers",
    "detect_scene_cut",
    "is_tactical_pitch_frame",
    "scan_match_video_coarse",
    "slice_match_sequences",
    "PitchKeypoint",
    "PlanarPitchHomography",
    "compute_homography_matrix",
    "project_screen_to_pitch",
    "project_pitch_to_screen",
    "DetectedPlayer",
    "YOLOPlayerDetector",
]
