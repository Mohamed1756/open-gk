"""
Unit tests for PlayerPoseEstimator and body facing angle extraction.
"""

import math
import numpy as np
from src.cv.pose_estimator import compute_facing_angle_from_keypoints


def test_facing_angle_from_horizontal_shoulders():
    """Verifies that horizontal shoulders facing down-screen produce expected angle."""
    keypoints = np.zeros((17, 3), dtype=np.float32)

    # Right shoulder at (40, 50), Left shoulder at (60, 50)
    keypoints[5] = [60.0, 50.0, 0.90]  # L-shoulder
    keypoints[6] = [40.0, 50.0, 0.90]  # R-shoulder
    # Hips below shoulders
    keypoints[11] = [58.0, 70.0, 0.90]  # L-hip
    keypoints[12] = [42.0, 70.0, 0.90]  # R-hip

    angle_rad, conf, is_valid = compute_facing_angle_from_keypoints(
        keypoints, homography=None
    )
    assert is_valid is True
    assert conf > 0.85
    # Normal to horizontal shoulder vector (dx=20, dy=0) points along +Y in screen space (angle = pi/2)
    assert abs(angle_rad - math.pi / 2.0) < 0.1


def test_facing_angle_low_confidence_fallback():
    """Verifies that low confidence keypoints are marked invalid."""
    keypoints = np.zeros((17, 3), dtype=np.float32)
    # All confidences 0.1
    keypoints[:, 2] = 0.10

    angle_rad, conf, is_valid = compute_facing_angle_from_keypoints(
        keypoints, homography=None
    )
    assert is_valid is False
    assert conf == 0.0


def test_front_back_disambiguation_with_head_keypoints():
    """Verifies that facial keypoints resolve front vs back orientation."""
    from src.cv.pose_estimator import compute_detailed_pose_orientation

    kp_front = np.zeros((17, 3), dtype=np.float32)
    # Horizontal shoulders: L-sh=(60, 50), R-sh=(40, 50)
    kp_front[5] = [60.0, 50.0, 0.90]
    kp_front[6] = [40.0, 50.0, 0.90]
    kp_front[11] = [58.0, 70.0, 0.90]
    kp_front[12] = [42.0, 70.0, 0.90]
    # Nose pointing down-screen (towards camera): nose=(50, 48), ears=(45, 42), (55, 42)
    kp_front[0] = [50.0, 48.0, 0.95]  # nose in front of ears
    kp_front[3] = [55.0, 42.0, 0.90]  # L-ear
    kp_front[4] = [45.0, 42.0, 0.90]  # R-ear

    res_front = compute_detailed_pose_orientation(kp_front, homography=None)
    assert res_front.is_valid is True
    # Facing down-screen in screen coordinates (+Y, angle = pi/2)
    assert abs(res_front.torso_facing_rad - math.pi / 2.0) < 0.15
    assert abs(res_front.gaze_facing_rad - math.pi / 2.0) < 0.15

    # Player facing AWAY from camera: nose is behind ears (higher Y in screen space, or nose occluded)
    kp_away = np.zeros((17, 3), dtype=np.float32)
    kp_away[5] = [60.0, 50.0, 0.90]
    kp_away[6] = [40.0, 50.0, 0.90]
    kp_away[11] = [58.0, 70.0, 0.90]
    kp_away[12] = [42.0, 70.0, 0.90]
    # Nose pointing up-screen (away from camera): nose=(50, 36), ears=(45, 42), (55, 42)
    kp_away[0] = [50.0, 36.0, 0.95]  # nose behind ears in screen space
    kp_away[3] = [55.0, 42.0, 0.90]
    kp_away[4] = [45.0, 42.0, 0.90]

    res_away = compute_detailed_pose_orientation(kp_away, homography=None)
    assert res_away.is_valid is True
    # Facing up-screen in screen coordinates (-Y, angle = -pi/2)
    assert abs(res_away.torso_facing_rad - (-math.pi / 2.0)) < 0.15
    assert abs(res_away.gaze_facing_rad - (-math.pi / 2.0)) < 0.15


def test_velocity_disambiguation_when_head_occluded():
    """Verifies that motion velocity disambiguates facing when head keypoints are occluded."""
    from src.cv.pose_estimator import compute_detailed_pose_orientation

    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = [60.0, 50.0, 0.90]
    kp[6] = [40.0, 50.0, 0.90]
    kp[11] = [58.0, 70.0, 0.90]
    kp[12] = [42.0, 70.0, 0.90]
    # Head keypoints all 0.0 confidence
    kp[0:5, 2] = 0.0

    # Running up-screen (vy = -10.0 px/frame)
    res_up = compute_detailed_pose_orientation(
        kp, homography=None, velocity_screen_uv=(0.0, -10.0)
    )
    assert res_up.is_valid is True
    # Inverted from default down-screen to up-screen (-pi/2)
    assert abs(res_up.torso_facing_rad - (-math.pi / 2.0)) < 0.15


def test_decoupled_gaze_direction():
    """Verifies that gaze angle can look diagonally while torso remains square."""
    from src.cv.pose_estimator import compute_detailed_pose_orientation

    kp = np.zeros((17, 3), dtype=np.float32)
    # Torso square down-screen
    kp[5] = [60.0, 50.0, 0.90]
    kp[6] = [40.0, 50.0, 0.90]
    kp[11] = [58.0, 70.0, 0.90]
    kp[12] = [42.0, 70.0, 0.90]

    # Head turned to the right (looking down-right: dx=10, dy=10)
    # Ear mid at (45, 42), nose at (55, 52) -> dir = (10, 10) -> angle = pi/4
    kp[0] = [55.0, 52.0, 0.95]
    kp[3] = [50.0, 42.0, 0.90]
    kp[4] = [40.0, 42.0, 0.90]

    res = compute_detailed_pose_orientation(kp, homography=None)
    assert res.is_valid is True
    # Torso points down (pi/2)
    assert abs(res.torso_facing_rad - math.pi / 2.0) < 0.15
    # Gaze points down-right (pi/4)
    assert abs(res.gaze_facing_rad - math.pi / 4.0) < 0.20


def test_circular_weighted_smooth():
    """Verifies that circular mean handles -pi/pi wrapping and confidence weighting."""
    from src.cv.pose_estimator import circular_weighted_smooth

    # Angles near +/- pi: +3.10 and -3.10 (both pointing almost due left)
    # Linear mean would be 0.0 (pointing right!). Circular mean should be ~pi.
    angles = [3.10, -3.10]
    weights = [1.0, 1.0]
    smoothed = circular_weighted_smooth(angles, weights)
    assert abs(abs(smoothed) - math.pi) < 0.10

    # Low-confidence outlier suppression
    angles_outlier = [0.0, 0.1, math.pi]  # math.pi is outlier with low confidence
    weights_outlier = [0.9, 0.9, 0.05]
    smoothed_outlier = circular_weighted_smooth(angles_outlier, weights_outlier)
    assert abs(smoothed_outlier - 0.05) < 0.15


def test_cervical_rotation_clamp():
    """Verifies that unnatural head rotation (>75 deg relative to torso) is anatomically clamped."""
    from src.cv.pose_estimator import (
        compute_detailed_pose_orientation,
        MAX_CERVICAL_ROTATION_RAD,
    )

    kp = np.zeros((17, 3), dtype=np.float32)
    # Torso square down-screen (+Y in screen space, angle = pi/2)
    kp[5] = [60.0, 50.0, 0.90]
    kp[6] = [40.0, 50.0, 0.90]
    kp[11] = [58.0, 70.0, 0.90]
    kp[12] = [42.0, 70.0, 0.90]

    # Extreme head vector pointing straight left (angle = pi), which is 90 deg away from torso (pi/2)
    # Ears at (50, 40), nose at (30, 40) -> vector = (-20, 0)
    kp[0] = [30.0, 40.0, 0.95]
    kp[3] = [50.0, 45.0, 0.90]
    kp[4] = [50.0, 35.0, 0.90]

    res = compute_detailed_pose_orientation(kp, homography=None)
    assert res.is_valid is True
    # Torso remains pi/2
    assert abs(res.torso_facing_rad - math.pi / 2.0) < 0.15
    # Gaze must be clamped to torso + 75 deg, NOT raw 90 deg
    expected_clamped = math.pi / 2.0 + MAX_CERVICAL_ROTATION_RAD
    assert abs(res.gaze_facing_rad - expected_clamped) < 0.05


def test_gaze_valid_requires_facial_evidence():
    """Back-of-head frames must not report torso direction as gaze."""
    from src.cv.pose_estimator import compute_detailed_pose_orientation

    kp = np.zeros((17, 3), dtype=np.float32)
    kp[5] = [60.0, 50.0, 0.90]
    kp[6] = [40.0, 50.0, 0.90]
    kp[11] = [58.0, 70.0, 0.90]
    kp[12] = [42.0, 70.0, 0.90]
    kp[0:5, 2] = 0.0

    res = compute_detailed_pose_orientation(kp, homography=None)
    assert res.is_valid is True
    assert res.gaze_valid is False

    kp[0] = [50.0, 48.0, 0.95]
    kp[3] = [55.0, 42.0, 0.90]
    kp[4] = [45.0, 42.0, 0.90]
    res_front = compute_detailed_pose_orientation(kp, homography=None)
    assert res_front.gaze_valid is True


def test_smooth_gaze_ignores_unknown_frames():
    """Smoothed gaze comes from facial-evidence frames only."""
    from src.cv.pose_estimator import (
        PlayerPoseEstimator,
        compute_detailed_pose_orientation,
    )

    est = PlayerPoseEstimator.__new__(PlayerPoseEstimator)
    kp_base = np.zeros((17, 3), dtype=np.float32)
    kp_base[5] = [60.0, 50.0, 0.90]
    kp_base[6] = [40.0, 50.0, 0.90]
    kp_base[11] = [58.0, 70.0, 0.90]
    kp_base[12] = [42.0, 70.0, 0.90]
    unknown = compute_detailed_pose_orientation(kp_base, homography=None)
    assert unknown.is_valid is True
    assert unknown.gaze_valid is False

    kp = kp_base.copy()
    kp[0] = [50.0, 48.0, 0.95]
    kp[3] = [55.0, 42.0, 0.90]
    kp[4] = [45.0, 42.0, 0.90]
    known = compute_detailed_pose_orientation(kp, homography=None)
    assert known.gaze_valid is True

    smoothed = est.smooth_pose_sequence([unknown, unknown, known])
    assert smoothed.gaze_valid is True
    assert abs(smoothed.gaze_facing_rad - math.pi / 2.0) < 0.15

    all_unknown = est.smooth_pose_sequence([unknown, unknown])
    assert all_unknown.gaze_valid is False
