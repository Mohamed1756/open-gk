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
