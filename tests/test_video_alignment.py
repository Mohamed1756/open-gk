"""
Unit tests for Broadcast Video Time-Alignment & Kickoff Synchronization Engine.
"""

import numpy as np
from src.cv.video_alignment import (
    BroadcastTimeAligner,
    compute_pitch_green_ratio,
    is_tactical_pitch_view,
    parse_clock_string,
    find_kickoff_timestamp_from_green_profile,
)


def test_pitch_green_ratio_calculation():
    """Verifies that pitch green turf pixels are accurately detected vs ads/closeups."""
    # Synthetic tactical pitch frame (dominant green)
    tactical_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tactical_frame[:, :, 0] = 40  # Red
    tactical_frame[:, :, 1] = 130  # Green (dominant)
    tactical_frame[:, :, 2] = 45  # Blue

    ratio = compute_pitch_green_ratio(tactical_frame)
    assert ratio > 0.90
    assert is_tactical_pitch_view(tactical_frame)

    # Synthetic non-pitch frame (studio/crowd/ad)
    studio_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    studio_frame[:, :, 0] = 180  # Red
    studio_frame[:, :, 1] = 80  # Green
    studio_frame[:, :, 2] = 200  # Blue

    non_pitch_ratio = compute_pitch_green_ratio(studio_frame)
    assert non_pitch_ratio < 0.10
    assert not is_tactical_pitch_view(studio_frame)


def test_clock_string_parsing():
    """Verifies parsing of scoreboard strings into elapsed seconds."""
    assert parse_clock_string("14:32") == 14 * 60 + 32
    assert parse_clock_string("00:00") == 0.0
    assert parse_clock_string("45:00'") == 45 * 60
    assert parse_clock_string("invalid") is None


def test_broadcast_time_aligner():
    """Verifies conversion from raw video timestamps to official match time."""
    # Kickoff occurred at 3 minutes (180.0s) into the raw video file
    aligner = BroadcastTimeAligner(kickoff_video_timestamp_s=180.0, fps=50.0)

    # Video timestamp at 240.0s should be match minute 01:00 (60s into match)
    match_s, clock_str = aligner.video_time_to_match_time(240.0)
    assert match_s == 60.0
    assert clock_str == "01:00"

    # Frame 9000 (180s * 50fps) should be 00:00 kickoff
    k_match_s, k_clock_str = aligner.frame_index_to_match_time(9000)
    assert k_match_s == 0.0
    assert k_clock_str == "00:00"


def test_kickoff_timestamp_detection():
    """Verifies scanning kickoff transition from pre-match intro to sustained pitch view."""
    timestamps = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    # Pre-match intro (low green) -> Sustained tactical view starting at t=40s
    green_ratios = [0.05, 0.10, 0.12, 0.20, 0.65, 0.70, 0.75, 0.68, 0.72]

    detected_t0 = find_kickoff_timestamp_from_green_profile(
        timestamps, green_ratios, sustained_window_frames=4, min_green_threshold=0.50
    )
    assert detected_t0 == 40.0
