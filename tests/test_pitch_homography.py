"""
Unit tests for Planar Pitch Homography and AR Tactical Overlay generation.
"""

from pathlib import Path
import numpy as np
from src.core.geometry import PitchPoint
from src.cv.pitch_homography import (
    PlanarPitchHomography,
    DynamicPitchHomography,
    CalibrationKeyframe,
    compute_homography_matrix,
    homographies_from_keyframes,
    is_inside_pitch_hull,
    project_screen_to_pitch,
    project_screen_to_pitch_valid,
    save_calibration_json,
    load_calibration_json,
)
from src.cv.video_overlay import generate_interactive_ar_overlay_html


def test_homography_dlt_computation():
    """Verifies that Direct Linear Transformation computes the correct 3x3 projective matrix."""
    screen_pts = [(100.0, 800.0), (300.0, 200.0), (1600.0, 250.0), (1500.0, 850.0)]
    pitch_pts = [(0.0, 0.0), (0.0, 68.0), (52.5, 68.0), (52.5, 0.0)]

    H = compute_homography_matrix(screen_pts, pitch_pts)
    assert H.shape == (3, 3)
    assert not np.isnan(H).any()

    for (u, v), (exp_x, exp_y) in zip(screen_pts, pitch_pts):
        pt = project_screen_to_pitch(u, v, H)
        assert abs(pt.x - exp_x) < 0.25
        assert abs(pt.y - exp_y) < 0.25


def test_inverse_pitch_to_screen_projection():
    """Verifies that inverse projection H^-1 correctly recovers screen pixel coordinates."""
    homography = PlanarPitchHomography()

    pitch_x, pitch_y = 16.5, 34.0
    u, v = homography.to_screen(pitch_x, pitch_y, z_m=0.0)

    assert 0 <= u <= 1920
    assert 0 <= v <= 1080

    pt_reconstructed = homography.to_pitch(float(u), float(v))
    assert abs(pt_reconstructed.x - pitch_x) < 0.5
    assert abs(pt_reconstructed.y - pitch_y) < 0.5


def test_augmented_reality_trajectory_overlay(tmp_path: Path):
    """Verifies AR flight trajectory vector generation and HTML report export."""
    homography = PlanarPitchHomography()

    gk = PitchPoint(x=12.0, y=34.0)
    target = PitchPoint(x=38.0, y=14.0)
    striker = PitchPoint(x=18.0, y=32.0)

    overlay = homography.generate_ar_trajectory_overlay(
        gk_pt=gk,
        target_pt=target,
        striker_pt=striker,
        swerve_lateral_m=2.10,
        flight_time_s=1.8,
        num_points=20,
    )

    assert len(overlay["flight_trajectory_pixels"]) == 20
    assert len(overlay["striker_danger_pixels"]) == 12
    assert overlay["magnus_swerve_m"] == 2.10

    test_html = tmp_path / "test_overlay.html"
    generate_interactive_ar_overlay_html(
        match_title="Test Match",
        timestamp_str="00:15",
        frame_image_rel_path="test.png",
        overlay_data=overlay,
        gk_name="Test Keeper",
        target_player_name="Test Target",
        presser_speed_ms=7.5,
        escape_success_pct=85.0,
        clearance_margin_s=0.85,
        tactical_insight="Test insight",
        output_path=test_html,
    )

    assert test_html.exists()
    assert test_html.stat().st_size > 500


def test_dynamic_pitch_homography_interpolation():
    """Verifies that DynamicPitchHomography smoothly interpolates coordinates between keyframes."""
    # Keyframe 0 (midfield camera)
    src_0 = [(100.0, 500.0), (100.0, 200.0), (800.0, 200.0), (800.0, 500.0)]
    dst_0 = [(20.0, 10.0), (20.0, 58.0), (60.0, 58.0), (60.0, 10.0)]
    H_0 = compute_homography_matrix(src_0, dst_0)

    # Keyframe 100 (shifted camera by +200px)
    src_100 = [(300.0, 500.0), (300.0, 200.0), (1000.0, 200.0), (1000.0, 500.0)]
    dst_100 = [(20.0, 10.0), (20.0, 58.0), (60.0, 58.0), (60.0, 10.0)]
    H_100 = compute_homography_matrix(src_100, dst_100)

    dyn = DynamicPitchHomography(keyframe_homographies={0.0: H_0, 4.0: H_100}, fps=25.0)

    # At t=0: matches H_0
    pt_f0 = dyn.to_pitch(100.0, 500.0, frame_idx=0)
    assert abs(pt_f0.x - 20.0) < 0.5
    assert abs(pt_f0.y - 10.0) < 0.5

    # At t=4s (frame 100): matches H_100
    pt_f100 = dyn.to_pitch(300.0, 500.0, frame_idx=100)
    assert abs(pt_f100.x - 20.0) < 0.5
    assert abs(pt_f100.y - 10.0) < 0.5

    # At t=2s (frame 50): intermediate smooth interpolation
    pt_mid = dyn.to_pitch(200.0, 500.0, frame_idx=50)
    assert abs(pt_mid.x - 20.0) < 1.5
    assert abs(pt_mid.y - 10.0) < 1.5


def test_outside_hull_flagged_not_clamped_silently():
    H = compute_homography_matrix(
        [(100.0, 500.0), (100.0, 200.0), (800.0, 200.0), (800.0, 500.0)],
        [(20.0, 10.0), (20.0, 58.0), (60.0, 58.0), (60.0, 10.0)],
    )
    _, valid_inside = project_screen_to_pitch_valid(400.0, 350.0, H)
    assert valid_inside
    _, valid_far = project_screen_to_pitch_valid(-5000.0, -5000.0, H)
    assert not valid_far
    assert not is_inside_pitch_hull(-50.0, 34.0)
    assert is_inside_pitch_hull(52.5, 34.0)


def test_calibration_roundtrip_timestamp_keyed(tmp_path):
    k0 = CalibrationKeyframe(
        timestamp_s=0.0,
        screen_points=[(100.0, 500.0), (100.0, 200.0), (800.0, 200.0), (800.0, 500.0)],
        pitch_points=[(20.0, 10.0), (20.0, 58.0), (60.0, 58.0), (60.0, 10.0)],
    )
    k1 = CalibrationKeyframe(
        timestamp_s=4.0,
        screen_points=[
            (300.0, 500.0),
            (300.0, 200.0),
            (1000.0, 200.0),
            (1000.0, 500.0),
        ],
        pitch_points=[(20.0, 10.0), (20.0, 58.0), (60.0, 58.0), (60.0, 10.0)],
    )
    path = tmp_path / "calib.json"
    save_calibration_json(str(path), [k0, k1], match_id="t", fps=25.0)
    loaded, payload = load_calibration_json(str(path))
    assert payload["fps"] == 25.0
    assert [k.timestamp_s for k in loaded] == [0.0, 4.0]
    homos = homographies_from_keyframes(loaded)
    assert set(homos.keys()) == {0.0, 4.0}
    dyn = DynamicPitchHomography(keyframe_homographies=homos, fps=25.0)
    assert abs(dyn.to_pitch(100.0, 500.0, frame_idx=0).x - 20.0) < 0.5
    assert abs(dyn.to_pitch(300.0, 500.0, timestamp_s=4.0).x - 20.0) < 0.5
