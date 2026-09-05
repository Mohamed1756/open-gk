"""Unit tests for automated full-match sequence segmenter (Module M2/M7).

Verifies coarse-to-fine segmentation and 4 broadcast edge cases:
1. Tactical pitch frame green-ratio filtering
2. Scene cut / camera angle switch detection
3. Bidirectional sliding buffer clustering (capturing 1-touch clearances)
4. Sequence extraction and slicing
"""

from pathlib import Path
import cv2
import numpy as np

from src.cv.match_segmenter import (
    CandidateSequence,
    cluster_possession_triggers,
    detect_scene_cut,
    is_tactical_pitch_frame,
    slice_match_sequences,
)


def test_tactical_pitch_frame_filter() -> None:
    """Verifies that green pitch frames pass and crowd/close-up frames are filtered."""
    # Synthetic tactical pitch frame (predominantly green grass)
    # In BGR: Blue=30, Green=160, Red=40 -> HSV H around 60 (green), high S and V
    pitch_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pitch_frame[:, :] = (35, 150, 45)
    assert is_tactical_pitch_frame(pitch_frame, min_grass_ratio=0.35) is True

    # Synthetic non-pitch frame (crowd/bench: dark gray/red)
    crowd_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crowd_frame[:, :] = (40, 40, 160)
    assert is_tactical_pitch_frame(crowd_frame, min_grass_ratio=0.35) is False

    # Empty frame
    assert is_tactical_pitch_frame(np.array([])) is False


def test_scene_cut_detection() -> None:
    """Verifies that significant frame-to-frame color histogram shifts trigger scene cut."""
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_a[:, :] = (35, 150, 45)  # Green

    frame_b = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b[:, :] = (35, 150, 45)  # Identical

    frame_c = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_c[:, :] = (180, 20, 20)  # Blue cut (different camera angle)

    # Identical frames do not trigger cut
    assert detect_scene_cut(frame_a, frame_b) is False
    # Radically different frames trigger cut
    assert detect_scene_cut(frame_a, frame_c) is True


def test_sequence_clustering_and_buffer() -> None:
    """Verifies that discrete triggers merge and expand with bidirectional buffers."""
    fps = 25.0
    # Group 1: GK touched ball at t=10s, 11s, 12s
    # Group 2: GK touched ball at t=40s, 41s
    triggers = [
        (10.0, 250),
        (11.0, 275),
        (12.0, 300),
        (40.0, 1000),
        (41.0, 1025),
    ]

    sequences = cluster_possession_triggers(
        triggers=triggers,
        pre_roll_s=3.0,
        post_roll_s=12.0,
        max_gap_s=4.0,
        fps=fps,
    )

    assert len(sequences) == 2

    seq1 = sequences[0]
    assert seq1.start_time_s == 7.0  # 10.0 - 3.0
    assert seq1.end_time_s == 24.0  # 12.0 + 12.0
    assert seq1.duration_s == 17.0
    assert seq1.start_frame == int(7.0 * fps)
    assert seq1.end_frame == int(24.0 * fps)

    seq2 = sequences[1]
    assert seq2.start_time_s == 37.0  # 40.0 - 3.0
    assert seq2.end_time_s == 53.0  # 41.0 + 12.0
    assert seq2.duration_s == 16.0


def test_slice_match_sequences_synthetic(tmp_path: Path) -> None:
    """Verifies slicing extracted candidate sequence into a valid 25 Hz video clip."""
    # Create a 50-frame synthetic video (2.0s at 25 fps)
    video_path = tmp_path / "synthetic_match.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (120, 80))
    for i in range(50):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[:, :] = (i * 4, 150, 45)
        out.write(frame)
    out.release()

    candidate = CandidateSequence(
        sequence_id="seq_test_01",
        start_time_s=0.4,
        end_time_s=1.2,
        duration_s=0.8,
        trigger_time_s=0.5,
        start_frame=10,
        end_frame=30,
    )

    output_dir = tmp_path / "clips"
    extracted = slice_match_sequences(
        video_path=video_path,
        sequences=[candidate],
        output_dir=output_dir,
    )

    assert len(extracted) == 1
    clip_file = extracted[0]
    assert clip_file.exists()

    cap = cv2.VideoCapture(str(clip_file))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # 10 to 30 inclusive is 21 frames
    assert frame_count == 21
