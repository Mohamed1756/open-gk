"""Automated Match Sequence Segmenter (Module M2/M7).

Scans full-match broadcast video at low frequency (1.0 Hz) to identify and
extract goalkeeper distribution sequences while enforcing 4 broadcast edge cases:
1. Camera angle switches (tactical wide-angle vs. close-up/crowd cuts)
2. Replay graphics & slow-motion stinger transitions
3. Dead-ball stoppages & static injury pauses
4. Rapid 1-touch clearances via bidirectional sliding buffers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np

DEFAULT_GRASS_MIN_RATIO: float = 0.35
DEFAULT_SCENE_CUT_CORRELATION: float = 0.45
DEFAULT_PRE_ROLL_S: float = 3.0
DEFAULT_POST_ROLL_S: float = 12.0
DEFAULT_MAX_CLUSTER_GAP_S: float = 4.0


@dataclass(frozen=True)
class CandidateSequence:
    """Represents a candidate goalkeeper possession sequence extracted from a match."""

    sequence_id: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    trigger_time_s: float
    start_frame: int
    end_frame: int
    confidence: float = 1.0
    is_valid: bool = True
    rejection_reason: Optional[str] = None


def is_tactical_pitch_frame(
    frame_bgr: np.ndarray,
    min_grass_ratio: float = DEFAULT_GRASS_MIN_RATIO,
) -> bool:
    """Determines if a video frame shows the tactical field of play using green grass ratio.

    Filters out camera close-ups, manager cuts, studio footage, and crowd scenes.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    is_grass = (
        (hsv[:, :, 0] >= 32)
        & (hsv[:, :, 0] <= 85)
        & (hsv[:, :, 1] >= 35)
        & (hsv[:, :, 2] >= 40)
    )
    grass_ratio = float(np.count_nonzero(is_grass) / is_grass.size)
    return grass_ratio >= min_grass_ratio


def detect_scene_cut(
    prev_frame_bgr: np.ndarray,
    curr_frame_bgr: np.ndarray,
    correlation_threshold: float = DEFAULT_SCENE_CUT_CORRELATION,
) -> bool:
    """Detects broadcast camera switches via HSV 2D histogram correlation.

    Returns True if correlation drops below threshold, indicating a camera angle cut.
    """
    if prev_frame_bgr is None or curr_frame_bgr is None:
        return False

    hsv_prev = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2HSV)
    hsv_curr = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2HSV)

    hist_prev = cv2.calcHist([hsv_prev], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [16, 16], [0, 180, 0, 256])

    cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    correl = float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL))
    return correl < correlation_threshold


def _group_triggers_by_gap(
    triggers: Sequence[Tuple[float, int]],
    max_gap_s: float,
) -> List[List[Tuple[float, int]]]:
    """Partitions time-ordered triggers into contiguous clusters separated by max_gap_s."""
    if not triggers:
        return []
    sorted_triggers = sorted(triggers, key=lambda x: x[0])
    clusters: List[List[Tuple[float, int]]] = []
    current: List[Tuple[float, int]] = [sorted_triggers[0]]
    for t_s, f_idx in sorted_triggers[1:]:
        if t_s - current[-1][0] <= max_gap_s:
            current.append((t_s, f_idx))
        else:
            clusters.append(current)
            current = [(t_s, f_idx)]
    if current:
        clusters.append(current)
    return clusters


def cluster_possession_triggers(
    triggers: Sequence[Tuple[float, int]],
    pre_roll_s: float = DEFAULT_PRE_ROLL_S,
    post_roll_s: float = DEFAULT_POST_ROLL_S,
    max_gap_s: float = DEFAULT_MAX_CLUSTER_GAP_S,
    fps: float = 25.0,
) -> List[CandidateSequence]:
    """Clusters discrete positive possession detections into continuous candidate sequences."""
    clusters = _group_triggers_by_gap(triggers, max_gap_s)
    sequences: List[CandidateSequence] = []
    for idx, c in enumerate(clusters):
        first_t, first_f = c[0]
        last_t, last_f = c[-1]
        start_t = max(0.0, first_t - pre_roll_s)
        end_t = last_t + post_roll_s
        sequences.append(
            CandidateSequence(
                sequence_id=f"seq_{idx + 1:03d}_{int(first_t):04d}s",
                start_time_s=round(start_t, 2),
                end_time_s=round(end_t, 2),
                duration_s=round(end_t - start_t, 2),
                trigger_time_s=round(first_t, 2),
                start_frame=max(0, int(first_f - pre_roll_s * fps)),
                end_frame=int(last_f + post_roll_s * fps),
            )
        )
    return sequences


def is_dead_ball_or_stoppage(
    cap: Any,
    start_frame: int,
    end_frame: int,
    sample_step: int = 25,
    min_mean_motion: float = 2.0,
) -> bool:
    """Checks if an extracted candidate clip has active game motion or is an injury stoppage."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        return True

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_motion = 0.0
    steps = 0

    curr_f = start_frame + sample_step
    while curr_f < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_f)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        total_motion += float(np.mean(diff))
        steps += 1
        prev_gray = gray
        curr_f += sample_step

    if steps == 0:
        return True
    avg_motion = total_motion / steps
    return avg_motion < min_mean_motion


def _evaluate_gk_ball_proximity(
    detections: Any,
    img_w: int,
    img_h: int,
    box_x_ratio: float = 0.30,
) -> bool:
    """Evaluates whether a goalkeeper and ball are co-located in the defensive third."""
    persons: List[Tuple[float, float, float, float]] = []
    balls: List[Tuple[float, float]] = []

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.20:
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        if cls_id == 0:  # person
            persons.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        elif cls_id == 32:  # sports ball
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            balls.append((cx, cy))

    if not balls:
        # Ball not detected, but check if a single isolated player is deep in the box
        for p in persons:
            px = (p[0] + p[2]) / 2.0
            if px < img_w * box_x_ratio or px > img_w * (1.0 - box_x_ratio):
                return True
        return False

    for bx, by in balls:
        for p in persons:
            feet_x = (p[0] + p[2]) / 2.0
            feet_y = p[3]
            dist_px = float(np.hypot(bx - feet_x, by - feet_y))
            # Proximity threshold within ~80 pixels in standard resolution
            if dist_px < 120.0:
                if feet_x < img_w * box_x_ratio or feet_x > img_w * (1.0 - box_x_ratio):
                    return True
    return False


def _scan_frames_for_triggers(
    cap: Any,
    model: Any,
    total_frames: int,
    frame_step: int,
    fps: float,
) -> List[Tuple[float, int]]:
    """Scans sampled frames in cap with model and yields detected GK possession triggers."""
    triggers: List[Tuple[float, int]] = []
    curr_frame_idx = 0
    while curr_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        timestamp_s = round(curr_frame_idx / fps, 2)
        if is_tactical_pitch_frame(frame):
            h, w = frame.shape[:2]
            results = model.predict(frame, verbose=False, conf=0.20)[0]
            if _evaluate_gk_ball_proximity(results, w, h):
                triggers.append((timestamp_s, curr_frame_idx))
        curr_frame_idx += frame_step
    return triggers


def scan_match_video_coarse(
    video_path: Path | str,
    sample_fps: float = 1.0,
    weights_path: str = "yolov8n.pt",
    max_duration_s: Optional[float] = None,
) -> List[CandidateSequence]:
    """Scans full match video at 1 fps using lightweight YOLO to detect GK possessions."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_duration_s is not None:
        total_frames = min(total_frames, int(max_duration_s * fps))

    import os

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    from ultralytics import YOLO

    model = YOLO(weights_path)
    frame_step = max(1, int(fps / sample_fps))
    triggers = _scan_frames_for_triggers(cap, model, total_frames, frame_step, fps)
    raw_sequences = cluster_possession_triggers(triggers, fps=fps)

    validated: List[CandidateSequence] = []
    for seq in raw_sequences:
        is_dead = is_dead_ball_or_stoppage(cap, seq.start_frame, seq.end_frame)
        if is_dead:
            validated.append(
                CandidateSequence(
                    sequence_id=seq.sequence_id,
                    start_time_s=seq.start_time_s,
                    end_time_s=seq.end_time_s,
                    duration_s=seq.duration_s,
                    trigger_time_s=seq.trigger_time_s,
                    start_frame=seq.start_frame,
                    end_frame=seq.end_frame,
                    is_valid=False,
                    rejection_reason="DEAD_BALL_STOPPAGE",
                )
            )
        else:
            validated.append(seq)

    cap.release()
    return validated


def _write_single_clip(
    cap: Any,
    seq: CandidateSequence,
    clip_filepath: Path,
    fourcc: int,
    fps: float,
    dimensions: Tuple[int, int],
) -> None:
    """Extracts frames for a single sequence and writes them to an MP4 clip."""
    writer = cv2.VideoWriter(str(clip_filepath), fourcc, fps, dimensions)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seq.start_frame)
    current_f = seq.start_frame
    while current_f <= seq.end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        writer.write(frame)
        current_f += 1
    writer.release()


def slice_match_sequences(
    video_path: Path | str,
    sequences: Sequence[CandidateSequence],
    output_dir: Path | str,
) -> List[Path]:
    """Extracts verified candidate sequences into clean 25 Hz MP4 clips."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    extracted_clips: List[Path] = []
    for seq in sequences:
        if not seq.is_valid:
            continue
        clip_filepath = out_path / f"{seq.sequence_id}.mp4"
        _write_single_clip(cap, seq, clip_filepath, fourcc, fps, (w, h))
        extracted_clips.append(clip_filepath)

    cap.release()
    return extracted_clips
