"""Full-Match Video Ingestion & Goalkeeper Sequence Slicing CLI.

Scans a 90-minute broadcast video at 1.0 fps, filters non-tactical cuts and replays,
detects goalkeeper possession sequences, and extracts clean 25 Hz clips for evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cv.match_segmenter import (
    scan_match_video_coarse,
    slice_match_sequences,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan full-match video and slice goalkeeper distribution sequences."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/video_raw/serie_a/gw1/torino_vs_milan.mp4",
        help="Path to full match MP4 video file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/video_raw/clips/auto_extracted",
        help="Output directory for sliced 25 Hz MP4 clips.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Coarse scan sampling rate in Hz (default: 1.0 fps).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional max match duration to scan in seconds (e.g. 600 for 10 min test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report candidate sequences without slicing video clips.",
    )

    args = parser.parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    print(
        f"🎬 Ingesting match video: {video_path} ({video_path.stat().st_size / 1e9:.2f} GB)"
    )
    print(
        "🔍 Running coarse 1.0 Hz scan with pitch green-ratio and scene cut filters..."
    )

    candidates = scan_match_video_coarse(
        video_path=video_path,
        sample_fps=args.sample_fps,
        max_duration_s=args.max_duration,
    )

    valid_seqs = [s for s in candidates if s.is_valid]
    print(
        f"\n📊 Scan Complete: Detected {len(candidates)} total triggers, {len(valid_seqs)} valid GK sequences:"
    )
    for s in valid_seqs:
        print(
            f"   • [{s.sequence_id}] {s.start_time_s:6.1f}s -> {s.end_time_s:6.1f}s "
            f"({s.duration_s:4.1f}s) | Trigger @ {s.trigger_time_s:.1f}s"
        )

    if args.dry_run:
        print("\n✅ Dry run complete. Video slicing skipped.")
        return

    print(
        f"\n✂️ Slicing {len(valid_seqs)} sequences into 25 Hz clips in {args.output_dir}..."
    )
    extracted = slice_match_sequences(
        video_path=video_path,
        sequences=valid_seqs,
        output_dir=Path(args.output_dir),
    )
    print(f"✅ Successfully sliced {len(extracted)} clips.")


if __name__ == "__main__":
    main()
