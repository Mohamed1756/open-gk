#!/usr/bin/env python3.11
"""
Rater crop extraction for the gaze-yaw labeling protocol (tests/golden/).

Samples keeper crops across pre-decision windows under blind filenames; the
episode/frame truth lives only in key.json, which raters never see. Rater-facing
manifest rows carry no truth. Target: 200 frames. See scripts/deal_gaze_manifest.py
for the blind A/B/overlap split.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_match import get_match_config
from scripts.evaluate_distribution import _split_frame, _team_sets

EPISODES = ["ep21", "ep2500", "ep7818"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/gaze_rating")
    parser.add_argument("--per-episode", type=int, default=70)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    try:
        import cv2
    except ImportError as e:
        raise ImportError(f"Crop extraction needs OpenCV: {e}") from e

    out_dir = Path(args.out)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    manifest = []
    key = {}

    for ep in EPISODES:
        match = get_match_config(ep)
        cache = Path(f"data/processed/{match.match_id}_tracks.json")
        if not cache.exists():
            print(f"[{ep}] missing cache {cache}, skipping")
            continue
        records = json.loads(cache.read_text())
        fps = float(records[0].get("fps", 25.0))
        decision_idx = min(
            len(records) - 1, int(round(float(match.decision_time_s) * fps))
        )
        possession, opponent, _ = _team_sets(match)
        window = [
            fi
            for fi in range(max(0, decision_idx - int(3.5 * fps)), decision_idx + 1)
            if fi < len(records)
        ]
        rng.shuffle(window)
        cap = cv2.VideoCapture(str(match.video_path))
        if not cap.isOpened():
            print(f"[{ep}] cannot open {match.video_path}, skipping")
            continue
        taken = 0
        for fi in window:
            if taken >= args.per_episode:
                break
            gk, _, _ = _split_frame(
                records[fi]["entities"], possession, opponent, match
            )
            if gk is None or "bbox" not in gk:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = (int(v) for v in gk["bbox"])
            crop = frame[max(0, y1) : y2, max(0, x1) : x2]
            if crop.size == 0:
                continue
            crop_id = f"crop_{secrets.token_hex(4)}"
            name = f"{crop_id}.jpg"
            cv2.imwrite(str(out_dir / "crops" / name), crop)
            key[crop_id] = {
                "episode": match.match_id,
                "frame": fi,
                "bbox": [x1, y1, x2, y2],
            }
            manifest.append(
                {
                    "id": crop_id,
                    "crop": name,
                    "yaw_deg": None,
                    "view": None,
                    "head_px": None,
                }
            )
            taken += 1
        cap.release()
        print(f"[{ep}] wrote {taken} crops")

    rng.shuffle(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "key.json").write_text(json.dumps(key, indent=2))
    print(f"Manifest: {len(manifest)} blind rows -> {out_dir}/manifest.json")
    print(f"Sealed key: {len(key)} rows -> {out_dir}/key.json (raters never see this)")


if __name__ == "__main__":
    main()
