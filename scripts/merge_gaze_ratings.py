#!/usr/bin/env python3.11
"""
Merge blind rater labels into the measured golden set.

Joins labels_A/B.json via the sealed key, hashes crop files for provenance,
prints inter-rater agreement on the shared overlap, and writes
tests/golden/gaze_rater_labels.json. Refuses to overwrite without --force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cv.pose_estimator import inter_rater_mae_deg

VIEWS = {"front", "side", "back"}


def _load_labels(path: Path, rater: str) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    out = {}
    missing = []
    for row in rows:
        if (
            row.get("yaw_deg") is None
            or row.get("view") not in VIEWS
            or row.get("head_px") is None
        ):
            missing.append(row.get("id"))
            continue
        out[row["id"]] = row
    if missing:
        raise ValueError(f"{path}: {len(missing)} unlabeled rows, e.g. {missing[:5]}")
    for row in out.values():
        row["rater"] = rater
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/gaze_rating")
    parser.add_argument("--out", default="tests/golden/gaze_rater_labels.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    work = Path(args.dir)
    key = json.loads((work / "key.json").read_text())
    lab_a = _load_labels(work / "labels_A.json", "A")
    lab_b = _load_labels(work / "labels_B.json", "B")

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; re-run with --force to overwrite")

    def _sha(crop: str) -> str:
        return hashlib.sha256((work / "crops" / crop).read_bytes()).hexdigest()

    rows = []
    for rid, lab in (("A", lab_a), ("B", lab_b)):
        for cid, row in lab.items():
            truth = key[cid]
            rows.append(
                {
                    "crop_sha256": _sha(row["crop"]),
                    "rater_id": rid,
                    "labeled_at": row.get("saved_at"),
                    "yaw_deg": float(row["yaw_deg"]),
                    "view": row["view"],
                    "head_px": float(row["head_px"]),
                    "episode": truth["episode"],
                    "frame": int(truth["frame"]),
                }
            )
    if any(r["labeled_at"] is None for r in rows):
        raise ValueError("rows missing saved_at timestamps")

    overlap = sorted(set(lab_a) & set(lab_b))
    pairs = [(lab_a[c]["yaw_deg"], lab_b[c]["yaw_deg"]) for c in overlap]
    print(
        f"rows: {len(rows)} (A: {len(lab_a)}, B: {len(lab_b)}, overlap: {len(overlap)})"
    )
    print(f"inter-rater MAE on overlap: {inter_rater_mae_deg(pairs):.2f} deg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
