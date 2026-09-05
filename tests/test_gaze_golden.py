"""
Pose-geometry math test against frozen SYNTHETIC fixtures (not sensor validation).
Measured rater labels live in tests/golden/gaze_rater_labels.json, gated below.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.cv.pose_estimator import (
    binned_mae_deg,
    compute_detailed_pose_orientation,
    gaze_yaw_abs_error_deg,
    head_size_bin,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pose_geometry.json"
RATER_PATH = Path(__file__).parent / "golden" / "gaze_rater_labels.json"
BIN_TOL_DEG = {"<8px": 30.0, "8-15px": 15.0, ">15px": 10.0}


def _load_fixtures():
    return json.loads(FIXTURE_PATH.read_text())["fixtures"]


def test_rater_labels_provenance_guard():
    """Measured set admits only rows with crop/rater/timestamp provenance that
    match no synthetic fixture hash. Synthetic rows are structurally unmergeable."""
    import pytest

    if not RATER_PATH.exists():
        pytest.skip("no measured rater labels yet")
    payload = json.loads(RATER_PATH.read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    synthetic_hashes = {
        hashlib.sha256(
            json.dumps(fix["keypoints"], sort_keys=True).encode()
        ).hexdigest()
        for fix in _load_fixtures()
    }
    assert len(rows) > 0
    for row in rows:
        assert row.get("crop_sha256"), row
        assert row.get("rater_id"), row
        assert row.get("labeled_at"), row
        if "keypoints" in row:
            kp_hash = hashlib.sha256(
                json.dumps(row["keypoints"], sort_keys=True).encode()
            ).hexdigest()
            assert kp_hash not in synthetic_hashes, row


def test_golden_gaze_yaw_accuracy():
    records = []
    for fix in _load_fixtures():
        kp = np.array(fix["keypoints"], dtype=np.float32)
        res = compute_detailed_pose_orientation(kp, homography=None)
        assert res.is_valid is True, fix["id"]
        assert res.gaze_valid is bool(fix["expected_gaze_valid"]), fix["id"]
        if not fix["expected_gaze_valid"]:
            continue
        err_gaze = gaze_yaw_abs_error_deg(
            float(fix["expected_gaze_deg"]), res.gaze_facing_rad
        )
        err_torso = gaze_yaw_abs_error_deg(
            float(fix["expected_torso_deg"]), res.torso_facing_rad
        )
        tol = BIN_TOL_DEG[head_size_bin(float(fix["head_px"]))]
        assert err_gaze <= tol, f"{fix['id']}: gaze err {err_gaze:.1f} > {tol}"
        assert err_torso <= tol, f"{fix['id']}: torso err {err_torso:.1f} > {tol}"
        records.append((head_size_bin(float(fix["head_px"])), err_gaze))
    mae = binned_mae_deg(records)
    assert mae[">15px"] <= 10.0
    assert math.isfinite(sum(mae.values()))


def test_inter_rater_metric_bounds():
    import pytest

    from src.cv.pose_estimator import inter_rater_mae_deg

    assert inter_rater_mae_deg([(90.0, 90.0), (45.0, 50.0)]) == pytest.approx(2.5)
    assert inter_rater_mae_deg([(179.0, -179.0)]) == pytest.approx(2.0)
    assert inter_rater_mae_deg([]) == 0.0
