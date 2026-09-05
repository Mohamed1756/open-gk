"""Unit tests for evaluate_distribution wiring helpers (no video/model needed)."""

from src.core.geometry import PitchPoint
from scripts.evaluate_distribution import (
    _actors_from_entities,
    _build_duel_payload,
    _pose_torso_facing,
    _shared_fate_notes,
)


class _StubResult:
    def __init__(self, is_valid, torso_facing_rad=1.0):
        self.is_valid = is_valid
        self.torso_facing_rad = torso_facing_rad


class _StubEstimator:
    def __init__(self, is_valid=True):
        self.is_valid = is_valid
        self.calls = 0

    def estimate_pose_in_crop(
        self, frame, bbox, frame_idx=None, velocity_screen_uv=None
    ):
        self.calls += 1
        return _StubResult(self.is_valid)


def _ent(tid, team="opp"):
    return {
        "track_id": tid,
        "class_name": "person",
        "team_label": team,
        "pitch_xy": [80.0, 30.0],
        "pitch_vel_ms": [0.0, 0.0],
        "screen_uv": [100.0, 200.0],
        "bbox": [90.0, 150.0, 130.0, 250.0],
        "pitch_valid": True,
    }


def test_pose_torso_facing_returns_pose_tuple():
    est = _StubEstimator(is_valid=True)
    out = _pose_torso_facing(est, object(), _ent(1), [], 0)
    assert out == (1.0, "pose")
    assert est.calls == 1


def test_pose_torso_facing_none_when_invalid_or_missing_bbox():
    assert (
        _pose_torso_facing(_StubEstimator(is_valid=False), object(), _ent(1), [], 0)
        is None
    )
    ent = _ent(2)
    del ent["bbox"]
    assert _pose_torso_facing(_StubEstimator(), object(), ent, [], 0) is None


def test_actors_attach_facing_from_map_and_leave_unmapped_none():
    actors = _actors_from_entities([_ent(1), _ent(2)], "opp", {1: (0.5, "pose")})
    by_id = {a.track_id: a for a in actors}
    assert by_id[1].facing_rad == 0.5
    assert by_id[1].facing_source == "pose"
    assert by_id[2].facing_rad is None


def _person(tid, label, x, y):
    return {
        "track_id": tid,
        "class_name": "person",
        "team_label": label,
        "pitch_xy": [x, y],
        "pitch_vel_ms": [0.0, 0.0],
        "screen_uv": [100.0, 200.0],
        "bbox": [90.0, 150.0, 130.0, 250.0],
        "pitch_valid": True,
    }


def _ball(x, y):
    return {
        "track_id": 0,
        "class_name": "sports ball",
        "team_label": "Match Ball",
        "pitch_xy": [x, y],
    }


def _records(n_frames, ball_xy, people):
    return [
        {
            "frame": fi,
            "timestamp_s": round(fi / 25.0, 3),
            "entities": list(people) + [ball_xy(fi)],
        }
        for fi in range(n_frames)
    ]


def _scenario():
    people = [
        _person(1, "own", 100.0, 29.0),
        _person(9, "own", 86.0, 25.0),
        _person(5, "own", 70.0, 50.0),
        _person(2, "opp", 88.0, 27.0),
        _person(3, "opp", 95.0, 30.0),
    ]

    def ball_xy(fi):
        if fi <= 10:
            return _ball(100.0, 29.0)
        if fi <= 15:
            f = (fi - 10) / 5.0
            return _ball(100.0 - 14.0 * f, 29.0 - 4.0 * f)
        return _ball(86.0, 25.0)

    records = _records(120, ball_xy, people)
    receivers = [
        {"track_id": 9, "pitch_xy": [86.0, 25.0], "pitch_vel_ms": [0.0, 0.0]},
        {"track_id": 5, "pitch_xy": [70.0, 50.0], "pitch_vel_ms": [0.0, 0.0]},
    ]
    evaluation = [
        {"track_id": 9, "presser_track_id": 2},
        {"track_id": 5, "presser_track_id": 3},
    ]
    return records, receivers, evaluation


def test_duel_payload_timelines_and_release():
    records, receivers, evaluation = _scenario()
    payload = _build_duel_payload(
        records,
        10,
        25.0,
        1,
        PitchPoint(x=100.0, y=29.0),
        receivers,
        evaluation,
        1.0,
    )
    assert set(payload["duels"]) == {"9", "5"}
    assert len(payload["duels"]["9"]["margins"]) == 11
    assert payload["duels"]["9"]["presser"] == 2
    assert payload["releasedOutlet"] == 9
    outcome = payload["duelOutcome"]
    assert outcome["outcome"] == "RETAINED"
    assert outcome["controller"] == 9
    assert outcome["contested"] is False


def test_duel_payload_empty_without_records():
    payload = _build_duel_payload(
        [], 0, 25.0, None, PitchPoint(x=0.0, y=0.0), [], [], 1.0
    )
    assert payload == {"duels": {}, "releasedOutlet": None, "duelOutcome": None}


def _window_records(people_fn, n=11):
    return [
        {
            "frame": fi,
            "timestamp_s": round(fi / 25.0, 3),
            "entities": people_fn(fi),
        }
        for fi in range(n)
    ]


def _p(tid, label, x, y):
    return {
        "track_id": tid,
        "class_name": "person",
        "team_label": label,
        "pitch_xy": [x, y],
        "pitch_vel_ms": [0.0, 0.0],
        "screen_uv": [100.0, 200.0],
        "bbox": [90.0, 150.0, 130.0, 250.0],
        "pitch_valid": True,
    }


def _same_lane_people(fi):
    return [
        _p(1, "own", 100.0, 29.0),
        _p(9, "own", 86.0, 28.0),
        _p(5, "own", 86.0, 30.0),
        _p(7, "opp", 93.0, 29.0),
    ]


def _split_mark_people(fi):
    return [
        _p(1, "own", 100.0, 29.0),
        _p(9, "own", 86.0, 28.0),
        _p(5, "own", 86.0, 30.0),
        _p(7, "opp", 93.0, 28.2),
        _p(8, "opp", 93.0, 29.8),
    ]


def test_shared_fate_flags_same_lane_same_presser():
    records = _window_records(_same_lane_people)
    shared = _shared_fate_notes(records, list(range(11)), 1, [9, 5], (100.0, 29.0))
    assert shared[9] == [5]
    assert shared[5] == [9]


def test_shared_fate_clean_with_split_marking():
    records = _window_records(_split_mark_people)
    shared = _shared_fate_notes(records, list(range(11)), 1, [9, 5], (100.0, 29.0))
    assert shared[9] == []
    assert shared[5] == []


def test_shared_fate_empty_without_frames_or_outlets():
    assert _shared_fate_notes([], [], 1, [9, 5], (100.0, 29.0)) == {
        9: [],
        5: [],
    }
    assert _shared_fate_notes([], [0], 1, [9], (100.0, 29.0)) == {9: []}


def test_duel_payload_no_release_when_keeper_holds():
    records, receivers, evaluation = _scenario()
    held = [
        {
            "frame": fi,
            "timestamp_s": round(fi / 25.0, 3),
            "entities": [
                _person(1, "own", 100.0, 29.0),
                _person(9, "own", 86.0, 25.0),
                _person(2, "opp", 88.0, 27.0),
                _ball(100.0, 29.0),
            ],
        }
        for fi in range(30)
    ]
    payload = _build_duel_payload(
        held,
        10,
        25.0,
        1,
        PitchPoint(x=100.0, y=29.0),
        [receivers[0]],
        [evaluation[0]],
        1.0,
    )
    assert payload["releasedOutlet"] is None
    assert payload["duelOutcome"] is None
