"""
Unit tests for action detection, situational classification, and context window slicing.
"""

import pytest
from src.segmentation.detector import GKActionDetector
from src.segmentation.window import ContextWindowBuilder
from src.core.types import GKDecision, CrossOutcome, ShotOutcome


def test_cross_situation_detection():
    mock_events = [
        {
            "id": "cross-1",
            "type": {"name": "Pass"},
            "minute": 15,
            "second": 30,
            "location": [105.0, 10.0],
            "pass": {
                "cross": True,
                "end_location": [115.0, 40.0],
                "technique": {"name": "Inswinging"},
            },
            "team": {"id": 1, "name": "Arsenal"},
        },
        {
            "id": "claim-1",
            "type": {"name": "Goal Keeper"},
            "minute": 15,
            "second": 32,
            "goalkeeper": {"type": {"name": "Claim"}},
            "player": {"id": 99, "name": "David Raya"},
            "team": {"id": 2, "name": "Chelsea"},
        },
    ]

    detector = GKActionDetector()
    crosses = detector.extract_cross_situations(mock_events)

    assert len(crosses) == 1
    c = crosses[0]
    assert c.decision == GKDecision.COME
    assert c.outcome == CrossOutcome.CLAIM_CLEAN
    assert c.gk_player_name == "David Raya"


def test_shot_situation_detection():
    mock_events = [
        {
            "id": "shot-1",
            "type": {"name": "Shot"},
            "minute": 42,
            "second": 10,
            "location": [108.0, 40.0],
            "shot": {
                "statsbomb_xg": 0.35,
                "outcome": {"name": "Saved"},
            },
            "team": {"id": 1, "name": "Arsenal"},
        }
    ]

    detector = GKActionDetector()
    shots = detector.extract_shot_situations(mock_events)

    assert len(shots) == 1
    s = shots[0]
    assert s.xg == pytest.approx(0.35)
    assert s.outcome == ShotOutcome.SAVED


def test_tempo_situation_detection():
    mock_events = [
        {
            "id": "event-dead",
            "type": {"name": "Foul Won"},
            "minute": 70,
            "second": 0,
        },
        {
            "id": "gk-pass",
            "type": {"name": "Pass"},
            "minute": 70,
            "second": 9,
            "position": {"name": "Goalkeeper"},
            "player": {"id": 99, "name": "Keeper"},
            "team": {"id": 2, "name": "Chelsea"},
            "pass": {"type": {"name": "Goal Kick"}},
        },
    ]

    detector = GKActionDetector()
    tempos = detector.extract_tempo_situations(mock_events)

    assert len(tempos) == 1
    t = tempos[0]
    assert t.latency_seconds == pytest.approx(9.0)
    assert t.restart_type == "goal_kick"


def test_context_window_builder():
    builder = ContextWindowBuilder(seconds_before=2.0, seconds_after=2.0)
    window = builder.create_window(action_timestamp=100.0)

    assert window.start_time == 98.0
    assert window.end_time == 102.0
