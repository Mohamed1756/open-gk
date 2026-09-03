"""Tests for unit line clustering and jump triggers (hand-built shapes)."""

from src.core.geometry import PitchPoint
from src.models.pressing.types import PressingActor
from src.models.pressing.units import cluster_lines, press_state


def _actor(tid, x, y, vx=0.0, vy=0.0, team="opp"):
    return PressingActor(
        track_id=tid, team_id=team, pos_m=PitchPoint(x=x, y=y), vel_ms=(vx, vy)
    )


def test_detects_3_2_press_shape():
    opponents = [
        _actor(1, 92.0, 20.0),
        _actor(2, 93.0, 30.0),
        _actor(3, 91.0, 40.0),
        _actor(4, 80.0, 25.0),
        _actor(5, 81.0, 36.0),
    ]
    lines = cluster_lines(opponents, "opp", attack_dir_x=1.0)
    assert lines.line_counts == (3, 2)
    assert lines.label == "3-2"


def test_detects_buildup_5():
    own = [_actor(i, 95.0 + i, 20.0 + i * 6.0, team="own") for i in range(5)]
    lines = cluster_lines(own, "own", attack_dir_x=-1.0)
    assert sum(lines.line_counts) == 5


def test_jump_trigger_on_first_line_sprint():
    opponents = [
        _actor(1, 92.0, 28.0, vx=5.0, vy=0.0),
        _actor(2, 92.5, 32.0, vx=5.5, vy=0.0),
        _actor(3, 80.0, 30.0, vx=0.5),
    ]
    state = press_state(opponents, ball_x=100.0, ball_y=29.0, attack_dir_x=1.0)
    assert state.jumping
    assert state.first_line_count == 2
    assert state.intensity_ms >= 4.0


def test_set_press_when_static():
    opponents = [_actor(1, 92.0, 28.0), _actor(2, 80.0, 30.0)]
    state = press_state(opponents, ball_x=100.0, ball_y=29.0, attack_dir_x=1.0)
    assert not state.jumping
    assert state.note == "set"


def test_empty_unit_labels_zero():
    assert cluster_lines([], "opp").label == "0"
    state = press_state([], ball_x=100.0, ball_y=29.0)
    assert state.note == "no-press"
