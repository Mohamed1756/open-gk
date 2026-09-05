"""Tests for unit line clustering and jump triggers (hand-built shapes)."""

from src.core.geometry import PitchPoint
from src.models.pressing.types import DefenderRole, PressingActor
from src.models.pressing.units import (
    classify_defender_role,
    cluster_lines,
    press_state,
)


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


def test_classifier_exhaustive_roles():
    ball = PitchPoint(x=100.0, y=29.0)
    # Sprinting at the ball: ENGAGE (head direction irrelevant)
    sprinter = _actor(1, 90.0, 29.0, vx=5.0, vy=0.0)
    assert classify_defender_role(sprinter, ball) is DefenderRole.ENGAGE
    # Between ball and defended goal (x=105), sprinting goalward: RECOVER
    runner = _actor(2, 102.0, 29.0, vx=3.0, vy=0.0)
    assert classify_defender_role(runner, ball) is DefenderRole.RECOVER
    # Sprinting away from the ball but toward a receiver is not recovery:
    # without corridors it settles as CONTAIN, never RECOVER.
    chaser = _actor(6, 88.0, 27.0, vx=-4.0, vy=-2.0)
    assert classify_defender_role(chaser, ball) is DefenderRole.CONTAIN
    # Settled near the ball: CONTAIN
    settler = _actor(3, 94.0, 29.0)
    assert classify_defender_role(settler, ball) is DefenderRole.CONTAIN
    # Far away beyond ball-reach in the horizon: DEEP
    far = _actor(4, 20.0, 29.0)
    assert classify_defender_role(far, ball) is DefenderRole.DEEP
    # Standing on the ray to a corridor target: SCREEN
    rayman = _actor(5, 90.0, 29.0)
    target = PitchPoint(x=70.0, y=29.0)
    assert (
        classify_defender_role(rayman, ball, corridor_targets=[target])
        is DefenderRole.SCREEN
    )
