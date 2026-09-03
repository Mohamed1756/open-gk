"""
Enumerations and core type definitions.
"""

from enum import Enum


class ActionType(str, Enum):
    """Goalkeeper primary action classifications."""

    SHOT_FACED = "shot_faced"
    CROSS_FACED = "cross_faced"
    SWEEP_THROUGHBALL = "sweep_throughball"
    DISTRIBUTION = "distribution"
    RESTART_TEMPO = "restart_tempo"


class GKDecision(str, Enum):
    """Decision choice made by the goalkeeper."""

    COME = "come"  # Claim, punch, rush-out
    STAY = "stay"  # Hold line, hold position
    UNKNOWN = "unknown"


class CrossOutcome(str, Enum):
    """Outcome of a cross situation."""

    CLAIM_CLEAN = "claim_clean"
    PUNCH_CLEAR = "punch_clear"
    FUMBLE_DANGER = "fumble_danger"
    GOAL_CONCEDED = "goal_conceded"
    HEADER_SHOT = "header_shot"
    CLEARED_BY_DEFENDER = "cleared_by_defender"
    OUT_OF_BOUNDS = "out_of_bounds"
    MISSED_CROSS = "missed_cross"


class ShotOutcome(str, Enum):
    """Outcome of a shot faced."""

    SAVED = "saved"
    SAVED_FUMBLED = "saved_fumbled"
    GOAL = "goal"
    OFF_TARGET = "off_target"
    BLOCKED = "blocked"
    POST_CROSSBAR = "post_crossbar"


class DistributionType(str, Enum):
    """Types of GK distributions."""

    GOAL_KICK = "goal_kick"
    THROW = "throw"
    PUNT = "punt"
    SHORT_PASS = "short_pass"
    LONG_PASS = "long_pass"


class DistributionOutcome(str, Enum):
    """Outcome of a distribution."""

    SUCCESS_RETAINED = "success_retained"
    TURNOVER = "turnover"
    OUT_OF_BOUNDS = "out_of_bounds"
    INTERCEPTED_DANGER = "intercepted_danger"


class SweepOutcome(str, Enum):
    """Outcome of a sweeping / through-ball situation."""

    INTERCEPTION_CLEAN = "interception_clean"
    CLEARANCE_TOUCH = "clearance_touch"
    TACKLE_WON = "tackle_won"
    FOUL_CONCEDED = "foul_conceded"
    MISSED_SWEEP = "missed_sweep"
    NO_ACTION_PASSIVE = "no_action_passive"
    OPPONENT_SHOT = "opponent_shot"


class Period(int, Enum):
    """Match period."""

    FIRST_HALF = 1
    SECOND_HALF = 2
    EXTRA_TIME_FIRST_HALF = 3
    EXTRA_TIME_SECOND_HALF = 4
    PENALTY_SHOOTOUT = 5
