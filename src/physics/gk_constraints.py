"""
Goalkeeper biomechanical and physics constants.
Legacy geometric constants for backward compatibility with geometry modules.
"""

from __future__ import annotations

# Legacy geometric constants (retained for backward compatibility with geometry modules)
GK_REACTION_TIME_FLOOR_S: float = 0.20
GK_REACTION_TIME_DEFAULT_S: float = 0.25
GK_LATERAL_BURST_SPEED_MAX_MS: float = 5.2
GK_LATERAL_BURST_SPEED_DEFAULT_MS: float = 4.2
GK_WINGSPAN_REACH_DEFAULT_M: float = 1.2
GK_DIVE_RECOVERY_TIME_S: float = 1.1
GK_SET_POSITION_LATENCY_S: float = 0.3
GK_MAX_BURST_ACCELERATION_MSS: float = 3.5

__all__ = [
    "GK_REACTION_TIME_FLOOR_S",
    "GK_REACTION_TIME_DEFAULT_S",
    "GK_LATERAL_BURST_SPEED_MAX_MS",
    "GK_LATERAL_BURST_SPEED_DEFAULT_MS",
    "GK_WINGSPAN_REACH_DEFAULT_M",
    "GK_DIVE_RECOVERY_TIME_S",
    "GK_SET_POSITION_LATENCY_S",
    "GK_MAX_BURST_ACCELERATION_MSS",
]
