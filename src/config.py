"""
Global configuration and pitch dimensions for the GK Decision Engine.
Standard canonical coordinate system:
  - Length (X): 0.0m (own goal-line) to 105.0m (opponent goal-line)
  - Width (Y): 0.0m (right touchline) to 68.0m (left touchline) or center at (52.5, 34.0)
  - Canonical orientation: Left-to-right attack (defending goal at X=0, attacking goal at X=105).
"""

# Pitch Dimensions (Meters - FIFA Standard)
PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0

# Goal Dimensions (Meters)
GOAL_WIDTH_METERS = 7.32
GOAL_HEIGHT_METERS = 2.44
GOAL_Y_MIN = (PITCH_WIDTH_METERS - GOAL_WIDTH_METERS) / 2.0  # 30.34m
GOAL_Y_MAX = (PITCH_WIDTH_METERS + GOAL_WIDTH_METERS) / 2.0  # 37.66m
GOAL_Y_CENTER = PITCH_WIDTH_METERS / 2.0  # 34.0m

# Penalty Box Dimensions (Meters)
# Extends 16.5m from goal line, 16.5m from each goal post (40.32m wide)
PENALTY_BOX_LENGTH = 16.5
PENALTY_BOX_WIDTH = 40.32
PENALTY_BOX_Y_MIN = (PITCH_WIDTH_METERS - PENALTY_BOX_WIDTH) / 2.0  # 13.84m
PENALTY_BOX_Y_MAX = (PITCH_WIDTH_METERS + PENALTY_BOX_WIDTH) / 2.0  # 54.16m

# 6-Yard Box (Goal Area) Dimensions (Meters)
# Extends 5.5m from goal line, 5.5m from each goal post (18.32m wide)
SIX_YARD_BOX_LENGTH = 5.5
SIX_YARD_BOX_WIDTH = 18.32
SIX_YARD_BOX_Y_MIN = (PITCH_WIDTH_METERS - SIX_YARD_BOX_WIDTH) / 2.0  # 24.84m
SIX_YARD_BOX_Y_MAX = (PITCH_WIDTH_METERS + SIX_YARD_BOX_WIDTH) / 2.0  # 43.16m

# Penalty Spot
PENALTY_SPOT_X = 11.0
PENALTY_SPOT_Y = GOAL_Y_CENTER

# Temporal / Sampling Parameters
DEFAULT_SAMPLING_RATE_HZ = 25.0
CONTEXT_WINDOW_SECONDS_BEFORE = 2.0
CONTEXT_WINDOW_SECONDS_AFTER = 2.0

# IFAB Rule Parameters
IFAB_GK_HOLD_LIMIT_SECONDS = 8.0
IFAB_COUNTDOWN_WARNING_SECONDS = 5.0
