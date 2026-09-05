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

# Receiver first-touch and control limits
# McMorris et al. / elite motor control first-touch deceleration window
RECEIVER_FIRST_TOUCH_LATENCY_S: float = 0.45
# Minimum post-receipt cushion before an arriving presser disrupts ball control
MIN_POST_RECEIPT_CUSHION_S: float = 0.20

# Empirical spatial turnover hazard parameters (conceded xG from turnover location)
# StatsBomb/Opta empirical conceded xG baseline from central Zone 14 turnovers
CENTRAL_TURNOVER_PEAK_XG: float = 0.55
CENTRAL_ZONE_SIGMA_X_M: float = 16.0
CENTRAL_ZONE_SIGMA_Y_M: float = 11.0
TURNOVER_HAZARD_FLOOR_XG: float = 0.03

# Continuous first-touch retention logistic curve parameters
# Steepness over a ~200ms physical disruption window
RETENTION_LOGISTIC_STEEPNESS_K: float = 15.0
# Cushion threshold where retention drops to 50% under direct tackle
RETENTION_CRITICAL_CUSHION_S: float = 0.15

# Downstream receiver exit affordance bounds
EXIT_AFFORDANCE_MIN_DIST_M: float = (
    4.0  # Minimum viable separation to receive a downstream pass
)
EXIT_AFFORDANCE_MAX_DIST_M: float = (
    50.0  # Maximum realistic ground pass distance from build-up
)
EXIT_AFFORDANCE_BACKWARD_TOLERANCE_M: float = (
    2.0  # Allowable backward offset for square/lateral passing
)
EXIT_AFFORDANCE_PINNED_MULT: float = (
    0.75  # Progression discount when receiver has 0 unblocked exit lanes
)
EXIT_AFFORDANCE_BASE_MULT: float = 0.85  # Base multiplier for open exit affordance
EXIT_AFFORDANCE_PER_LANE: float = (
    0.15  # Marginal threat multiplier per open downstream exit lane
)
EXIT_AFFORDANCE_MAX_MULT: float = (
    1.40  # Maximum threat multiplier cap for open distribution hubs
)
OBSTACLE_INTERCEPTION_RADIUS_M: float = (
    0.75  # Anthropometric defender obstruction corridor radius
)

# Expected Value score normalization scaling
EV_SCORE_SCALE_DELTA_V: float = (
    0.06  # 0.06 net xG delta maps to standard high-value line break threshold
)

# Kinematic ball mechanics & turf friction parameters
# Initial strike velocity for professional driven goalkeeper ground passes (~79.2 km/h)
PASS_GROUND_INITIAL_SPEED_MS: float = 22.0
# Coulomb rolling friction on FIFA standard hybrid turf: mu_roll ~ 0.25 * 9.81 m/s^2
PASS_TURF_ROLLING_DECELERATION_MSS: float = 2.45

# Presser sprint acceleration profile (Morin et al. sprint biomechanics)
# Peak burst acceleration from standing/jogging stance
PRESSER_MAX_BURST_ACCEL_MSS: float = 3.5
# Maximum sustained sprint speed for professional outfield presser (~30.6 km/h)
PRESSER_SPRINT_MAX_SPEED_MS: float = 8.5

# Aerial chipped pass & vertical obstacle clearance parameters
# Anthropometric maximum jumping reach of outfield defender (1.85m height + 0.60m vertical leap)
DEFENDER_MAX_JUMP_REACH_M: float = 2.45
# Standard parabolic apex height for clipped lofted ball over first pressing line
CHIPPED_PASS_APEX_M: float = 3.5
# First-touch motor control latency for trapping/cushioning an aerial descending ball
RECEIVER_AERIAL_TOUCH_LATENCY_S: float = 0.70
# Maximum realistic lead passing sprint displacement before receiver checks/slows run
MAX_LEAD_DISPLACEMENT_M: float = 8.0

# Empirical off-screen defender boundary prior (90th percentile distance of wide defending markers off-camera)
OFF_SCREEN_DEFENDER_PRIOR_M: float = 14.0
# Distance under which a closing forward directly compresses goalkeeper decision time
GK_PRESS_URGENCY_RADIUS_M: float = 8.0
# Tackle proximity floor where goalkeeper turnover probability nears 1.0 if holding
GK_PRESS_CRITICAL_RADIUS_M: float = 2.0
# Perception and execution score noise margin for co-optimal release equivalence classes
RELEASE_EQUIVALENCE_DELTA_SCORE: float = 8.0
# Break-even Net EV score threshold for positive-value distribution recommendations
MIN_RECOMMENDED_SCORE: float = 50.0
# Minimum completion probability for a distribution to qualify as a release window option
MIN_VIABLE_XP: float = 0.50
# Standard lateral step into open space away from defender's pressing angle
SPACE_PASS_LATERAL_OFFSET_M: float = 2.0

# Minimum clearance from touchlines and goal lines to avoid rollout execution variance
PITCH_TOUCHLINE_BUFFER_M: float = 2.0
# Longitudinal distance error ratio for driven professional goalkeeper distributions (7% of distance)
KICK_DISPERSION_RADIAL_RATIO: float = 0.07
# Arrival time difference threshold under which two teammates suffer target ownership ambiguity
TARGET_AMBIGUITY_THRESHOLD_S: float = 0.30
# Motor control hesitation latency penalty injected when a pass is placed into ambiguous shared space
TARGET_AMBIGUITY_PENALTY_S: float = 0.40
# Standard progressive space lead distance into receiver running stride (baseline fallback)
MANIFOLD_CHANNEL_LEAD_DIST_M: float = 3.5
# Extra sprint burst velocity differential into running stride (m/s)
MANIFOLD_BURST_SPEED_BOOST_MS: float = 1.8
# Minimum lead displacement floor for progressive channel passes (m)
MANIFOLD_MIN_BURST_LEAD_M: float = 1.5
# Maximum kinematic lead ceiling over typical pass flight horizon (m)
MANIFOLD_MAX_BURST_LEAD_M: float = 6.0
# Standard lateral displacement away from pressing opponent's approach angle
MANIFOLD_SHIELD_OFFSET_DIST_M: float = 2.2
# Maximum spatial separation between teammates to trigger target ownership ambiguity (m)
# Derived from maximum overlapping run / cross dispute spatial window
TARGET_AMBIGUITY_MAX_SEPARATION_M: float = 12.0
# Time window relative to ball arrival within which concurrent teammate arrival causes hesitation (s)
TARGET_AMBIGUITY_BALL_SYNC_WINDOW_S: float = 0.40
# Saturation cap on defensive arrival cushion where marginal tactical utility flattens (s)
# Derived from unpressed receiver threshold (>10m opponent separation)
CUSHION_SATURATION_CAP_S: float = 2.5
# Net expected value baseline for an intentional clearance / out-of-bounds dump (xG)
CLEARANCE_BASELINE_NET_EV: float = -0.01
# Arrival cushion reference floor for clearance actions (s)
CLEARANCE_BASELINE_CUSHION_S: float = 0.0
# Minimum physical window duration for human perception-action visual motor kicking execution (s)
# Derived from visual-motor latency literature (150-200ms perceptual delay + 150-200ms strike mechanics)
MIN_BIOMECHANICAL_WINDOW_S: float = 0.35
# Hysteresis threshold for opening a temporal passing window to filter sensor jitter (s)
WINDOW_OPEN_THRESHOLD_S: float = 0.20
# Hysteresis threshold for closing a temporal passing window to prevent chattering (s)
WINDOW_CLOSE_THRESHOLD_S: float = -0.10
# Safety time buffer required before goalkeeper tackle point to allow unharassed release (s)
PASSER_HARASSMENT_BUFFER_S: float = 0.40
# Moving average frame window size for 25 Hz tracking noise suppression (frames = 280ms)
SMOOTHING_WINDOW_FRAMES: int = 7
# Outstretched-leg control radius: boot plus leg reach beside the body (m)
DUEL_CONTROL_RADIUS_M: float = 1.2
# Control plus first action cycle: trap, look up, play the next ball (s)
RETENTION_WINDOW_S: float = 3.0
# Longest ball-detection gap bridged during retention (tracking gaps run <=8 frames)
RETENTION_MAX_GAP_FRAMES: int = 3
# A pass counts as played toward an outlet only if the ball comes within 2.5
# control radii of its lead target (m)
RELEASE_APPROACH_GATE_M: float = 3.0


__all__ = [
    "GK_REACTION_TIME_FLOOR_S",
    "GK_REACTION_TIME_DEFAULT_S",
    "GK_LATERAL_BURST_SPEED_MAX_MS",
    "GK_LATERAL_BURST_SPEED_DEFAULT_MS",
    "GK_WINGSPAN_REACH_DEFAULT_M",
    "GK_DIVE_RECOVERY_TIME_S",
    "GK_SET_POSITION_LATENCY_S",
    "GK_MAX_BURST_ACCELERATION_MSS",
    "RECEIVER_FIRST_TOUCH_LATENCY_S",
    "MIN_POST_RECEIPT_CUSHION_S",
    "CENTRAL_TURNOVER_PEAK_XG",
    "CENTRAL_ZONE_SIGMA_X_M",
    "CENTRAL_ZONE_SIGMA_Y_M",
    "TURNOVER_HAZARD_FLOOR_XG",
    "RETENTION_LOGISTIC_STEEPNESS_K",
    "RETENTION_CRITICAL_CUSHION_S",
    "EXIT_AFFORDANCE_MIN_DIST_M",
    "EXIT_AFFORDANCE_MAX_DIST_M",
    "EXIT_AFFORDANCE_BACKWARD_TOLERANCE_M",
    "EXIT_AFFORDANCE_PINNED_MULT",
    "EXIT_AFFORDANCE_BASE_MULT",
    "EXIT_AFFORDANCE_PER_LANE",
    "EXIT_AFFORDANCE_MAX_MULT",
    "OBSTACLE_INTERCEPTION_RADIUS_M",
    "EV_SCORE_SCALE_DELTA_V",
    "PASS_GROUND_INITIAL_SPEED_MS",
    "PASS_TURF_ROLLING_DECELERATION_MSS",
    "PRESSER_MAX_BURST_ACCEL_MSS",
    "PRESSER_SPRINT_MAX_SPEED_MS",
    "DEFENDER_MAX_JUMP_REACH_M",
    "CHIPPED_PASS_APEX_M",
    "RECEIVER_AERIAL_TOUCH_LATENCY_S",
    "MAX_LEAD_DISPLACEMENT_M",
    "OFF_SCREEN_DEFENDER_PRIOR_M",
    "GK_PRESS_URGENCY_RADIUS_M",
    "GK_PRESS_CRITICAL_RADIUS_M",
    "RELEASE_EQUIVALENCE_DELTA_SCORE",
    "MIN_RECOMMENDED_SCORE",
    "MIN_VIABLE_XP",
    "SPACE_PASS_LATERAL_OFFSET_M",
    "PITCH_TOUCHLINE_BUFFER_M",
    "KICK_DISPERSION_RADIAL_RATIO",
    "TARGET_AMBIGUITY_THRESHOLD_S",
    "TARGET_AMBIGUITY_PENALTY_S",
    "MANIFOLD_CHANNEL_LEAD_DIST_M",
    "MANIFOLD_BURST_SPEED_BOOST_MS",
    "MANIFOLD_MIN_BURST_LEAD_M",
    "MANIFOLD_MAX_BURST_LEAD_M",
    "MANIFOLD_SHIELD_OFFSET_DIST_M",
    "TARGET_AMBIGUITY_MAX_SEPARATION_M",
    "TARGET_AMBIGUITY_BALL_SYNC_WINDOW_S",
    "CUSHION_SATURATION_CAP_S",
    "CLEARANCE_BASELINE_NET_EV",
    "CLEARANCE_BASELINE_CUSHION_S",
    "MIN_BIOMECHANICAL_WINDOW_S",
    "WINDOW_OPEN_THRESHOLD_S",
    "WINDOW_CLOSE_THRESHOLD_S",
    "PASSER_HARASSMENT_BUFFER_S",
    "SMOOTHING_WINDOW_FRAMES",
    "DUEL_CONTROL_RADIUS_M",
    "RETENTION_WINDOW_S",
    "RETENTION_MAX_GAP_FRAMES",
    "RELEASE_APPROACH_GATE_M",
]
