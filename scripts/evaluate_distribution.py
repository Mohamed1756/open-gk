#!/usr/bin/env python3.11
# Justification: Bundles the valuation evaluation engine and self-contained interactive
# multi-scenario HTML visualization dashboard for zero-dependency standalone tactical review.
"""
Goalkeeper Distribution Valuation Engine & Interactive Multi-Episode Dashboard.

Evaluates goalkeeper distribution decisions across any match episode driven by MatchConfig:
computes xP, xT, turnover risk, visual field occlusion, biomechanical execution penalties,
and arrival margins, rendering an interactive tactical dashboard with multi-episode switching.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GOAL_Y_CENTER
from src.core.geometry import PitchPoint
from src.core.types import DistributionType
from src.cv.pitch_homography import DynamicPitchHomography
from src.cv.player_detector import (
    is_keeper_label,
    is_official_label,
)
from src.models.perception.visual_field import (
    is_in_visual_cone,
    compute_los_occlusion,
    evaluate_chipped_clearance,
    evaluate_pressing_approach_cut,
    update_spatial_memory_buffer,
    visible_person_ids,
    vision_logit_penalty,
    summarize_scan_windows,
    merge_swept_cones,
    SCAN_KNOWN_THRESHOLD,
    SCAN_CONFIRMED_THRESHOLD,
    SCAN_KNOWN_PENALTY,
    OUT_OF_VISION_PENALTY,
    TACTICAL_PRIOR_FLOOR,
)
from src.physics.gk_constraints import (
    MIN_POST_RECEIPT_CUSHION_S,
    RECEIVER_FIRST_TOUCH_LATENCY_S,
    RECEIVER_AERIAL_TOUCH_LATENCY_S,
    RETENTION_LOGISTIC_STEEPNESS_K,
    RETENTION_CRITICAL_CUSHION_S,
    EXIT_AFFORDANCE_MIN_DIST_M,
    EXIT_AFFORDANCE_MAX_DIST_M,
    EXIT_AFFORDANCE_BACKWARD_TOLERANCE_M,
    OBSTACLE_INTERCEPTION_RADIUS_M,
    EV_SCORE_SCALE_DELTA_V,
    PASS_GROUND_INITIAL_SPEED_MS,
    DEFENDER_MAX_JUMP_REACH_M,
    OFF_SCREEN_DEFENDER_PRIOR_M,
    GK_PRESS_URGENCY_RADIUS_M,
    GK_PRESS_CRITICAL_RADIUS_M,
    RELEASE_EQUIVALENCE_DELTA_SCORE,
    MIN_RECOMMENDED_SCORE,
    MIN_VIABLE_XP,
    TARGET_AMBIGUITY_PENALTY_S,
    KICK_DISPERSION_RADIAL_RATIO,
)
from src.physics.biomechanics import (
    compute_hip_pivot_latency,
    compute_biomechanical_execution_penalty,
    evaluate_effective_press_closure,
    is_pass_inertia_locked,
    compute_projected_closing_velocity,
)
from src.models.distribution.evaluator import DistributionEvaluator
from src.models.distribution.manifold import (
    ManifoldTarget,
    clamp_to_pitch_buffer,
    generate_receiver_manifold,
)
from src.models.distribution.first_touch import (
    find_first_touch,
    decision_possession_frames,
)
from src.models.distribution.frontier import compute_decision_frontier
from src.models.pressing.types import PressingActor, PressingSnapshot
from src.models.pressing.units import cluster_lines, press_state
from src.models.pressing.arrival import arrival_margin, flight_time_s, lead_target
from src.models.pressing.duels import track_duel
from src.models.pressing.matchup import assign_pressers
from src.models.pressing.readiness import receiver_readiness
from src.models.distribution.outcomes import release_frame_idx, resolve_duel_outcome
from src.models.distribution.temporal_windows import calculate_offside_line_x
from src.physics.gk_constraints import RELEASE_APPROACH_GATE_M
from src.config_match import (
    MatchConfig,
    episode21_config,
    episode_2500_2525_config,
    episode_7818_7828_config,
    get_match_config,
)


def _is_keeper_entity(team_label: str, color: str, match: MatchConfig) -> bool:
    if is_keeper_label(team_label):
        return True
    keeper_kits = [k for k in match.kits if k.is_keeper]
    if any(k.label.lower() in team_label.lower() for k in keeper_kits):
        return True
    keeper_colors = [k.color_hex.lower() for k in keeper_kits]
    return color.lower() in keeper_colors


def _team_sets(match: MatchConfig) -> tuple[list[str], list[str], str]:
    possession = [
        k.label
        for k in match.kits
        if not k.is_official
        and not k.is_keeper
        and k.team_id == match.possession_team_id
    ]
    if not possession:
        possession = [
            k.label for k in match.kits if not k.is_official and not k.is_keeper
        ]
    keeper_label = match.keeper_kit().label if match.keeper_kit() else ""
    opponent = [
        k.label
        for k in match.kits
        if not k.is_official
        and not k.is_keeper
        and k.team_id != match.possession_team_id
    ]
    return possession, opponent, keeper_label


def _split_frame(
    frame_entities: list[dict[str, Any]],
    possession_labels: list[str],
    opponent_labels: list[str],
    match: MatchConfig,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Only persons can be outfield teammates, opponents, or goalkeepers (never the ball)
    persons = [
        e
        for e in frame_entities
        if e.get("class_name") == "person" and e.get("pitch_valid", True)
    ]
    gk = next(
        (
            e
            for e in persons
            if _is_keeper_entity(e.get("team_label", ""), e.get("color", ""), match)
        ),
        None,
    )
    team = [
        e
        for e in persons
        if e.get("team_label") in possession_labels
        or e.get("team_label") == "Unknown-Outfield"
        or (
            match.possession_team_id
            and match.possession_team_id in e.get("team_label", "").lower()
        )
    ]
    opp = [
        e
        for e in persons
        if e.get("team_label") in opponent_labels
        or (
            e not in team and e != gk and not is_official_label(e.get("team_label", ""))
        )
    ]
    return gk, team, opp


def _actors_from_entities(
    frame_entities: list[dict[str, Any]],
    team_id: str,
    facing_by_track: Optional[Dict[int, tuple[float, str]]] = None,
) -> List[PressingActor]:
    actors = []
    for e in frame_entities:
        if e.get("class_name") != "person" or not e.get("pitch_valid", True):
            continue
        f = (facing_by_track or {}).get(e["track_id"])
        vel = e.get("pitch_vel_ms", [0.0, 0.0])
        actors.append(
            PressingActor(
                track_id=int(e["track_id"]),
                team_id=team_id,
                pos_m=PitchPoint(x=e["pitch_xy"][0], y=e["pitch_xy"][1]),
                vel_ms=(float(vel[0]), float(vel[1])),
                facing_rad=float(f[0]) if f else None,
                facing_source=str(f[1]) if f else "unknown",
                pitch_valid=True,
            )
        )
    return actors


def _ball_xy(frame_entities: list[dict[str, Any]]) -> Optional[list[float]]:
    b = next(
        (e["pitch_xy"] for e in frame_entities if e.get("class_name") == "sports ball"),
        None,
    )
    return [float(b[0]), float(b[1])] if b else None


def _passer_xy(
    frame_entities: list[dict[str, Any]], passer_track_id: Optional[int]
) -> Optional[list[float]]:
    if passer_track_id is None:
        return None
    p = next(
        (e for e in frame_entities if e.get("track_id") == passer_track_id),
        None,
    )
    if p is None:
        return None
    return [float(p["pitch_xy"][0]), float(p["pitch_xy"][1])]


def _screen_velocity_px_per_frame(
    tracking_records: list[dict[str, Any]],
    track_id: int,
    cur_uv: tuple[float, float],
    cur_idx: int,
    lookback: int = 5,
) -> Optional[tuple[float, float]]:
    for prev_idx in range(cur_idx - 1, max(-1, cur_idx - 1 - lookback) - 1, -1):
        if prev_idx < 0 or prev_idx >= len(tracking_records):
            continue
        prev_ent = next(
            (
                e
                for e in tracking_records[prev_idx]["entities"]
                if e.get("track_id") == track_id and e.get("screen_uv") is not None
            ),
            None,
        )
        if prev_ent is None:
            continue
        dt = cur_idx - prev_idx
        if dt <= 0:
            continue
        puv = prev_ent["screen_uv"]
        return ((cur_uv[0] - float(puv[0])) / dt, (cur_uv[1] - float(puv[1])) / dt)
    return None


def _pose_torso_facing(
    estimator: Any,
    frame: Any,
    ent: dict[str, Any],
    tracking_records: list[dict[str, Any]],
    fi: int,
) -> Optional[tuple[float, str]]:
    if "bbox" not in ent:
        return None
    e_uv = tuple(ent.get("screen_uv", [0.0, 0.0]))
    e_vel = _screen_velocity_px_per_frame(
        tracking_records,
        int(ent["track_id"]),
        (float(e_uv[0]), float(e_uv[1])),
        fi,
    )
    res = estimator.estimate_pose_in_crop(
        frame, tuple(ent["bbox"]), frame_idx=fi, velocity_screen_uv=e_vel
    )
    if not res.is_valid:
        return None
    return (float(res.torso_facing_rad), "pose")


BAD_STATUSES = {
    "INERTIA_LOCKED",
    "SHADOWED_LANE",
    "PRESS_TRAP",
    "APPROACH_CUT",
    "CONTROL_TRAP",
}


def _eval_single_target(
    cand: Dict[str, Any],
    mt: ManifoldTarget,
    passer_pos: PitchPoint,
    opponents: List[PitchPoint],
    opp_actors: List[PressingActor],
    rec_actor: Optional[PressingActor],
    receiver_candidates: List[Dict[str, Any]],
    evaluator: DistributionEvaluator,
    passer_facing_angle_rad: Optional[float],
    passer_gaze_angle_rad: Optional[float],
    memory_weights: Optional[Dict[int, float]],
    touch_vel_xy_ms: Optional[tuple[float, float]],
    nearest_presser_vel_ms: Optional[tuple[float, float]],
    attack_dir_x: float,
    gk_turnover_hazard: float,
    gk_press_dist: float,
    closing_speed_ms: float,
    pairing: Optional[Any],
    nearest_presser_actor: Optional[PressingActor],
) -> Dict[str, Any]:
    target_pos = mt.target_pos
    dist_m = float(math.hypot(target_pos.x - passer_pos.x, target_pos.y - passer_pos.y))
    pass_flight_s = mt.flight_time_s

    min_press_dist = 99.0
    if opponents:
        min_press_dist = min(
            float(((target_pos.x - op.x) ** 2 + (target_pos.y - op.y) ** 2) ** 0.5)
            for op in opponents
        )

    if attack_dir_x >= 0:
        opponents_pressed_count = sum(
            1 for op in opponents if passer_pos.x <= op.x <= target_pos.x
        )
    else:
        opponents_pressed_count = sum(
            1 for op in opponents if target_pos.x <= op.x <= passer_pos.x
        )

    dist_type = (
        DistributionType.SHORT_PASS if dist_m <= 25.0 else DistributionType.LONG_PASS
    )
    raw_xp = evaluator.estimate_completion_probability(
        dist_m, min_press_dist, dist_type
    )
    if mt.is_boundary_discounted:
        raw_xp = raw_xp * mt.boundary_discount_mult

    pass_angle = math.atan2(target_pos.y - passer_pos.y, target_pos.x - passer_pos.x)
    in_cone = True
    diff_deg = 0.0
    is_blocked = False
    is_chipped = False
    chipped_clearance_h = 0.0
    prep_latency = 0.15
    exec_mult = 1.0

    effective_gaze_rad = (
        passer_gaze_angle_rad
        if passer_gaze_angle_rad is not None
        else passer_facing_angle_rad
    )
    if effective_gaze_rad is not None:
        in_cone, abs_diff = is_in_visual_cone(
            passer_pos, effective_gaze_rad, target_pos, fov_deg=140.0
        )
        diff_deg = round(math.degrees(abs_diff), 0)
        dispersion_corridor_inflation = min(1.2, mt.dispersion_sigma_m * 0.35)
        is_blocked, blocking_op = compute_los_occlusion(
            passer_pos,
            target_pos,
            obstacles=opponents,
            obstacle_radius_m=OBSTACLE_INTERCEPTION_RADIUS_M
            + dispersion_corridor_inflation,
        )
        if is_blocked and blocking_op is not None:
            is_chipped_clear, clearance_h = evaluate_chipped_clearance(
                passer_pos=passer_pos,
                target_pos=target_pos,
                obstacle_pos=blocking_op,
            )
            if is_chipped_clear:
                is_blocked = False
                is_chipped = True
                chipped_clearance_h = clearance_h

    if passer_facing_angle_rad is not None:
        prep_latency = compute_hip_pivot_latency(passer_facing_angle_rad, pass_angle)
        exec_mult = compute_biomechanical_execution_penalty(
            passer_facing_angle_rad, pass_angle
        )

    is_approach_cut = False
    if nearest_presser_vel_ms is not None and opponents:
        nearest_op = min(
            opponents,
            key=lambda op: (passer_pos.x - op.x) ** 2 + (passer_pos.y - op.y) ** 2,
        )
        is_approach_cut, min_intercept_dist, time_delta = (
            evaluate_pressing_approach_cut(
                passer_pos=passer_pos,
                target_pos=target_pos,
                presser_pos=nearest_op,
                presser_vel_xy_ms=nearest_presser_vel_ms,
                pass_speed_ms=PASS_GROUND_INITIAL_SPEED_MS,
                prep_latency_s=prep_latency,
                time_horizon_s=None,
                corridor_width_m=1.5,
                interception_time_tolerance_s=0.35,
            )
        )

    is_inertia_locked = False
    if touch_vel_xy_ms is not None:
        is_inertia_locked, total_delay, eff_gk_cushion = is_pass_inertia_locked(
            touch_vel_xy_ms=touch_vel_xy_ms,
            target_pass_angle_rad=pass_angle,
            body_facing_angle_rad=passer_facing_angle_rad or 0.0,
            nearest_presser_dist_m=gk_press_dist,
            presser_closing_speed_ms=closing_speed_ms,
        )
        prep_latency = total_delay

    arrival = (
        arrival_margin(
            rec_actor,
            opp_actors,
            passer_pos,
            prep_latency_s=prep_latency,
            target_pos=target_pos,
        )
        if rec_actor is not None
        else None
    )

    effective_margin = arrival.margin_s if arrival is not None else 99.0
    if mt.is_ambiguity_penalized:
        effective_margin -= TARGET_AMBIGUITY_PENALTY_S
    if is_chipped and arrival is not None:
        effective_margin -= pass_flight_s * 0.25

    is_margin_trapped = arrival is not None and effective_margin < 0.0
    is_pincer = bool(pairing.is_pincer) if pairing is not None else False

    presser_closing = False
    rec_presser_closing_speed_ms = 0.0
    if nearest_presser_actor is not None:
        dx = target_pos.x - nearest_presser_actor.pos_m.x
        dy = target_pos.y - nearest_presser_actor.pos_m.y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            vx, vy = nearest_presser_actor.vel_ms
            proj_speed = (vx * dx + vy * dy) / dist
            rec_presser_closing_speed_ms = max(0.0, proj_speed)
            presser_closing = proj_speed > 1.0

    readiness = (
        receiver_readiness(
            rec_actor,
            passer_pos,
            nearest_presser=nearest_presser_actor,
            presser_closing=presser_closing,
            ball_pos=passer_pos,
        )
        if rec_actor is not None
        else None
    )

    touch_latency = (
        RECEIVER_AERIAL_TOUCH_LATENCY_S
        if is_chipped
        else RECEIVER_FIRST_TOUCH_LATENCY_S
    )
    if arrival is not None and readiness is not None:
        t_post_cushion = round(
            effective_margin - (touch_latency + readiness.turn_latency_s), 3
        )
    elif arrival is not None:
        t_post_cushion = round(effective_margin - touch_latency, 3)
    else:
        t_post_cushion = 99.0

    effective_press_dist = evaluate_effective_press_closure(
        min_press_dist,
        presser_closing_speed_ms=rec_presser_closing_speed_ms,
        prep_latency_s=prep_latency,
    )

    is_off_screen_prior = False
    if (
        abs(target_pos.y - GOAL_Y_CENTER) >= 18.0
        and min_press_dist > OFF_SCREEN_DEFENDER_PRIOR_M
    ):
        effective_press_dist = min(effective_press_dist, OFF_SCREEN_DEFENDER_PRIOR_M)
        t_post_cushion = min(t_post_cushion, 2.0)
        is_off_screen_prior = True

    bounded_raw_xp = float(np.clip(raw_xp, 0.05, 0.95))
    base_logit = math.log(bounded_raw_xp / (1.0 - bounded_raw_xp))

    memory_weight = 0.0
    if memory_weights is not None:
        memory_weight = max(
            0.0, min(1.0, float(memory_weights.get(cand["track_id"], 0.0)))
        )
    scan_known = (not in_cone) and memory_weight >= SCAN_KNOWN_THRESHOLD

    logit_penalty_biomech = -math.log(max(0.25, exec_mult))
    logit_penalty_vis = vision_logit_penalty(memory_weight) if not in_cone else 0.0
    logit_penalty_los = 1.10 if is_blocked else 0.0
    logit_penalty_approach = 1.35 if is_approach_cut else 0.0
    logit_penalty_inertia = 2.20 if is_inertia_locked else 0.0
    logit_penalty_pincer = 0.70 if is_pincer else 0.0

    total_penalty = (
        logit_penalty_biomech
        + logit_penalty_vis
        + logit_penalty_los
        + logit_penalty_approach
        + logit_penalty_inertia
        + logit_penalty_pincer
    )
    calibrated_logit = base_logit - total_penalty
    effective_xp = round(float(1.0 / (1.0 + math.exp(-calibrated_logit))), 3)
    if readiness is not None:
        effective_xp = round(min(0.99, effective_xp * readiness.readiness_mult), 3)

    exit_lanes_count = 0
    for other_cand in receiver_candidates:
        if other_cand.get("track_id") == cand.get("track_id"):
            continue
        other_xy = other_cand["pitch_xy"]
        tm_target = PitchPoint(x=float(other_xy[0]), y=float(other_xy[1]))
        dx = (
            (tm_target.x - target_pos.x)
            if attack_dir_x >= 0
            else (target_pos.x - tm_target.x)
        )
        lane_dist = math.hypot(tm_target.x - target_pos.x, tm_target.y - target_pos.y)
        if (
            dx >= -EXIT_AFFORDANCE_BACKWARD_TOLERANCE_M
            and EXIT_AFFORDANCE_MIN_DIST_M <= lane_dist <= EXIT_AFFORDANCE_MAX_DIST_M
        ):
            is_exit_blocked, _ = compute_los_occlusion(
                observer_pos=target_pos,
                target_pos=tm_target,
                obstacles=opponents,
                obstacle_radius_m=OBSTACLE_INTERCEPTION_RADIUS_M,
                max_check_dist_m=lane_dist,
            )
            if not is_exit_blocked:
                exit_lanes_count += 1

    prog_threat = evaluator.compute_progression_threat(
        origin=passer_pos,
        target=target_pos,
        opponents_pressed_count=opponents_pressed_count,
        attack_dir_x=attack_dir_x,
        exit_lanes_count=exit_lanes_count,
    )
    turnover_cost = evaluator.compute_turnover_risk_cost(
        target=target_pos,
        attack_dir_x=attack_dir_x,
    )

    perception_penalty = 0.0
    if is_inertia_locked:
        path_status = "INERTIA_LOCKED"
        grade_color = "#ef4444"
        vis_label = "MOMENTUM LOCKED"
        bottleneck = "First-touch momentum opposes strike angle under press"
    elif is_approach_cut:
        path_status = "APPROACH_CUT"
        grade_color = "#f97316"
        vis_label = "PRESSER CUTTING LANE"
        bottleneck = "Defender sprint trajectory cuts arrival window"
    elif is_margin_trapped:
        assert arrival is not None
        path_status = "PRESS_TRAP"
        grade_color = "#ef4444"
        vis_label = f"TRAPPED (press {-effective_margin:.1f}s early)"
        bottleneck = (
            f"Presser #{arrival.best_presser_id} reaches the lead target "
            f"{-effective_margin:.2f}s before the ball"
        )
    elif is_blocked:
        path_status = "SHADOWED_LANE"
        grade_color = "#ef4444"
        vis_label = "BLOCKED BY STRIKER"
        bottleneck = "Striker cover shadow blocks direct passing corridor"
    elif arrival is not None and t_post_cushion < MIN_POST_RECEIPT_CUSHION_S:
        path_status = "CONTROL_TRAP"
        grade_color = "#ef4444"
        vis_label = f"CONTROL TRAP ({t_post_cushion:.2f}s cushion)"
        bottleneck = (
            f"Receiver tackled during first-touch control "
            f"({t_post_cushion:.2f}s cushion < {MIN_POST_RECEIPT_CUSHION_S:.2f}s)"
        )
    elif is_chipped:
        path_status = "CHIPPED_OUTLET"
        grade_color = "#38bdf8"
        vis_label = f"CHIPPED ({chipped_clearance_h:.1f}m)"
        bottleneck = (
            f"Clipped aerial pass clears pressing striker "
            f"({chipped_clearance_h:.1f}m clearance > {DEFENDER_MAX_JUMP_REACH_M:.2f}m)"
        )
    elif effective_press_dist < 4.0:
        path_status = "PRESS_TRAP"
        grade_color = "#ef4444"
        vis_label = f"IN VISION ({diff_deg}°)" if in_cone else "PRESS TRAP"
        bottleneck = "Receiver will be immediately tackled (< 4.0m cushion)"
    elif effective_press_dist < 8.0:
        path_status = "CONTESTED_POCKET"
        grade_color = "#facc15"
        vis_label = f"IN VISION ({diff_deg}°)" if in_cone else "CONTESTED"
        bottleneck = f"Compressed pocket ({effective_press_dist:.1f}m cushion) requires delayed release to open"
    elif (not in_cone) and memory_weight >= SCAN_CONFIRMED_THRESHOLD:
        torso_pass_diff_deg = abs(
            math.degrees(
                math.atan2(
                    math.sin(pass_angle - (passer_facing_angle_rad or 0.0)),
                    math.cos(pass_angle - (passer_facing_angle_rad or 0.0)),
                )
            )
        )
        if torso_pass_diff_deg > 45.0:
            path_status = "DISGUISED_OUTLET"
            grade_color = "#38bdf8"
            vis_label = f"DISGUISED ({memory_weight:.0%})"
            bottleneck = "Disguised line-breaking outlet unlocked via pre-scan"
        else:
            path_status = "SCANNED_OPEN"
            grade_color = "#22c55e"
            vis_label = f"SCANNED ({memory_weight:.0%})"
            bottleneck = "Open passing lane unlocked via pre-scan awareness"
    elif scan_known:
        path_status = "KNOWN_FROM_SCAN"
        grade_color = "#2dd4bf"
        vis_label = f"KNOWN FROM SCAN ({memory_weight:.0%})"
        bottleneck = "Outside current gaze but scanned within memory horizon"
        perception_penalty = SCAN_KNOWN_PENALTY
    elif not in_cone:
        path_status = "OUT_OF_VISION"
        grade_color = "#94a3b8"
        vis_label = f"OUT OF CONE ({diff_deg}°)"
        bottleneck = "Outside 140° visual field (scanning blindspot)"
        perception_penalty = OUT_OF_VISION_PENALTY
    else:
        path_status = "FEASIBLE_OPEN"
        grade_color = "#22c55e"
        vis_label = f"IN VISION ({diff_deg}°)"
        bottleneck = "Open passing lane with viable receiver separation"
        perception_penalty = 0.0

    if is_pincer and path_status not in ("APPROACH_CUT", "INERTIA_LOCKED"):
        bottleneck = (
            f"{bottleneck} (2nd man #{pairing.second_presser_id} synchronized)"
            if pairing is not None and pairing.second_presser_id is not None
            else f"{bottleneck} (pincer)"
        )

    if is_off_screen_prior:
        bottleneck = f"{bottleneck} (off-screen marker prior: {OFF_SCREEN_DEFENDER_PRIOR_M:.0f}m cap)"

    if mt.action_type == "PROGRESSIVE_CHANNEL":
        bottleneck = f"{bottleneck} [Channel lead: +{mt.offset_dist_m:.1f}m stride]"
    elif mt.action_type == "SHIELDED_POCKET":
        bottleneck = f"{bottleneck} [Shielded pocket: +{mt.offset_dist_m:.1f}m away from presser]"

    if mt.is_ambiguity_penalized:
        bottleneck = f"{bottleneck} [Target ambiguity: concurrent #{mt.ambiguity_partner_id} arrival (-{TARGET_AMBIGUITY_PENALTY_S:.2f}s)]"

    if mt.is_boundary_discounted:
        bottleneck = f"{bottleneck} [Touchline buffer discount: {mt.boundary_discount_mult:.2f}x]"

    if arrival is not None:
        retention_mult = float(
            1.0
            / (
                1.0
                + math.exp(
                    -RETENTION_LOGISTIC_STEEPNESS_K
                    * (t_post_cushion - RETENTION_CRITICAL_CUSHION_S)
                )
            )
        )
    else:
        retention_mult = 1.0
    retention_xp = effective_xp * retention_mult

    hazard_reduction = max(0.0, gk_turnover_hazard - turnover_cost)
    d_eff_gk = max(0.0, gk_press_dist - closing_speed_ms * 0.35)
    if d_eff_gk >= GK_PRESS_URGENCY_RADIUS_M:
        gk_urgency = 0.0
    elif d_eff_gk <= GK_PRESS_CRITICAL_RADIUS_M:
        gk_urgency = 1.0
    else:
        gk_urgency = (GK_PRESS_URGENCY_RADIUS_M - d_eff_gk) / (
            GK_PRESS_URGENCY_RADIUS_M - GK_PRESS_CRITICAL_RADIUS_M
        )
    relief_value = hazard_reduction * max(0.20, gk_urgency)

    net_ev = evaluator.compute_net_distribution_ev(
        retention_xp=retention_xp,
        prog_threat=prog_threat,
        target_turnover_cost=turnover_cost,
        relief_value=relief_value,
    )

    cushion_ratio = float(
        np.clip((effective_press_dist - 4.0) / (12.0 - 4.0), 0.0, 1.0)
    )
    score_safety = round(retention_xp * 35.0, 1)
    score_cushion = round(35.0 * cushion_ratio * retention_xp, 1)
    score_evasion = round(
        min(30.0, opponents_pressed_count * 10.0) * retention_xp * cushion_ratio,
        1,
    )

    raw_path_score = (
        50.0 + 50.0 * math.tanh(net_ev / EV_SCORE_SCALE_DELTA_V) - perception_penalty
    )

    if path_status in BAD_STATUSES:
        final_path_score = round(max(0.0, min(15.0, raw_path_score * 0.15)), 1)
    else:
        final_path_score = round(max(0.0, min(100.0, raw_path_score)), 1)

    return {
        "track_id": cand["track_id"],
        "role": cand["team_label"],
        "action_type": mt.action_type,
        "offset_dist_m": mt.offset_dist_m,
        "dispersion_sigma_m": mt.dispersion_sigma_m,
        "is_boundary_discounted": mt.is_boundary_discounted,
        "boundary_discount_mult": mt.boundary_discount_mult,
        "is_ambiguity_penalized": mt.is_ambiguity_penalized,
        "ambiguity_partner_id": mt.ambiguity_partner_id,
        "target_pos": [round(target_pos.x, 1), round(target_pos.y, 1)],
        "distance_m": round(dist_m, 1),
        "nearest_presser_m": round(min_press_dist, 1),
        "effective_press_m": round(effective_press_dist, 1),
        "bypassed_pressers": opponents_pressed_count,
        "xp_completion_prob": effective_xp,
        "raw_xp": round(raw_xp, 3),
        "path_score": final_path_score,
        "score_safety": score_safety,
        "score_evasion": score_evasion,
        "score_cushion": score_cushion,
        "decision_grade": path_status,
        "path_status": path_status,
        "grade_color": grade_color,
        "in_visual_cone": in_cone,
        "is_los_blocked": is_blocked,
        "is_chipped": is_chipped,
        "chipped_clearance_h": round(chipped_clearance_h, 2),
        "is_inertia_locked": is_inertia_locked,
        "is_approach_cut": is_approach_cut,
        "memory_weight": round(memory_weight, 3),
        "scan_known": scan_known,
        "margin_s": arrival.margin_s if arrival is not None else 99.0,
        "best_presser_id": arrival.best_presser_id if arrival is not None else None,
        "is_pincer": is_pincer,
        "second_presser_id": pairing.second_presser_id if pairing is not None else None,
        "presser_track_id": pairing.presser_track_id if pairing is not None else None,
        "free_man": bool(pairing.is_free) if pairing is not None else False,
        "readiness_mult": readiness.readiness_mult if readiness is not None else 1.0,
        "turn_latency_s": readiness.turn_latency_s if readiness is not None else 0.0,
        "facing_source": readiness.facing_source
        if readiness is not None
        else "unknown",
        "visual_label": vis_label,
        "diff_deg": diff_deg,
        "bottleneck_diagnostic": bottleneck,
        "hip_latency_s": prep_latency,
        "exec_multiplier": exec_mult,
        "t_post_cushion_s": round(t_post_cushion, 2),
        "prog_threat": round(prog_threat, 4),
        "turnover_hazard": round(turnover_cost, 4),
        "relief_value": round(relief_value, 4),
        "is_off_screen_prior": is_off_screen_prior,
        "space_target_pos": [
            round(target_pos.x, 1),
            round(target_pos.y, 1),
        ]
        if mt.action_type in ("SHIELDED_POCKET", "PROGRESSIVE_CHANNEL")
        else None,
        "space_cushion_s": round(t_post_cushion, 2)
        if mt.action_type in ("SHIELDED_POCKET", "PROGRESSIVE_CHANNEL")
        else None,
        "net_ev": round(net_ev, 4),
        "exit_lanes_count": exit_lanes_count,
    }


def evaluate_distribution_decision(
    passer_pos: PitchPoint,
    receiver_candidates: List[Dict[str, Any]],
    opponents: List[PitchPoint],
    evaluator: DistributionEvaluator,
    passer_facing_angle_rad: Optional[float] = None,
    passer_gaze_angle_rad: Optional[float] = None,
    touch_vel_xy_ms: Optional[tuple[float, float]] = None,
    nearest_presser_vel_ms: Optional[tuple[float, float]] = None,
    attack_dir_x: float = -1.0,
    memory_weights: Optional[Dict[int, float]] = None,
    opponent_actors: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluates candidate pass options taking into account goalkeeper field of view
    (140-deg visual cone from head gaze), line-of-sight occlusion, first-touch momentum inertia locks,
    biomechanical hip pivot latency from torso heading, and dynamic pressing approach trajectory cuts.
    """

    def _actor_from_cand(cand: Dict[str, Any]) -> PressingActor:
        vel = cand.get("pitch_vel_ms", [0.0, 0.0])
        return PressingActor(
            track_id=int(cand["track_id"]),
            team_id="own",
            pos_m=PitchPoint(x=cand["pitch_xy"][0], y=cand["pitch_xy"][1]),
            vel_ms=(float(vel[0]), float(vel[1])),
            facing_rad=cand.get("facing_rad"),
            facing_source=cand.get("facing_source", "unknown"),
            pitch_valid=bool(cand.get("pitch_valid", True)),
        )

    if opponent_actors is not None:
        opp_actors: List[PressingActor] = list(opponent_actors)
    else:
        opp_actors = [
            PressingActor(track_id=-1 - i, team_id="opp", pos_m=op)
            for i, op in enumerate(opponents)
        ]
    recv_actors = [_actor_from_cand(c) for c in receiver_candidates]
    recv_by_id = {a.track_id: a for a in recv_actors}
    passer_actor = PressingActor(
        track_id=-999,
        team_id="own",
        pos_m=passer_pos,
        facing_rad=passer_facing_angle_rad,
        facing_source="pose" if passer_facing_angle_rad is not None else "unknown",
    )
    matchup = assign_pressers(
        PressingSnapshot(
            passer=passer_actor,
            receivers=recv_actors,
            opponents=opp_actors,
            ball_pos_m=passer_pos,
            attack_dir_x=attack_dir_x,
            memory_weights=memory_weights or {},
        )
    )
    pair_by_recv = {o.receiver_track_id: o for o in matchup.outlets}

    gk_turnover_hazard = evaluator.compute_turnover_risk_cost(
        target=passer_pos,
        attack_dir_x=attack_dir_x,
    )

    gk_press_dist = 99.0
    closing_speed_ms = 4.0
    if opponents:
        nearest_op = min(
            opponents,
            key=lambda op: (passer_pos.x - op.x) ** 2 + (passer_pos.y - op.y) ** 2,
        )
        gk_press_dist = math.hypot(
            passer_pos.x - nearest_op.x, passer_pos.y - nearest_op.y
        )
        if nearest_presser_vel_ms is not None:
            closing_speed_ms = compute_projected_closing_velocity(
                presser_pos_m=(nearest_op.x, nearest_op.y),
                presser_vel_xy_ms=nearest_presser_vel_ms,
                target_pos_m=(passer_pos.x, passer_pos.y),
                min_floor_speed_ms=2.0,
            )

    option_evals: List[Dict[str, Any]] = []

    for cand in receiver_candidates:
        rec_actor = recv_by_id.get(int(cand["track_id"]))
        pairing = pair_by_recv.get(int(cand["track_id"]))
        nearest_presser_actor = None
        if pairing is not None and pairing.presser_track_id is not None:
            nearest_presser_actor = next(
                (o for o in opp_actors if o.track_id == pairing.presser_track_id),
                None,
            )

        if rec_actor is not None:
            manifold_targets = generate_receiver_manifold(
                passer_pos=passer_pos,
                receiver=rec_actor,
                nearest_presser=nearest_presser_actor,
                teammates=recv_actors,
                attack_dir_x=attack_dir_x,
            )
        else:
            raw_x = cand["pitch_xy"][0]
            raw_y = cand["pitch_xy"][1]
            rec_vx = cand.get("pitch_vel_ms", [0.0, 0.0])[0]
            rec_vy = cand.get("pitch_vel_ms", [0.0, 0.0])[1]
            raw_dist = math.hypot(raw_x - passer_pos.x, raw_y - passer_pos.y)
            pass_flight_s = flight_time_s(raw_dist)
            raw_tgt = lead_target(
                PitchPoint(x=raw_x, y=raw_y), (rec_vx, rec_vy), pass_flight_s
            )
            clamped_tgt, disc, mult = clamp_to_pitch_buffer(raw_tgt)
            d_pass = math.hypot(
                clamped_tgt.x - passer_pos.x, clamped_tgt.y - passer_pos.y
            )
            manifold_targets = [
                ManifoldTarget(
                    action_type="FEET",
                    target_pos=clamped_tgt,
                    flight_time_s=flight_time_s(d_pass),
                    pass_dist_m=round(d_pass, 2),
                    offset_dist_m=0.0,
                    is_boundary_discounted=disc,
                    boundary_discount_mult=round(mult, 3),
                    is_ambiguity_penalized=False,
                    ambiguity_partner_id=None,
                    dispersion_sigma_m=round(d_pass * KICK_DISPERSION_RADIAL_RATIO, 2),
                )
            ]

        cand_evals = [
            _eval_single_target(
                cand=cand,
                mt=mt,
                passer_pos=passer_pos,
                opponents=opponents,
                opp_actors=opp_actors,
                rec_actor=rec_actor,
                receiver_candidates=receiver_candidates,
                evaluator=evaluator,
                passer_facing_angle_rad=passer_facing_angle_rad,
                passer_gaze_angle_rad=passer_gaze_angle_rad,
                memory_weights=memory_weights,
                touch_vel_xy_ms=touch_vel_xy_ms,
                nearest_presser_vel_ms=nearest_presser_vel_ms,
                attack_dir_x=attack_dir_x,
                gk_turnover_hazard=gk_turnover_hazard,
                gk_press_dist=gk_press_dist,
                closing_speed_ms=closing_speed_ms,
                pairing=pairing,
                nearest_presser_actor=nearest_presser_actor,
            )
            for mt in manifold_targets
        ]

        best_eval = max(cand_evals, key=lambda x: x["path_score"])
        best_eval["manifold_evals"] = [
            {
                "action_type": e["action_type"],
                "target_pos": e["target_pos"],
                "path_score": e["path_score"],
                "net_ev": e["net_ev"],
                "path_status": e["path_status"],
            }
            for e in cand_evals
        ]
        shielded_eval = next(
            (e for e in cand_evals if e["action_type"] == "SHIELDED_POCKET"),
            None,
        )
        if shielded_eval is not None:
            best_eval["space_target_pos"] = shielded_eval["target_pos"]
            best_eval["space_cushion_s"] = shielded_eval["t_post_cushion_s"]

        option_evals.append(best_eval)

    option_evals.sort(key=lambda x: x["path_score"], reverse=True)
    top_score = option_evals[0]["path_score"] if option_evals else 0.0

    for rank_idx, opt in enumerate(option_evals, 1):
        opt["path_rank"] = rank_idx
        # Release-window membership: physically viable, unblocked, non-hospital option with positive net value
        is_viable_physics = (
            opt["path_status"] not in BAD_STATUSES
            and opt["path_status"] != "OUT_OF_VISION"
            and opt.get("t_post_cushion_s", 99.0) >= MIN_POST_RECEIPT_CUSHION_S
            and opt.get("xp_completion_prob", 0.0) >= MIN_VIABLE_XP
        )
        opt["in_release_window"] = is_viable_physics and opt["path_score"] >= 45.0
        opt["is_optimal"] = bool(opt["in_release_window"])

        if opt["in_release_window"]:
            if rank_idx == 1 and opt["path_score"] >= MIN_RECOMMENDED_SCORE:
                opt["optimality_class"] = "RECOMMENDED"
                opt["grade_color"] = (
                    "#38bdf8" if opt["path_status"] == "DISGUISED_OUTLET" else "#22c55e"
                )
                opt["decision_grade"] = (
                    "DISGUISED_OUTLET"
                    if opt["path_status"] == "DISGUISED_OUTLET"
                    else "RECOMMENDED"
                )
            elif (
                opt["path_score"] >= (top_score - RELEASE_EQUIVALENCE_DELTA_SCORE)
                and opt["path_score"] >= MIN_RECOMMENDED_SCORE
            ):
                # Co-optimal release window equivalence class
                opt["optimality_class"] = "RECOMMENDED"
                opt["grade_color"] = "#22c55e"
                opt["decision_grade"] = "RECOMMENDED"
            else:
                opt["optimality_class"] = "VIABLE_OUTLET"
                if opt["path_status"] == "SCANNED_OPEN":
                    opt["grade_color"] = "#22c55e"
                    opt["decision_grade"] = "SCANNED_OPEN"
                elif opt["path_status"] == "KNOWN_FROM_SCAN":
                    opt["grade_color"] = "#2dd4bf"
                    opt["decision_grade"] = "KNOWN_FROM_SCAN"
                elif opt["path_status"] == "DISGUISED_OUTLET":
                    opt["grade_color"] = "#38bdf8"
                    opt["decision_grade"] = "DISGUISED_OUTLET"
                else:
                    opt["grade_color"] = "#38bdf8"
                    opt["decision_grade"] = "VIABLE_OUTLET"
        else:
            opt["optimality_class"] = opt["path_status"]
            opt["decision_grade"] = opt["path_status"]

    return option_evals


def _task_label(task: Any) -> str:
    role = getattr(task, "role", None)
    role_s = role.value if role is not None else "?"
    kind = task.kind.value if hasattr(task.kind, "value") else str(task.kind)
    target = task.target_id or ""
    if task.second_target_id:
        target += f"+{task.second_target_id}"
    return f"{role_s} {kind} {target} ({task.reason})".strip()


def project_decision_to_screen(
    homography: Any,
    passer_pos: PitchPoint,
    decision_eval: List[Dict[str, Any]],
    facing_rad: float,
    decision_frame_idx: int,
    cone_radius_m: float = 12.0,
    cone_steps: int = 16,
    scan_gaze_samples: Optional[List[float]] = None,
    defender_tasks: Optional[Dict[int, Any]] = None,
    ball_presser_id: Optional[int] = None,
) -> Dict[str, Any]:
    def _px(x_m: float, y_m: float) -> List[int]:
        u, v = homography.to_screen(x_m, y_m, frame_idx=decision_frame_idx)
        return [int(u), int(v)]

    anchor_px = _px(passer_pos.x, passer_pos.y)
    lanes = []
    for opt in decision_eval:
        tx, ty = opt["target_pos"][0], opt["target_pos"][1]
        status = opt.get("path_status", "")
        lanes.append(
            {
                "track_id": opt["track_id"],
                "target_px": _px(float(tx), float(ty)),
                "action_type": opt.get("action_type", "FEET"),
                "offset_dist_m": opt.get("offset_dist_m", 0.0),
                "grade_color": opt["grade_color"],
                "is_optimal": bool(opt.get("is_optimal")),
                "path_rank": int(opt.get("path_rank", 99)),
                "is_bad": status in BAD_STATUSES,
                "is_scan_known": status
                in ("KNOWN_FROM_SCAN", "DISGUISED_OUTLET", "SCANNED_OPEN"),
                "xp": opt["xp_completion_prob"],
                "path_score": opt["path_score"],
                "visual_label": opt["visual_label"],
            }
        )
    cone_px = [anchor_px]
    half = math.radians(70.0)
    for i in range(cone_steps + 1):
        a = facing_rad - half + (2.0 * half * i / cone_steps)
        cone_px.append(
            _px(
                passer_pos.x + cone_radius_m * math.cos(a),
                passer_pos.y + cone_radius_m * math.sin(a),
            )
        )
    facing_tip_px = _px(
        passer_pos.x + cone_radius_m * math.cos(facing_rad),
        passer_pos.y + cone_radius_m * math.sin(facing_rad),
    )
    matchups = []
    for opt in decision_eval:
        if opt.get("presser_track_id") is None:
            continue
        task = (defender_tasks or {}).get(int(opt["presser_track_id"]))
        matchups.append(
            {
                "presser_track_id": opt["presser_track_id"],
                "receiver_track_id": opt["track_id"],
                "beaten": bool(opt.get("margin_s", 99.0) < 0.0),
                "free_man": bool(opt.get("free_man", False)),
                "margin_s": opt.get("margin_s", 99.0),
                "task_label": _task_label(task) if task is not None else "",
            }
        )
    if ball_presser_id is not None:
        task = (defender_tasks or {}).get(int(ball_presser_id))
        matchups.append(
            {
                "presser_track_id": int(ball_presser_id),
                "receiver_track_id": None,
                "is_ball_press": True,
                "beaten": False,
                "free_man": False,
                "margin_s": 99.0,
                "task_label": _task_label(task) if task is not None else "",
            }
        )
    scan_cones_px: List[Dict[str, Any]] = []
    if scan_gaze_samples:
        stride = max(1, len(scan_gaze_samples) // 9)
        for gaze_rad in scan_gaze_samples[::stride]:
            edge_px = [anchor_px]
            for i in range(cone_steps + 1):
                a = gaze_rad - half + (2.0 * half * i / cone_steps)
                edge_px.append(
                    _px(
                        passer_pos.x + cone_radius_m * math.cos(a),
                        passer_pos.y + cone_radius_m * math.sin(a),
                    )
                )
            scan_cones_px.append(
                {"gaze_rad": round(float(gaze_rad), 4), "cone_px": edge_px}
            )
    return {
        "anchor_px": anchor_px,
        "lanes": lanes,
        "cone_px": cone_px,
        "scan_cones_px": scan_cones_px,
        "facing_tip_px": facing_tip_px,
        "matchups": matchups,
    }


def _duel_team(tid: int, own_ids: set, opp_ids: set) -> str:
    if tid in own_ids:
        return "own"
    if tid in opp_ids:
        return "opp"
    return "other"


def _build_duel_payload(
    tracking_records: list[dict[str, Any]],
    decision_frame_idx: int,
    fps: float,
    passer_track_id: Optional[int],
    passer_pos: PitchPoint,
    receivers: List[Dict[str, Any]],
    decision_evaluation: List[Dict[str, Any]],
    attack_dir_x: float,
) -> Dict[str, Any]:
    """Pre-decision duel timelines per paired outlet plus released-outlet resolution.

    Pure over tracking slices: facing falls back to velocity (decision-frame
    pose is not dense), so timelines carry kinematics honestly. No scoring
    input changes here; consumers read the payload downstream.
    """
    payload: Dict[str, Any] = {"duels": {}, "releasedOutlet": None, "duelOutcome": None}
    if not tracking_records or fps <= 0:
        return payload
    goal_sign = 1.0 if attack_dir_x >= 0 else -1.0
    mem_lo = max(0, decision_frame_idx - int(3.5 * fps))
    own_ids = {int(c["track_id"]) for c in receivers}
    if passer_track_id is not None:
        own_ids.add(int(passer_track_id))
    opp_ids: set = set()
    for rec in tracking_records[mem_lo : decision_frame_idx + 1]:
        for e in rec["entities"]:
            if e.get("class_name") == "person" and e.get("track_id") not in own_ids:
                opp_ids.add(int(e["track_id"]))

    def _actor_at(fi: int, tid: int, team: str) -> Optional[PressingActor]:
        ent = next(
            (
                e
                for e in tracking_records[fi]["entities"]
                if e.get("track_id") == tid and e.get("class_name") == "person"
            ),
            None,
        )
        if ent is None:
            return None
        vel = ent.get("pitch_vel_ms", [0.0, 0.0])
        return PressingActor(
            track_id=int(tid),
            team_id=team,
            pos_m=PitchPoint(x=ent["pitch_xy"][0], y=ent["pitch_xy"][1]),
            vel_ms=(float(vel[0]), float(vel[1])),
            pitch_valid=bool(ent.get("pitch_valid", True)),
        )

    lead_by_recv: Dict[int, PitchPoint] = {}
    for opt in decision_evaluation:
        presser_id = opt.get("presser_track_id")
        if presser_id is None:
            continue
        rid = int(opt["track_id"])
        rec_actors, pre_actors, balls, stamps = [], [], [], []
        for fi in range(mem_lo, decision_frame_idx + 1):
            rec = _actor_at(fi, rid, "own")
            pre = _actor_at(fi, int(presser_id), "opp")
            b_xy = _ball_xy(tracking_records[fi]["entities"])
            if rec is None or pre is None or b_xy is None:
                continue
            rec_actors.append(rec)
            pre_actors.append(pre)
            balls.append(PitchPoint(x=b_xy[0], y=b_xy[1]))
            stamps.append(round(fi / fps, 3))
        if len(rec_actors) < 2:
            continue
        dist = math.hypot(
            rec_actors[-1].pos_m.x - passer_pos.x,
            rec_actors[-1].pos_m.y - passer_pos.y,
        )
        lead = lead_target(
            rec_actors[-1].pos_m, rec_actors[-1].vel_ms, flight_time_s(dist)
        )
        lead_by_recv[rid] = lead
        track = track_duel(
            rid,
            int(presser_id),
            rec_actors,
            pre_actors,
            balls,
            stamps,
            corridor_targets=[lead],
            goal_sign=goal_sign,
        )
        payload["duels"][str(rid)] = {
            "presser": int(presser_id),
            "timestamps": [f.timestamp_s for f in track.frames],
            "margins": [f.margin_s for f in track.frames],
            "phases": [f.phase for f in track.frames],
            "roles": [f.presser_role for f in track.frames],
        }

    post = tracking_records[decision_frame_idx:]
    post_balls = [
        tuple(b) if (b := _ball_xy(rec["entities"])) is not None else None
        for rec in post
    ]
    post_persons = [
        [
            (
                int(e["track_id"]),
                _duel_team(int(e["track_id"]), own_ids, opp_ids),
                float(e["pitch_xy"][0]),
                float(e["pitch_xy"][1]),
            )
            for e in rec["entities"]
            if e.get("class_name") == "person"
        ]
        for rec in post
    ]
    passer_xy = _passer_xy(
        tracking_records[decision_frame_idx]["entities"], passer_track_id
    )
    if passer_xy is None or not lead_by_recv:
        return payload
    gaps = [
        round(math.hypot(b[0] - passer_xy[0], b[1] - passer_xy[1]), 2)
        if b is not None
        else None
        for b in post_balls
    ]
    best_rid, best_approach = None, RELEASE_APPROACH_GATE_M
    for rid, lead in lead_by_recv.items():
        approach = min(
            (
                math.hypot(b[0] - lead.x, b[1] - lead.y)
                for b in post_balls[: int(2 * fps)]
                if b is not None
            ),
            default=None,
        )
        if approach is not None and approach < best_approach:
            best_rid, best_approach = rid, approach
    if best_rid is None:
        return payload
    payload["releasedOutlet"] = best_rid
    rel = release_frame_idx(gaps, 0)
    rel_frame = (
        min(decision_frame_idx + rel, len(tracking_records) - 1)
        if rel is not None
        else decision_frame_idx
    )
    rel_ents = tracking_records[rel_frame]["entities"]
    rel_opp = _actors_from_entities(
        [e for e in rel_ents if int(e.get("track_id", -1)) in opp_ids], "opp"
    )
    rel_rec = next(
        (e for e in rel_ents if int(e.get("track_id", -2)) == best_rid), None
    )
    line_x = calculate_offside_line_x(rel_opp, attack_dir_x)
    rec_x = rel_rec["pitch_xy"][0] if rel_rec else 0.0
    offside = (rec_x > line_x) if attack_dir_x >= 0 else (rec_x < line_x)
    outcome = resolve_duel_outcome(
        post_balls,
        gaps,
        post_persons,
        0,
        "own",
        ["opp"],
        fps=fps,
        offside_invalid=bool(offside),
    )
    payload["duelOutcome"] = {
        "outlet": best_rid,
        "outcome": outcome.outcome.value,
        "controller": outcome.controller_track_id,
        "contested": outcome.contested,
        "note": outcome.note,
    }
    return payload


def evaluate_match_episode(
    match: MatchConfig,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Evaluates a single match episode: extracts/loads tracks, runs first-touch detection,
    computes perception, biomechanics, arrival margins, and screen projections.
    """
    video_path = Path(args.video) if args.video else Path(match.video_path)
    calib_path = (
        Path(args.calibration) if args.calibration else Path(match.calibration_path)
    )
    decision_time_s = (
        float(args.decision_time_s)
        if args.decision_time_s is not None
        else float(match.decision_time_s)
    )

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    homography = DynamicPitchHomography.from_calibration_file(
        str(calib_path), default_timestamp_s=decision_time_s
    )

    possession_labels, opponent_labels, keeper_label = _team_sets(match)
    evaluator = DistributionEvaluator(press_danger_radius_m=6.0)

    cache_candidates: list[Path] = []
    if args.cache:
        cache_candidates.append(Path(args.cache))
    cache_candidates.append(Path(f"data/processed/{match.match_id}_tracks.json"))
    if match.match_id == "torino_milan_0708_0720":
        cache_candidates.append(Path("data/processed/episode21_tracks.json"))
    elif match.match_id == "torino_milan_2500_2525":
        cache_candidates.append(Path("data/processed/episode_2500_2525_tracks.json"))
    elif match.match_id == "torino_milan_7818_7828":
        cache_candidates.append(Path("data/processed/episode_7818_7828_tracks.json"))

    cache_path = next((p for p in cache_candidates if p.exists()), None)
    use_cache = (not args.no_cache) and (cache_path is not None)

    if use_cache and cache_path is not None:
        tracking_records = json.loads(cache_path.read_text())
        fps = (
            float(tracking_records[0].get("fps", match.fps))
            if tracking_records
            else float(match.fps)
        )
        total_frames = len(tracking_records)
        decision_frame_idx = min(total_frames - 1, int(round(decision_time_s * fps)))
        homography.fps = float(fps)
        print(
            f"[{match.match_id}] Loaded cached tracks: {cache_path} ({total_frames} frames @ {fps}fps)"
        )
        cap = None
        tracker = None
        pose_estimator = None
    else:
        try:
            import cv2
            from src.cv.player_detector import YOLOPlayerDetector
            from src.cv.player_tracker import MultiObjectVideoTracker
            from src.cv.pose_estimator import PlayerPoseEstimator
        except ImportError as e:
            raise ImportError(
                f"Video extraction requires OpenCV and PyTorch. Run with python3.11: {e}"
            ) from e

        detector = YOLOPlayerDetector(
            weights_path="yolov8m.pt",
            homography=homography,
            conf_threshold=0.20,
            ball_conf_threshold=0.08,
            kits=list(match.kits),
        )
        tracker = MultiObjectVideoTracker(
            detector=detector,
            homography=homography,
            max_age=15,
            min_hits=1,
            dist_threshold_px=75.0,
            fps=float(match.fps),
        )
        pose_estimator = PlayerPoseEstimator(
            weights_path="yolov8n-pose.pt", homography=homography
        )

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or float(match.fps)
        tracker.fps = float(fps)
        homography.fps = float(fps)
        print(
            f"[{match.match_id}] Tracking {match.keeper_name} across {total_frames} frames @ {fps}fps..."
        )
        tracking_records = []
        decision_frame_idx = int(round(decision_time_s * fps))

    frame_idx = len(tracking_records) if use_cache else 0
    facing_by_track: Dict[int, tuple[float, str]] = {}
    opp_facing_by_track: Dict[int, tuple[float, str]] = {}
    passer_track_id: int | None = None
    passer_pos = PitchPoint(x=95.0, y=34.0)
    receivers: List[Dict[str, Any]] = []
    opponents: List[PitchPoint] = []
    gk_facing_angle_rad = 1.571
    gk_gaze_angle_rad = 1.571
    gk_pose_window: List[Tuple[int, Any]] = []
    gk_gaze_history: Dict[int, float] = {}
    gk_gaze_conf: Dict[int, float] = {}

    while cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_s = round(frame_idx / fps, 2)
        active_tracks = tracker.update(frame, frame_idx)

        frame_entities = []
        for t in active_tracks:
            ent = {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "team_label": t.team_label,
                "color": t.team_color_hex,
                "screen_uv": [
                    round(float(t.state_xyuv[0]), 1),
                    round(float(t.state_xyuv[1]), 1),
                ],
                "pitch_xy": [
                    round(float(t.pitch_pos_m[0]), 2),
                    round(float(t.pitch_pos_m[1]), 2),
                ],
                "speed_ms": round(float(t.speed_ms), 1),
                "pitch_vel_ms": [
                    round(float(t.pitch_vel_ms[0]), 2),
                    round(float(t.pitch_vel_ms[1]), 2),
                ],
                "bbox": [round(float(x), 1) for x in t.bbox_xyxy],
                "pitch_valid": bool(getattr(t, "pitch_valid", True)),
            }
            frame_entities.append(ent)

        # Track keeper pose across pre-decision window to smooth out stride cycle and arm cross occlusions
        if (
            pose_estimator is not None
            and max(0, decision_frame_idx - 15) <= frame_idx <= decision_frame_idx
        ):
            gk_cand, _, _ = _split_frame(
                frame_entities, possession_labels, opponent_labels, match
            )
            if gk_cand is not None and "bbox" in gk_cand:
                gk_uv = tuple(gk_cand.get("screen_uv", [0.0, 0.0]))
                gk_vel = _screen_velocity_px_per_frame(
                    tracking_records,
                    int(gk_cand["track_id"]),
                    (float(gk_uv[0]), float(gk_uv[1])),
                    frame_idx,
                )
                p_res = pose_estimator.estimate_pose_in_crop(
                    frame,
                    tuple(gk_cand["bbox"]),
                    frame_idx=frame_idx,
                    velocity_screen_uv=gk_vel,
                )
                if p_res.is_valid:
                    gk_pose_window.append((frame_idx, p_res))
                    if p_res.gaze_valid:
                        gk_gaze_history[frame_idx] = p_res.gaze_facing_rad
                        gk_gaze_conf[frame_idx] = p_res.gaze_confidence

        if frame_idx == decision_frame_idx:
            gk_ent, team_ents, opp_ents = _split_frame(
                frame_entities, possession_labels, opponent_labels, match
            )
            passer = gk_ent
            if passer is None and team_ents:
                b_xy = _ball_xy(frame_entities)
                bx, by = (b_xy[0], b_xy[1]) if b_xy else (95.0, 34.0)
                passer = min(
                    team_ents,
                    key=lambda p: (p["pitch_xy"][0] - bx) ** 2
                    + (p["pitch_xy"][1] - by) ** 2,
                )
            if passer is not None:
                passer_track_id = int(passer["track_id"])
                passer_pos = PitchPoint(
                    x=passer["pitch_xy"][0], y=passer["pitch_xy"][1]
                )

                if gk_pose_window and pose_estimator is not None:
                    smoothed_gk = pose_estimator.smooth_pose_sequence(
                        [res for _, res in gk_pose_window]
                    )
                    if smoothed_gk.is_valid:
                        gk_facing_angle_rad = smoothed_gk.torso_facing_rad
                        gk_gaze_angle_rad = smoothed_gk.gaze_facing_rad
                elif "bbox" in passer and pose_estimator is not None:
                    passer_uv = tuple(passer.get("screen_uv", [0.0, 0.0]))
                    passer_vel = _screen_velocity_px_per_frame(
                        tracking_records,
                        int(passer["track_id"]),
                        (float(passer_uv[0]), float(passer_uv[1])),
                        frame_idx,
                    )
                    pose_res = pose_estimator.estimate_pose_in_crop(
                        frame,
                        tuple(passer["bbox"]),
                        frame_idx=frame_idx,
                        velocity_screen_uv=passer_vel,
                    )
                    if pose_res.is_valid:
                        gk_facing_angle_rad = pose_res.torso_facing_rad
                        gk_gaze_angle_rad = pose_res.gaze_facing_rad

                receivers = [
                    p
                    for p in team_ents
                    if p["track_id"] != passer["track_id"]
                    and p.get("pitch_valid", True)
                ]
                opponents = [
                    PitchPoint(x=p["pitch_xy"][0], y=p["pitch_xy"][1])
                    for p in opp_ents
                    if p.get("pitch_valid", True)
                ]

                for p in receivers:
                    if "bbox" in p and pose_estimator is not None:
                        p_uv = tuple(p.get("screen_uv", [0.0, 0.0]))
                        p_vel = _screen_velocity_px_per_frame(
                            tracking_records,
                            int(p["track_id"]),
                            (float(p_uv[0]), float(p_uv[1])),
                            frame_idx,
                        )
                        cand_pose = pose_estimator.estimate_pose_in_crop(
                            frame,
                            tuple(p["bbox"]),
                            frame_idx=frame_idx,
                            velocity_screen_uv=p_vel,
                        )
                        if cand_pose.is_valid:
                            facing_by_track[int(p["track_id"])] = (
                                float(cand_pose.torso_facing_rad),
                                "pose",
                            )

                if pose_estimator is not None:
                    for o in opp_ents:
                        facing = _pose_torso_facing(
                            pose_estimator, frame, o, tracking_records, frame_idx
                        )
                        if facing is not None:
                            opp_facing_by_track[int(o["track_id"])] = facing

        tracking_records.append(
            {
                "frame": frame_idx,
                "timestamp_s": timestamp_s,
                "fps": round(float(fps), 3),
                "entities": frame_entities,
            }
        )
        frame_idx += 1

    if cap is not None:
        cap.release()
        save_cache_path = Path(f"data/processed/{match.match_id}_tracks.json")
        save_cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_cache_path.write_text(json.dumps(tracking_records))

    if use_cache:
        dec = tracking_records[decision_frame_idx]
        gk_dec, team_dec, opp_dec = _split_frame(
            dec["entities"], possession_labels, opponent_labels, match
        )
        if gk_dec is not None:
            passer_track_id = int(gk_dec["track_id"])
            passer_pos = PitchPoint(x=gk_dec["pitch_xy"][0], y=gk_dec["pitch_xy"][1])
        elif team_dec:
            best = min(team_dec, key=lambda p: p["pitch_xy"][0])
            passer_pos = PitchPoint(x=best["pitch_xy"][0], y=best["pitch_xy"][1])
        receivers = [
            p for p in team_dec if gk_dec is None or p["track_id"] != gk_dec["track_id"]
        ]
        opponents = [
            PitchPoint(x=p["pitch_xy"][0], y=p["pitch_xy"][1]) for p in opp_dec
        ]

        if video_path.exists():
            cap_pose = None
            try:
                import cv2
                from src.cv.pose_estimator import PlayerPoseEstimator

                cap_pose = cv2.VideoCapture(str(video_path))
                if cap_pose.isOpened():
                    pose_est = PlayerPoseEstimator(
                        weights_path="yolov8n-pose.pt", homography=homography
                    )
                    window_start = max(0, decision_frame_idx - 15)
                    cap_pose.set(cv2.CAP_PROP_POS_FRAMES, window_start)
                    gk_pose_window = []
                    for fi in range(window_start, decision_frame_idx + 1):
                        ret, fr = cap_pose.read()
                        if not ret:
                            break
                        if fi < len(tracking_records):
                            gk_e, _, _ = _split_frame(
                                tracking_records[fi]["entities"],
                                possession_labels,
                                opponent_labels,
                                match,
                            )
                            if gk_e is not None and "bbox" in gk_e:
                                e_uv = tuple(gk_e.get("screen_uv", [0.0, 0.0]))
                                e_vel = _screen_velocity_px_per_frame(
                                    tracking_records,
                                    int(gk_e["track_id"]),
                                    (float(e_uv[0]), float(e_uv[1])),
                                    fi,
                                )
                                p_res = pose_est.estimate_pose_in_crop(
                                    fr,
                                    tuple(gk_e["bbox"]),
                                    frame_idx=fi,
                                    velocity_screen_uv=e_vel,
                                )
                                if p_res.is_valid:
                                    gk_pose_window.append((fi, p_res))
                                    if p_res.gaze_valid:
                                        gk_gaze_history[fi] = p_res.gaze_facing_rad
                                        gk_gaze_conf[fi] = p_res.gaze_confidence
                    if gk_pose_window:
                        smoothed_gk = pose_est.smooth_pose_sequence(
                            [r for _, r in gk_pose_window]
                        )
                        if smoothed_gk.is_valid:
                            gk_facing_angle_rad = smoothed_gk.torso_facing_rad
                            gk_gaze_angle_rad = smoothed_gk.gaze_facing_rad
                    cap_pose.set(cv2.CAP_PROP_POS_FRAMES, decision_frame_idx)
                    ret, dec_frame = cap_pose.read()
                    if ret:
                        for p in receivers:
                            if "bbox" in p:
                                rp_uv = tuple(p.get("screen_uv", [0.0, 0.0]))
                                rp_vel = _screen_velocity_px_per_frame(
                                    tracking_records,
                                    int(p["track_id"]),
                                    (float(rp_uv[0]), float(rp_uv[1])),
                                    decision_frame_idx,
                                )
                                cand_pose = pose_est.estimate_pose_in_crop(
                                    dec_frame,
                                    tuple(p["bbox"]),
                                    frame_idx=decision_frame_idx,
                                    velocity_screen_uv=rp_vel,
                                )
                                if cand_pose.is_valid:
                                    facing_by_track[int(p["track_id"])] = (
                                        float(cand_pose.torso_facing_rad),
                                        "pose",
                                    )
                                for o in opp_dec:
                                    facing = _pose_torso_facing(
                                        pose_est,
                                        dec_frame,
                                        o,
                                        tracking_records,
                                        decision_frame_idx,
                                    )
                                    if facing is not None:
                                        opp_facing_by_track[int(o["track_id"])] = facing
                    cap_pose.release()
            except Exception as exc:
                print(f"[pose] cache-window pose replay failed: {exc}")
                raise
            finally:
                if cap_pose is not None:
                    cap_pose.release()

    # First-touch detection
    first_touch_note = ""
    if tracking_records and passer_track_id is not None:
        lo = max(0, decision_frame_idx - 25)
        hi = min(len(tracking_records), decision_frame_idx + 10)
        ball_traj = [_ball_xy(tracking_records[fi]["entities"]) for fi in range(lo, hi)]
        passer_traj = [
            _passer_xy(tracking_records[fi]["entities"], passer_track_id)
            for fi in range(lo, hi)
        ]
        touch = find_first_touch(ball_traj, passer_traj, decision_frame_idx - lo)
        if touch is not None:
            touch_frame = lo + touch.frame_offset
            gap_txt = (
                f"{touch.gap_at_decision_m}m at F{decision_frame_idx}"
                if touch.gap_at_decision_m is not None
                else "no ball at decision frame"
            )
            first_touch_note = f"first-touch F{touch_frame} (ball {touch.gap_m}m from passer; {gap_txt})"
            if touch_frame != decision_frame_idx:
                decision_frame_idx = touch_frame
                t_gk, t_team, t_opp = _split_frame(
                    tracking_records[touch_frame]["entities"],
                    possession_labels,
                    opponent_labels,
                    match,
                )
                if t_gk is not None:
                    passer_track_id = int(t_gk["track_id"])
                passer_pos = PitchPoint(x=touch.ball_xy_m[0], y=touch.ball_xy_m[1])
                receivers = [
                    p
                    for p in t_team
                    if t_gk is None or p["track_id"] != t_gk["track_id"]
                ]
                opponents = [
                    PitchPoint(x=p["pitch_xy"][0], y=p["pitch_xy"][1]) for p in t_opp
                ]

    # Decision possession frames
    full_gaps: List[Optional[float]] = []
    for rec in tracking_records:
        b_xy = _ball_xy(rec["entities"])
        g_xy = _passer_xy(rec["entities"], passer_track_id)
        if b_xy is None or g_xy is None:
            full_gaps.append(None)
        else:
            full_gaps.append(round(math.hypot(b_xy[0] - g_xy[0], b_xy[1] - g_xy[1]), 2))
    decision_frames = decision_possession_frames(full_gaps)
    if not decision_frames and tracking_records:
        decision_frames = list(
            range(
                max(0, decision_frame_idx - 12),
                min(len(tracking_records), decision_frame_idx + 51),
            )
        )

    for p in receivers:
        f = facing_by_track.get(p["track_id"])
        if f is not None:
            p["facing_rad"] = f[0]
            p["facing_source"] = f[1]

    eval_frame_ents = tracking_records[decision_frame_idx]["entities"]
    _, _, eval_opp_ents = _split_frame(
        eval_frame_ents, possession_labels, opponent_labels, match
    )
    for o in eval_opp_ents:
        f = opp_facing_by_track.get(o["track_id"])
        if f is not None:
            o["facing_rad"] = f[0]
            o["facing_source"] = f[1]
    opponent_actor_list = _actors_from_entities(
        eval_opp_ents, "opp", opp_facing_by_track
    )
    print(
        f"[{match.match_id}] pose facing: {len(facing_by_track)}/{len(receivers)} "
        f"receivers, {len(opp_facing_by_track)}/{len(eval_opp_ents)} opponents"
    )

    press_note = ""
    if tracking_records:
        own_actors = _actors_from_entities(eval_frame_ents, "own", facing_by_track)
        press_lines = cluster_lines(
            opponent_actor_list, "opp", attack_dir_x=-float(match.attack_dir_x)
        )
        build_lines = cluster_lines(
            own_actors, "own", attack_dir_x=float(match.attack_dir_x)
        )
        ball_now = _ball_xy(eval_frame_ents)
        state = press_state(
            opponent_actor_list,
            ball_x=ball_now[0] if ball_now else passer_pos.x,
            ball_y=ball_now[1] if ball_now else passer_pos.y,
            attack_dir_x=-float(match.attack_dir_x),
        )
        press_note = (
            f"press {press_lines.label} vs buildup {build_lines.label} "
            f"({state.note}, intensity {state.intensity_ms} m/s)"
        )

    times_s = []
    ball_xs = []
    ball_ys = []
    lo_approach = max(0, decision_frame_idx - 8)
    for fi in range(lo_approach, decision_frame_idx + 1):
        b_ent = next(
            (
                e["pitch_xy"]
                for e in tracking_records[fi]["entities"]
                if e.get("class_name") == "sports ball"
            ),
            None,
        )
        if b_ent:
            times_s.append((fi - lo_approach) / fps)
            ball_xs.append(b_ent[0])
            ball_ys.append(b_ent[1])

    touch_vel_xy = None
    if len(times_s) >= 4:
        t_arr = np.array(times_s)
        vx = float(np.polyfit(t_arr, np.array(ball_xs), 1)[0])
        vy = float(np.polyfit(t_arr, np.array(ball_ys), 1)[0])
        touch_vel_xy = (round(vx, 2), round(vy, 2))

    nearest_presser_vel = None
    nearest_opp_candidates = [e for e in eval_opp_ents if e.get("pitch_valid", True)]
    if nearest_opp_candidates:
        nearest_t = min(
            nearest_opp_candidates,
            key=lambda p: (p["pitch_xy"][0] - passer_pos.x) ** 2
            + (p["pitch_xy"][1] - passer_pos.y) ** 2,
        )
        prev_idx = max(0, decision_frame_idx - 5)
        f_prev_t = next(
            (
                e
                for e in tracking_records[prev_idx]["entities"]
                if e.get("track_id") == nearest_t["track_id"]
            ),
            nearest_t,
        )
        dt_p = (decision_frame_idx - prev_idx) / fps
        if dt_p > 0 and f_prev_t.get("pitch_valid", True):
            nearest_presser_vel = (
                (nearest_t["pitch_xy"][0] - f_prev_t["pitch_xy"][0]) / dt_p,
                (nearest_t["pitch_xy"][1] - f_prev_t["pitch_xy"][1]) / dt_p,
            )

    memory_weights: Dict[int, float] = {}
    scan_first_seen: Dict[int, float] = {}
    if tracking_records:
        mem_lo = max(0, decision_frame_idx - int(3.5 * fps))
        last_seen: Dict[int, float] = {}
        for fi in range(mem_lo, min(decision_frame_idx + 1, len(tracking_records))):
            ents = tracking_records[fi]["entities"]
            t_now = fi / fps
            gk_xy = _passer_xy(ents, passer_track_id)
            if gk_xy is None:
                continue
            if fi not in gk_gaze_history:
                # No pose measurement on this frame: decay prior scans without
                # fabricating a new gaze direction from the ball bearing.
                last_seen, memory_weights = update_spatial_memory_buffer(
                    last_seen,
                    [],
                    t_now,
                    tactical_prior_floor=TACTICAL_PRIOR_FLOOR,
                    prior_weights=memory_weights,
                )
                continue
            facing = gk_gaze_history[fi]
            frame_gaze_conf = max(0.0, min(1.0, float(gk_gaze_conf.get(fi, 1.0))))
            persons = [
                (e["track_id"], (e["pitch_xy"][0], e["pitch_xy"][1]))
                for e in ents
                if e.get("class_name") == "person"
                and not is_official_label(e.get("team_label", ""))
            ]
            opp_obstacles = [
                PitchPoint(x=e["pitch_xy"][0], y=e["pitch_xy"][1])
                for e in ents
                if e.get("team_label") in opponent_labels and e.get("pitch_valid", True)
            ]
            visible = visible_person_ids(
                (gk_xy[0], gk_xy[1]),
                facing,
                persons,
                obstacles=opp_obstacles,
            )
            for tid in visible:
                if tid not in scan_first_seen:
                    scan_first_seen[tid] = t_now
            last_seen, memory_weights = update_spatial_memory_buffer(
                last_seen,
                visible,
                t_now,
                tactical_prior_floor=TACTICAL_PRIOR_FLOOR,
                visible_confidences={tid: frame_gaze_conf for tid in visible},
                prior_weights=memory_weights,
            )
    decision_time_s = decision_frame_idx / fps if fps else 0.0
    scan_windows = summarize_scan_windows(
        scan_first_seen,
        dict(last_seen) if tracking_records else {},
        memory_weights,
        decision_time_s,
    )
    measured_gaze_samples = [gk_gaze_history[fi] for fi in sorted(gk_gaze_history)]
    scan_envelope = merge_swept_cones(measured_gaze_samples, math.radians(70.0))

    decision_evaluation = evaluate_distribution_decision(
        passer_pos,
        receivers,
        opponents,
        evaluator,
        passer_facing_angle_rad=gk_facing_angle_rad,
        passer_gaze_angle_rad=gk_gaze_angle_rad,
        touch_vel_xy_ms=touch_vel_xy,
        nearest_presser_vel_ms=nearest_presser_vel,
        attack_dir_x=float(match.attack_dir_x),
        memory_weights=memory_weights,
        opponent_actors=opponent_actor_list,
    )

    # Display-only designation sweep: same deterministic inputs as the scoring
    # path, recomputed here so every opponent's task reaches the dashboard
    # without touching any scoring code.
    display_recv = [
        PressingActor(
            track_id=int(c["track_id"]),
            team_id="own",
            pos_m=PitchPoint(x=c["pitch_xy"][0], y=c["pitch_xy"][1]),
            vel_ms=(
                float(c.get("pitch_vel_ms", [0.0, 0.0])[0]),
                float(c.get("pitch_vel_ms", [0.0, 0.0])[1]),
            ),
            facing_rad=c.get("facing_rad"),
            facing_source=c.get("facing_source", "unknown"),
            pitch_valid=bool(c.get("pitch_valid", True)),
        )
        for c in receivers
    ]
    display_matchup = assign_pressers(
        PressingSnapshot(
            passer=PressingActor(
                track_id=-999,
                team_id="own",
                pos_m=passer_pos,
                facing_rad=gk_facing_angle_rad,
                facing_source="pose",
            ),
            receivers=display_recv,
            opponents=opponent_actor_list,
            ball_pos_m=passer_pos,
            attack_dir_x=float(match.attack_dir_x),
            memory_weights=memory_weights,
        )
    )
    display_tasks = display_matchup.defender_tasks
    surplus_ids = sorted(
        tid for tid, t in display_tasks.items() if t.reason == "surplus-converger"
    )
    if surplus_ids:
        press_note += (
            f" · surplus converger{'s' if len(surplus_ids) > 1 else ''} "
            + ", ".join(f"#{tid}" for tid in surplus_ids)
        )

    decision_overlay = project_decision_to_screen(
        homography,
        passer_pos,
        decision_evaluation,
        gk_gaze_angle_rad,
        int(decision_frame_idx),
        cone_radius_m=12.0,
        scan_gaze_samples=measured_gaze_samples,
        defender_tasks=display_tasks,
        ball_presser_id=display_matchup.ball_presser_id,
    )

    all_cand_evals: List[Dict[str, Any]] = []
    for opt in decision_evaluation:
        all_cand_evals.append(opt)
        for m_eval in opt.get("manifold_evals", []):
            if m_eval.get("action_type") != opt.get("action_type"):
                sub_opt = dict(opt)
                sub_opt.update(m_eval)
                all_cand_evals.append(sub_opt)

    frontier_result = compute_decision_frontier(all_cand_evals)

    duel_payload = _build_duel_payload(
        tracking_records,
        int(decision_frame_idx),
        float(fps),
        passer_track_id,
        passer_pos,
        receivers,
        decision_evaluation,
        float(match.attack_dir_x),
    )

    opt_rows = []
    for opt in decision_evaluation:
        badge = (
            f"<span class='badge' style='background:{opt['grade_color']}33; "
            f"color:{opt['grade_color']}; border:1px solid {opt['grade_color']}66;'>"
            f"{opt['optimality_class']}</span>"
        )
        opt_tag = (
            " <span class='badge' style='background:#22c55e22; color:#22c55e; border:1px solid #22c55e66; margin-left:4px; font-size:0.7rem;'>IN WINDOW</span>"
            if opt.get("in_release_window", opt.get("is_optimal"))
            else ""
        )
        # True Field of View column
        if opt.get("in_visual_cone"):
            fov_label = f"IN VISION ({opt.get('diff_deg', 0):.0f}°)"
            fov_color = "#22c55e"
        elif opt.get("path_status") == "DISGUISED_OUTLET":
            fov_label = f"DISGUISED ({opt.get('memory_weight', 0):.0%})"
            fov_color = "#38bdf8"
        elif opt.get("path_status") == "SCANNED_OPEN":
            fov_label = f"SCANNED ({opt.get('memory_weight', 0):.0%})"
            fov_color = "#22c55e"
        elif opt.get("scan_known"):
            fov_label = f"KNOWN FROM SCAN ({opt.get('memory_weight', 0):.0%})"
            fov_color = "#2dd4bf"
        else:
            fov_label = f"OUT OF CONE ({opt.get('diff_deg', 0):.0f}°)"
            fov_color = "#94a3b8"

        action_tag = opt.get("action_type", "FEET")
        action_badge = ""
        if action_tag == "PROGRESSIVE_CHANNEL":
            action_badge = f"<br><span style='font-size:0.7rem; color:#38bdf8; font-weight:600;'>CHANNEL (+{opt.get('offset_dist_m', 0.0):.1f}m)</span>"
        elif action_tag == "SHIELDED_POCKET":
            action_badge = f"<br><span style='font-size:0.7rem; color:#facc15; font-weight:600;'>SHIELDED (+{opt.get('offset_dist_m', 0.0):.1f}m)</span>"
        else:
            action_badge = "<br><span style='font-size:0.7rem; color:var(--text-muted);'>FEET</span>"

        opt_rows.append(
            f"<tr>"
            f"<td><strong>#{opt['track_id']}</strong>{opt_tag}{action_badge}</td>"
            f"<td>{opt['distance_m']} m</td>"
            f"<td><span style='color:{fov_color}; font-weight:600;'>{fov_label}</span></td>"
            f"<td>{opt['effective_press_m']} m</td>"
            f"<td><strong>{opt['bypassed_pressers']}</strong> opponents</td>"
            f"<td style='color:var(--text-muted); font-size:0.8rem;'>{opt['bottleneck_diagnostic']}</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )

    n_frames = len(tracking_records)
    duration_s = (n_frames / fps) if fps else 0.0

    return {
        "id": match.match_id,
        "label": match.title or match.match_id,
        "sublabel": match.tactical_context or "Decision Valuation",
        "title": f"{match.keeper_name} Distribution Valuation ({match.title or match.match_id})",
        "subtitle": f"Module M7 / M2.1: Goalkeeper Decision-Value · {match.tactical_context or ''}{' · ' + press_note if press_note else ''}",
        "badge": f"{match.keeper_name.upper()} (PURPLE KIT)",
        "meta": f"Torino vs. AC Milan | {duration_s:.1f}s Ingestion ({n_frames} frames)",
        "videoSrc": f"../data/video_raw/clips/{video_path.name}",
        "fps": round(float(fps), 2),
        "nFrames": n_frames,
        "decisionIdx": int(decision_frame_idx),
        "decisionGk": [round(passer_pos.x, 2), round(passer_pos.y, 2)],
        "keeperTrackId": passer_track_id,
        "facingRad": round(float(gk_facing_angle_rad), 4),
        "gazeRad": round(float(gk_gaze_angle_rad), 4),
        "scannedAngles": [round(float(a), 4) for a in measured_gaze_samples],
        "scanEnvelope": [[float(s), float(e)] for s, e in scan_envelope],
        "scanWindows": scan_windows,
        "scanFrames": {
            str(fi): round(float(a), 4) for fi, a in sorted(gk_gaze_history.items())
        },
        "coneRadiusM": 12.0,
        "firstTouchNote": first_touch_note,
        "pressNote": press_note,
        "tableTimeNote": f"Release-Point Option Telemetry (t = {decision_frame_idx / fps:.2f}s · {first_touch_note})",
        "decisionFrames": sorted(decision_frames),
        "decisionPx": decision_overlay,
        "evalData": decision_evaluation,
        "frontier": frontier_result.to_dict(),
        "duels": duel_payload["duels"],
        "releasedOutlet": duel_payload["releasedOutlet"],
        "duelOutcome": duel_payload["duelOutcome"],
        "opponentsPitch": [
            [round(float(op.x), 2), round(float(op.y), 2)] for op in opponents
        ],
        "trackingData": tracking_records,
        "tableRowsHtml": "".join(opt_rows),
    }


def generate_multi_episode_dashboard_html(
    episodes: list[Dict[str, Any]],
    out_html: Path,
) -> None:
    """
    Renders an interactive tactical decision-value dashboard equipped with an
    Episode Switcher bar to toggle across multiple match clips seamlessly.
    """
    episodes_dict = {ep["id"]: ep for ep in episodes}
    episodes_json = json.dumps(episodes_dict)
    default_ep_key = episodes[0]["id"]
    default_ep = episodes[0]

    # Render episode switcher pills
    pills_html = []
    for idx, ep in enumerate(episodes):
        is_first = idx == 0
        active_style = (
            "background:#2563eb; color:#ffffff; border:1px solid #38bdf8;"
            if is_first
            else "background:#1e293b; color:#94a3b8; border:1px solid #334155;"
        )
        pills_html.append(
            f'<button class="ep-pill-btn" data-ep="{ep["id"]}" style="{active_style} padding:0.45rem 0.95rem; border-radius:0.5rem; font-size:0.8rem; font-weight:700; cursor:pointer; transition:0.2s;">'
            f"{ep['label']}"
            f"</button>"
        )
    switcher_pills_markup = "\n".join(pills_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Goalkeeper Distribution Valuation Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0a0f1d;
            --bg-secondary: #111827;
            --bg-card: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-yellow: #facc15;
            --accent-red: #ef4444;
            --maignan-purple: #c026d3;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        body {{ background: var(--bg-primary); color: var(--text); padding: 1.5rem; line-height: 1.5; }}
        .container {{ max-width: 1440px; margin: 0 auto; }}
        header {{ border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ padding: 0.25rem 0.65rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; display: inline-block; }}
        .badge-green {{ background: rgba(34, 197, 94, 0.15); color: var(--accent-green); border: 1px solid rgba(34, 197, 94, 0.3); }}
        .cockpit-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
        @media (max-width: 1024px) {{ .cockpit-grid {{ grid-template-columns: 1fr; }} }}
        .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.25rem; }}
        .card-header {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }}
        .viewport {{ position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 0.5rem; overflow: hidden; border: 1px solid var(--border); }}
        video {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
        .overlay-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
        #video-overlay .lane-hit {{ pointer-events: stroke; cursor: pointer; }}
        .pitch-canvas {{ width: 100%; height: auto; background: #0f3d1e; border-radius: 0.5rem; border: 2px solid #22c55e; }}
        .controls-bar {{ background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.75rem 1.25rem; margin-top: 1rem; display: flex; align-items: center; gap: 1rem; }}
        button.action-btn {{ background: #2563eb; color: #fff; border: none; padding: 0.5rem 1.2rem; border-radius: 0.375rem; font-weight: 600; cursor: pointer; transition: 0.2s; }}
        button.action-btn:hover {{ background: #1d4ed8; }}
        .scrubber {{ flex: 1; accent-color: #38bdf8; cursor: pointer; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.85rem; }}
        th, td {{ padding: 0.65rem 0.85rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg-secondary); color: var(--text-muted); font-weight: 600; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Multi-Episode Switcher Bar -->
        <div class="switcher-bar">
            <div style="display:flex; align-items:center; gap:0.85rem; flex-wrap:wrap;">
                <span style="font-size:0.8rem; font-weight:800; color:var(--accent-blue); letter-spacing:0.05em; text-transform:uppercase;">Scenario Switcher:</span>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                    {switcher_pills_markup}
                </div>
            </div>
            <div id="scenario-context-tag" style="font-size:0.82rem; color:var(--text-muted); font-weight:600;">{default_ep["sublabel"]}</div>
        </div>

        <header>
            <div>
                <h1 id="header-title" style="font-size: 1.45rem; font-weight: 800;">{default_ep["title"]}</h1>
                <p id="header-subtitle" style="color: var(--text-muted); font-size: 0.88rem;">{default_ep["subtitle"]}</p>
            </div>
            <div style="text-align: right;">
                <div id="header-badge" class="badge badge-green">{default_ep["badge"]}</div>
                <div id="header-meta" style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.25rem;">{default_ep["meta"]}</div>
            </div>
        </header>

        <!-- TOP: THE QUANTITATIVE DECISION COCKPIT (1. Decision Frontier hero, video below; 2D physics hidden) -->
        <div class="cockpit-grid">
            <!-- Left: 1. Risk-Reward Decision Frontier -->
            <div class="card" id="frontier-card">
                <div class="card-header">
                    <span>1. Decision Frontier (Defensive Cushion vs. Net EV)</span>
                    <span style="display:flex; gap:0.5rem; align-items:center;">
                        <a href="#" id="physics-toggle" style="font-size:0.72rem; color:var(--text-muted);">2D physics (internal)</a>
                        <span id="frontier-status-badge" class="badge"></span>
                    </span>
                </div>
                <div style="position:relative; width:100%; aspect-ratio: 500 / 320; background:#0b1329; border-radius:0.5rem; border:1px solid var(--border); overflow:hidden;">
                    <svg id="frontier-svg" viewBox="0 0 500 320" style="width:100%; height:100%; display:block;"></svg>
                    <div id="frontier-tooltip" style="display:none; position:absolute; pointer-events:none; background:rgba(15,23,42,0.96); border:1px solid #38bdf8; border-radius:6px; padding:8px 12px; font-size:0.75rem; color:#f8fafc; z-index:20; box-shadow:0 4px 16px rgba(0,0,0,0.6); line-height:1.4;"></div>
                </div>
                <div style="display:flex; gap:1rem; margin-top:0.75rem; font-size:0.75rem; color:var(--text-muted); flex-wrap:wrap;">
                    <div><span style="display:inline-block;width:14px;height:2px;background:#38bdf8;border-bottom:1px dashed #38bdf8;vertical-align:middle;"></span> Pareto Frontier</div>
                    <div><span style="color:#facc15;font-weight:bold;">★</span> Optimal Target</div>
                    <div><span style="display:inline-block;width:8px;height:8px;background:#64748b;transform:rotate(45deg);vertical-align:middle;"></span> Clearance Baseline</div>
                    <div><span style="color:#38bdf8;">┊</span> τ_safe Cap (2.5s)</div>
                </div>
                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.35rem; line-height:1.4;">
                    Evaluates multi-objective trade-off: Safety Margin (<span style="color:#38bdf8;">Δt</span>) vs Tactical Progression (<span style="color:#22c55e;">ΔxG</span>) with touchline rollout discount.
                </div>
            </div>

            <!-- Right: 2. Spatial Physics & Interception Map (internal tool, hidden by default) -->
            <div class="card" id="spatial-card" style="display:none;">
                <div class="card-header">
                    <span>2. Spatial Physics & Interception Map</span>
                    <span style="display:flex; gap:0.5rem; align-items:center;">
                        <label style="font-size:0.75rem; color:var(--text-muted);"><input type="checkbox" id="lanes-toggle" checked style="accent-color:#22c55e;"> decision lanes</label>
                        <span class="badge" style="background: rgba(192, 38, 211, 0.2); color: #e879f9; border: 1px solid #c026d3;">GK: Maignan</span>
                    </span>
                </div>
                <div style="position:relative; width:100%; aspect-ratio: 525 / 340; background:#0f3d1e; border-radius:0.5rem; border:1px solid var(--border); overflow:hidden;">
                    <svg id="spatial-svg" class="pitch-canvas" viewBox="0 0 525 340">
                        <rect x="0" y="0" width="525" height="340" fill="#0f3d1e"/>
                        <!-- Touchline safety buffer (2m) -->
                        <rect x="10" y="10" width="505" height="320" fill="none" stroke="#22c55e" stroke-width="1" stroke-dasharray="4 4" opacity="0.3"/>
                        <!-- Pitch Markings -->
                        <line x1="262.5" y1="0" x2="262.5" y2="340" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                        <circle cx="262.5" cy="170" r="45.75" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                        <circle cx="262.5" cy="170" r="2" fill="#ffffff"/>
                        <rect x="0" y="66" width="82.5" height="208" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                        <rect x="0" y="116" width="27.5" height="108" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                        <rect x="442.5" y="66" width="82.5" height="208" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                        <rect x="497.5" y="116" width="27.5" height="108" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>

                        <!-- Layers: Shadows, Corridors, Manifolds, Nodes -->
                        <g id="radar-shadows"></g>
                        <g id="radar-lanes"></g>
                        <g id="radar-nodes"></g>
                    </svg>
                </div>
                <div style="display:flex; gap:1rem; margin-top:0.75rem; font-size:0.75rem; color:var(--text-muted); flex-wrap:wrap;">
                    <div><span style="display:inline-block;width:10px;height:10px;background:rgba(239,68,68,0.3);border:1px solid #ef4444;vertical-align:middle;"></span> Occlusion Shadow</div>
                    <div><span style="display:inline-block;width:10px;height:10px;border:1px dashed #38bdf8;border-radius:50%;vertical-align:middle;"></span> Dispersion (±σr)</div>
                    <div><span style="display:inline-block;width:8px;height:8px;background:#c026d3;border-radius:50%;vertical-align:middle;"></span> Goalkeeper</div>
                    <div><span style="display:inline-block;width:8px;height:8px;background:#22c55e;border-radius:50%;vertical-align:middle;"></span> Dominant Target</div>
                </div>
                <div id="hull-note" style="font-size:0.72rem; color:var(--text-muted); margin-top:0.35rem;"></div>
            </div>
        </div>

        <!-- MIDDLE: MATCH VIDEO WITH LIVE TRACKING (Retained) -->
        <div class="card" style="margin-bottom: 1.5rem;">
            <div class="card-header">
                <span>3. Grounding Video & Live Telemetry Stream</span>
                <span id="time-display" style="font-family: monospace; color: var(--accent-blue); font-weight: 700;">0.00s</span>
            </div>
            <div class="viewport">
                <video id="match-video" src="{default_ep["videoSrc"]}" playsinline muted preload="auto"></video>
                <svg id="video-overlay" class="overlay-svg" viewBox="0 0 1280 720" preserveAspectRatio="xMidYMid meet"></svg>
                <div id="video-error-banner" style="display:none; position:absolute; inset:0; background:rgba(10,15,29,0.92); z-index:15; flex-direction:column; align-items:center; justify-content:center; padding:2rem; text-align:center;">
                    <div style="font-size:1.8rem; margin-bottom:0.5rem;">🎬</div>
                    <div style="font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:0.4rem;">Browser Blocked Local Video Access (file:// protocol)</div>
                    <div style="font-size:0.82rem; color:#94a3b8; max-width:460px; line-height:1.5; margin-bottom:1rem;">
                        Modern browsers restrict local <code>file://</code> video decoding across directories. To stream the video pixels with telemetry, launch a local HTTP server:
                    </div>
                    <code style="background:#1e293b; border:1px solid #334155; padding:0.5rem 1.1rem; border-radius:0.375rem; color:#38bdf8; font-family:monospace; font-size:0.85rem;">
                        python3.11 -m http.server 8000
                    </code>
                    <div style="margin-top:0.6rem; font-size:0.78rem; color:#64748b;">
                        Then open: <a href="http://localhost:8000/reports/goalkeeper_distribution_valuation.html" style="color:#38bdf8; text-decoration:underline;">http://localhost:8000/reports/goalkeeper_distribution_valuation.html</a>
                    </div>
                </div>
                <div style="position:absolute; top:12px; right:16px; background:rgba(10,15,29,0.85); backdrop-filter:blur(8px); border:1px solid rgba(56,189,248,0.35); border-radius:6px; padding:5px 12px; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; color:#38bdf8; display:flex; align-items:center; gap:8px; z-index:10; pointer-events:none; box-shadow:0 4px 12px rgba(0,0,0,0.5);">
                    <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e;"></span>
                    GK-VALUE ENGINE • LIVE TELEMETRY
                </div>
            </div>
            <div class="controls-bar">
                <button id="play-btn" class="action-btn">▶ Play</button>
                <input type="range" id="scrubber" class="scrubber" min="0" max="{default_ep["nFrames"] - 1}" value="{default_ep["decisionIdx"]}" step="1">
                <span id="frame-counter" style="font-family: monospace; font-size: 0.85rem; color: var(--text-muted);">F: {default_ep["decisionIdx"]}</span>
            </div>
            <div style="display:flex; gap:1rem; margin-top:0.5rem; font-size:0.75rem; color:var(--text-muted); flex-wrap:wrap;">
                <label><input type="checkbox" id="vid-lanes-toggle" checked style="accent-color:#22c55e;"> decision lanes</label>
                <label><input type="checkbox" id="vid-cone-toggle" checked style="accent-color:#38bdf8;"> visual cone</label>
                <label><input type="checkbox" id="vid-trail-toggle" checked style="accent-color:#e8e8e8;"> ball journey</label>
                <span><span style="display:inline-block;width:14px;height:2px;background:#ef4444;vertical-align:middle;"></span> beaten matchup</span>
                <span><span style="display:inline-block;width:14px;height:2px;background:#f59e0b;vertical-align:middle;"></span> imminent (&lt;0.5s)</span>
                <span><span style="display:inline-block;width:10px;height:10px;border:1px dashed #22c55e;border-radius:50%;vertical-align:middle;"></span> free man</span>
            </div>
        </div>


        <!-- Release-Point Option Telemetry Table -->
        <div class="card">
            <div class="card-header">
                <span id="table-time-note">{default_ep["tableTimeNote"]}</span>
            </div>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: -0.5rem; margin-bottom: 0.75rem;">
                Sequential constraint filter: eliminates trajectory-cut corridors and blindspot options to isolate viable release channels.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Outlet</th>
                        <th>Distance</th>
                        <th>Field of View</th>
                        <th>Defensive Cushion</th>
                        <th>Press Bypassed</th>
                        <th>Corridor Diagnostic</th>
                        <th>Channel Status</th>
                    </tr>
                </thead>
                <tbody id="telemetry-tbody">
                    {default_ep["tableRowsHtml"]}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const EPISODES = {episodes_json};
        let currentEpKey = "{default_ep_key}";
        let selectedTrackId = null;

        function selectTrack(tid) {{
            selectedTrackId = tid;
            const ep = EPISODES[currentEpKey];
            if (ep) renderFrontier(ep, tid);
            document.querySelectorAll('#telemetry-tbody tr').forEach(tr => {{
                if (tid !== null && tr.textContent.includes(`#${{tid}}`)) {{
                    tr.style.background = 'rgba(56, 189, 248, 0.15)';
                }} else {{
                    tr.style.background = '';
                }}
            }});
            const sc = document.getElementById('scrubber');
            if (ep && tid !== null) {{
                video.pause();
                playBtn.textContent = '▶ Play';
                video.currentTime = ep.decisionIdx / ep.fps;
                sc.value = ep.decisionIdx;
                renderFrame(ep.decisionIdx);
            }} else {{
                renderFrame(parseInt((sc && sc.value) || '0', 10));
            }}
        }}

        const video = document.getElementById('match-video');
        const scrubber = document.getElementById('scrubber');
        const playBtn = document.getElementById('play-btn');
        const timeDisplay = document.getElementById('time-display');
        const frameCounter = document.getElementById('frame-counter');
        const videoOverlay = document.getElementById('video-overlay');
        const radarLanes = document.getElementById('radar-lanes');
        const radarNodes = document.getElementById('radar-nodes');

        video.addEventListener('error', () => {{
            const banner = document.getElementById('video-error-banner');
            if (banner) banner.style.display = 'flex';
        }});
        video.addEventListener('canplay', () => {{
            const banner = document.getElementById('video-error-banner');
            if (banner) banner.style.display = 'none';
        }});

        function switchEpisode(epKey) {{
            if (!EPISODES[epKey]) return;
            currentEpKey = epKey;
            const ep = EPISODES[epKey];

            // Update switcher buttons
            document.querySelectorAll('.ep-pill-btn').forEach(btn => {{
                if (btn.getAttribute('data-ep') === epKey) {{
                    btn.style.background = '#2563eb';
                    btn.style.color = '#ffffff';
                    btn.style.border = '1px solid #38bdf8';
                }} else {{
                    btn.style.background = '#1e293b';
                    btn.style.color = '#94a3b8';
                    btn.style.border = '1px solid #334155';
                }}
            }});

            document.getElementById('header-title').textContent = ep.title;
            document.getElementById('header-subtitle').textContent = ep.subtitle;
            document.getElementById('header-badge').textContent = ep.badge;
            document.getElementById('header-meta').textContent = ep.meta;
            document.getElementById('scenario-context-tag').textContent = ep.sublabel;
            document.getElementById('table-time-note').textContent = ep.tableTimeNote;
            document.getElementById('telemetry-tbody').innerHTML = ep.tableRowsHtml;

            video.src = ep.videoSrc;
            video.load();
            scrubber.min = 0;
            scrubber.max = ep.nFrames - 1;
            scrubber.value = ep.decisionIdx;

            const seekDecision = () => {{
                video.currentTime = ep.decisionIdx / ep.fps;
                renderFrame(ep.decisionIdx);
            }};
            if (video.readyState >= 1) {{
                seekDecision();
            }} else {{
                video.addEventListener('loadedmetadata', seekDecision, {{ once: true }});
            }}
            selectedTrackId = null;
            renderFrontier(ep, null);
            renderFrame(ep.decisionIdx);
        }}

        document.querySelectorAll('.ep-pill-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                switchEpisode(btn.getAttribute('data-ep'));
            }});
        }});

        function renderFrontier(ep, activeTrackId = null) {{
            if (!ep || !ep.frontier) return;
            const f = ep.frontier;
            const svg = document.getElementById('frontier-svg');
            if (!svg) return;

            const badge = document.getElementById('frontier-status-badge');
            if (badge) {{
                if (f.is_press_collapse) {{
                    badge.style.background = 'rgba(239, 68, 68, 0.2)';
                    badge.style.color = '#ef4444';
                    badge.style.border = '1px solid #ef4444';
                    badge.textContent = '⚠ PRESS COLLAPSE · CLEARANCE MANDATED';
                }} else if (f.recommended_point) {{
                    badge.style.background = 'rgba(34, 197, 94, 0.15)';
                    badge.style.color = '#22c55e';
                    badge.style.border = '1px solid rgba(34, 197, 94, 0.3)';
                    const actName = f.recommended_point.action_type === 'PROGRESSIVE_CHANNEL' ? 'CHANNEL' : (f.recommended_point.action_type === 'SHIELDED_POCKET' ? 'SHIELD' : 'FEET');
                    badge.textContent = `OPTIMAL: #${{f.recommended_point.candidate_id}} (${{actName}})`;
                }}
            }}

            const margin = {{ left: 55, right: 25, top: 25, bottom: 45 }};
            const width = 500 - margin.left - margin.right;
            const height = 320 - margin.top - margin.bottom;

            const xMin = -2.0, xMax = 3.0;
            const yMin = -0.35, yMax = 0.25;

            const scaleX = (x) => margin.left + Math.max(0, Math.min(width, ((x - xMin) / (xMax - xMin)) * width));
            const scaleY = (y) => margin.top + Math.max(0, Math.min(height, ((yMax - y) / (yMax - yMin)) * height));

            const x0 = scaleX(0.0);
            const y0 = scaleY(0.0);
            const xSat = scaleX(f.cushion_saturation_cap_s || 2.5);

            let html = '';

            // Quadrant Background Shading
            // 1. Dominant (Top-Right): X > 0, Y > 0
            html += `<rect x="${{x0}}" y="${{margin.top}}" width="${{scaleX(xMax) - x0}}" height="${{y0 - margin.top}}" fill="rgba(34, 197, 94, 0.08)"/>`;
            html += `<text x="${{x0 + 10}}" y="${{margin.top + 16}}" fill="#22c55e" font-size="8.5" font-weight="700" letter-spacing="0.5px">DOMINANT (SAFE &amp; PROGRESSIVE)</text>`;

            // 2. Hospital Pass / High Risk (Top-Left): X < 0, Y > 0
            html += `<rect x="${{margin.left}}" y="${{margin.top}}" width="${{x0 - margin.left}}" height="${{y0 - margin.top}}" fill="rgba(249, 115, 22, 0.08)"/>`;
            html += `<text x="${{margin.left + 8}}" y="${{margin.top + 16}}" fill="#f97316" font-size="8.5" font-weight="700" letter-spacing="0.5px">CONTROL TRAP (RECEIPT UNDER PRESS)</text>`;

            // 3. Safe Stagnation / Recycle (Bottom-Right): X > 0, Y < 0
            html += `<rect x="${{x0}}" y="${{y0}}" width="${{scaleX(xMax) - x0}}" height="${{scaleY(yMin) - y0}}" fill="rgba(250, 204, 21, 0.04)"/>`;
            html += `<text x="${{x0 + 10}}" y="${{scaleY(yMin) - 10}}" fill="#eab308" font-size="8.5" font-weight="700" letter-spacing="0.5px">SAFE STAGNATION (BACKPASS)</text>`;

            // 4. Catastrophic / Beaten (Bottom-Left): X < 0, Y < 0
            html += `<rect x="${{margin.left}}" y="${{y0}}" width="${{x0 - margin.left}}" height="${{scaleY(yMin) - y0}}" fill="rgba(239, 68, 68, 0.16)"/>`;
            html += `<text x="${{margin.left + 8}}" y="${{scaleY(yMin) - 10}}" fill="#ef4444" font-size="8.5" font-weight="700" letter-spacing="0.5px">BEATEN / CATASTROPHIC</text>`;

            // Gridlines & Axis Values
            for (let xVal = -1.5; xVal <= 2.5; xVal += 0.5) {{
                if (Math.abs(xVal) < 0.01) continue;
                const gx = scaleX(xVal);
                html += `<line x1="${{gx}}" y1="${{margin.top}}" x2="${{gx}}" y2="${{scaleY(yMin)}}" stroke="#1e293b" stroke-width="1" stroke-dasharray="2 2"/>`;
                html += `<text x="${{gx}}" y="${{scaleY(yMin) + 14}}" fill="#64748b" font-size="8" text-anchor="middle">${{xVal > 0 ? '+' : ''}}${{xVal.toFixed(1)}}s</text>`;
            }}
            for (let yVal = -0.3; yVal <= 0.2; yVal += 0.1) {{
                if (Math.abs(yVal) < 0.01) continue;
                const gy = scaleY(yVal);
                html += `<line x1="${{margin.left}}" y1="${{gy}}" x2="${{scaleX(xMax)}}" stroke="#1e293b" stroke-width="1" stroke-dasharray="2 2"/>`;
                html += `<text x="${{margin.left - 6}}" y="${{gy + 3}}" fill="#64748b" font-size="8" text-anchor="end">${{yVal > 0 ? '+' : ''}}${{yVal.toFixed(2)}}</text>`;
            }}

            // Zero axes
            html += `<line x1="${{x0}}" y1="${{margin.top}}" x2="${{x0}}" y2="${{scaleY(yMin)}}" stroke="#475569" stroke-width="1.5"/>`;
            html += `<line x1="${{margin.left}}" y1="${{y0}}" x2="${{scaleX(xMax)}}" stroke="#475569" stroke-width="1.5"/>`;
            html += `<text x="${{x0}}" y="${{scaleY(yMin) + 14}}" fill="#94a3b8" font-size="8.5" font-weight="700" text-anchor="middle">0.0s</text>`;
            html += `<text x="${{margin.left - 6}}" y="${{y0 + 3}}" fill="#94a3b8" font-size="8.5" font-weight="700" text-anchor="end">0.00</text>`;

            // Saturation line tau_safe
            html += `<line x1="${{xSat}}" y1="${{margin.top}}" x2="${{xSat}}" y2="${{scaleY(yMin)}}" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="3 3" opacity="0.6"/>`;
            html += `<text x="${{xSat}}" y="${{margin.top + 8}}" fill="#38bdf8" font-size="7.5" font-weight="600" text-anchor="middle" opacity="0.8">τ_safe cap (2.5s)</text>`;

            // Axis titles
            html += `<text x="${{margin.left + width / 2}}" y="${{320 - 10}}" fill="#94a3b8" font-size="9.5" font-weight="600" text-anchor="middle">Defensive Arrival Cushion Δt = t_press - t_receipt (s)</text>`;
            html += `<text x="14" y="${{margin.top + height / 2}}" fill="#94a3b8" font-size="9.5" font-weight="600" text-anchor="middle" transform="rotate(-90 14 ${{margin.top + height / 2}})">Net Expected Value ΔxG</text>`;

            // Pareto Frontier Curve
            if (f.pareto_frontier && f.pareto_frontier.length > 0) {{
                const polyPoints = f.pareto_frontier.map(p => `${{scaleX(p.effective_cushion_s)}},${{scaleY(p.net_ev)}}`).join(' ');
                html += `<polyline points="${{polyPoints}}" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-dasharray="4 3" opacity="0.85"/>`;
            }}

            // Clearance baseline marker
            if (f.clearance_baseline) {{
                const cbX = scaleX(f.clearance_baseline.effective_cushion_s);
                const cbY = scaleY(f.clearance_baseline.net_ev);
                html += `<polygon points="${{cbX}},${{cbY - 6}} ${{cbX + 6}},${{cbY}} ${{cbX}},${{cbY + 6}} ${{cbX - 6}},${{cbY}}" fill="#64748b" stroke="#ffffff" stroke-width="1.2"/>`;
                html += `<text x="${{cbX + 8}}" y="${{cbY + 3}}" fill="#94a3b8" font-size="7.5" font-weight="600">CLEARANCE</text>`;
            }}

            // Press Collapse Alert Overlay
            if (f.is_press_collapse) {{
                html += `<rect x="65" y="32" width="370" height="26" fill="rgba(239, 68, 68, 0.92)" rx="4"/>`;
                html += `<text x="250" y="49" text-anchor="middle" fill="#ffffff" font-size="10.5" font-weight="bold">⚠ PRESS COLLAPSE DETECTED: CLEARANCE MANDATED</text>`;
            }}

            // Candidate Points
            const pts = f.team_points || [];
            pts.forEach(p => {{
                const px = scaleX(p.effective_cushion_s);
                const py = scaleY(p.net_ev);
                const isRec = f.recommended_point && f.recommended_point.candidate_id === p.candidate_id;
                const isActive = activeTrackId !== null && p.candidate_id === activeTrackId;

                let col = '#38bdf8';
                if (p.is_occluded) col = '#64748b';
                else if (p.path_status === 'FEASIBLE_OPEN') col = '#22c55e';
                else if (p.path_status === 'CONTROL_TRAP') col = '#f97316';
                else if (p.path_status === 'PRESS_TRAP') col = '#ef4444';

                // Glowing ring for Pareto or Active
                if (p.is_pareto_optimal || isActive) {{
                    html += `<circle cx="${{px}}" cy="${{py}}" r="${{isActive ? 13 : 10}}" fill="none" stroke="${{col}}" stroke-width="${{isActive ? 2.5 : 1.5}}" stroke-dasharray="${{p.is_pareto_optimal ? 'none' : '2 2'}}" opacity="0.85"/>`;
                }}

                html += `<circle cx="${{px}}" cy="${{py}}" r="${{isRec ? 7 : 5.5}}" fill="${{col}}" stroke="#ffffff" stroke-width="1.5" style="cursor:pointer;" data-track="${{p.candidate_id}}"/>`;

                // Action and Player Label
                const actionShort = p.action_type === 'PROGRESSIVE_CHANNEL' ? 'CH' : (p.action_type === 'SHIELDED_POCKET' ? 'SH' : 'FT');
                const star = isRec ? '★ ' : '';
                html += `<text x="${{px + 8}}" y="${{py - 3}}" fill="${{isRec ? '#facc15' : '#f8fafc'}}" font-size="9" font-weight="bold">${{star}}#${{p.candidate_id}} (${{actionShort}})</text>`;
            }});

            svg.innerHTML = html;

            // Attach hover/click events
            svg.querySelectorAll('circle[data-track]').forEach(el => {{
                const tid = parseInt(el.getAttribute('data-track'));
                const pt = pts.find(p => p.candidate_id === tid);
                if (!pt) return;

                el.addEventListener('mouseenter', (ev) => {{
                    const tt = document.getElementById('frontier-tooltip');
                    if (!tt) return;
                    const actionName = pt.action_type === 'PROGRESSIVE_CHANNEL' ? 'Progressive Channel Lead' : (pt.action_type === 'SHIELDED_POCKET' ? 'Shielded Pocket Lead' : 'Centroid Feet Lead');
                    tt.innerHTML = `
                        <div style="font-weight:bold; color:#38bdf8; margin-bottom:2px;">#${{pt.candidate_id}} · ${{actionName}}</div>
                        <div>Cushion: <strong style="color:${{pt.effective_cushion_s >= 0.6 ? '#22c55e' : '#ef4444'}};">${{pt.raw_cushion_s > 0 ? '+' : ''}}${{pt.raw_cushion_s.toFixed(2)}}s</strong> (eff: ${{pt.effective_cushion_s.toFixed(2)}}s)</div>
                        <div>Net EV: <strong style="color:${{pt.net_ev > 0 ? '#22c55e' : '#ef4444'}};">${{pt.net_ev > 0 ? '+' : ''}}${{pt.net_ev.toFixed(4)}} xG</strong></div>
                        <div>Dispersion: ±${{pt.dispersion_sigma_m.toFixed(1)}}m | xP: ${{(pt.xp_completion_prob * 100).toFixed(0)}}%</div>
                        <div style="font-size:0.7rem; color:#94a3b8; margin-top:2px;">Status: <strong>${{pt.path_status}}</strong> ${{pt.is_pareto_optimal ? '· PARETO OPTIMAL' : ''}}</div>
                    `;
                    tt.style.display = 'block';
                    const rect = svg.getBoundingClientRect();
                    tt.style.left = `${{ev.clientX - rect.left + 12}}px`;
                    tt.style.top = `${{ev.clientY - rect.top - 20}}px`;
                }});
                el.addEventListener('mouseleave', () => {{
                    const tt = document.getElementById('frontier-tooltip');
                    if (tt) tt.style.display = 'none';
                }});
                el.addEventListener('click', () => {{
                    selectTrack(tid);
                }});
            }});
            svg.onclick = (ev) => {{
                if (ev.target === svg) selectTrack(null);
            }};
        }}

        function renderFrame(frameIdx) {{
            const ep = EPISODES[currentEpKey];
            if (!ep || !ep.trackingData) return;
            const data = ep.trackingData[frameIdx];
            if (!data) return;

            const timeS = frameIdx / ep.fps;
            const totalS = ep.nFrames / ep.fps;
            timeDisplay.textContent = `${{timeS.toFixed(2)}}s / ${{totalS.toFixed(2)}}s`;
            frameCounter.textContent = `F: ${{String(frameIdx).padStart(3, '0')}}/${{ep.nFrames}} @${{ep.fps}}fps`;

            // 1. Video Overlay
            let vSvg = '';
            for (const e of data.entities) {{
                const [x1, y1, x2, y2] = e.bbox;
                const w = x2 - x1;
                const h = y2 - y1;
                if (e.class_name === 'sports ball') {{
                    vSvg += `<circle cx="${{e.screen_uv[0]}}" cy="${{e.screen_uv[1]}}" r="7" fill="#eab308" stroke="#ffffff" stroke-width="1.5"/>
                             <text x="${{x1-10}}" y="${{Math.max(12, y1-4)}}" fill="#eab308" font-size="10" font-weight="bold">⚽ BALL</text>`;
                }} else {{
                    vSvg += `<rect x="${{x1}}" y="${{y1}}" width="${{w}}" height="${{h}}" fill="none" stroke="${{e.color}}" stroke-width="2" rx="3"/>
                             <rect x="${{x1}}" y="${{Math.max(0, y1-16)}}" width="65" height="16" fill="${{e.color}}" rx="2"/>
                             <text x="${{x1+3}}" y="${{Math.max(11, y1-4)}}" fill="#0f172a" font-size="10" font-weight="bold">#${{e.track_id}}</text>`;
                }}
            }}

            // Ball journey trail
            const trailToggle = document.getElementById('vid-trail-toggle');
            if (!trailToggle || trailToggle.checked) {{
                const trailPts = [];
                for (let f = 0; f <= frameIdx; f++) {{
                    const d = ep.trackingData[f];
                    if (!d) continue;
                    const b = d.entities.find(e => e.class_name === 'sports ball');
                    if (b && b.screen_uv) trailPts.push(b.screen_uv);
                }}
                if (trailPts.length > 1) {{
                    const thirds = Math.max(1, Math.floor(trailPts.length / 3));
                    const segs = [
                        {{pts: trailPts.slice(0, thirds * 2), op: 0.10}},
                        {{pts: trailPts.slice(thirds * 2, thirds * 3), op: 0.20}},
                        {{pts: trailPts.slice(thirds * 3), op: 0.35}},
                    ];
                    for (const s of segs) {{
                        if (s.pts.length < 2) continue;
                        const d = 'M ' + s.pts.map(p => `${{p[0].toFixed(1)}} ${{p[1].toFixed(1)}}`).join(' L ');
                        vSvg += `<path d="${{d}}" fill="none" stroke="#f8fafc" stroke-width="2" stroke-linecap="round" opacity="${{s.op}}"/>`;
                    }}
                }}
                if (trailPts.length > 2) {{
                    const last = trailPts[trailPts.length - 1];
                    const prev = trailPts[trailPts.length - 3];
                    const dx = last[0] - prev[0], dy = last[1] - prev[1];
                    const tipX = last[0] + dx * 3, tipY = last[1] + dy * 3;
                        vSvg += `<line x1="${{last[0]}}" y1="${{last[1]}}" x2="${{tipX}}" y2="${{tipY}}" stroke="#f8fafc" stroke-width="2" stroke-dasharray="5 4" opacity="0.4"/>`;
                }}
            }}

            // Decision lanes on video
            const vidLanesToggle = document.getElementById('vid-lanes-toggle');
            const vidConeToggle = document.getElementById('vid-cone-toggle');
            const decisionFramesSet = new Set(ep.decisionFrames);
            const hasBall = decisionFramesSet.has(frameIdx);
            if (hasBall) {{
                vSvg += `<rect x="12" y="12" width="220" height="22" fill="#0a0f1d" opacity="0.85" rx="11"/>`;
                vSvg += `<text x="24" y="27" fill="#22c55e" font-size="12" font-weight="bold">DECISION · keeper on ball</text>`;
            }}
            if (hasBall && ep.decisionPx && ep.decisionPx.lanes && ep.decisionPx.lanes.length > 0) {{
                const [ax, ay] = ep.decisionPx.anchor_px;
                if (vidConeToggle && vidConeToggle.checked && ep.decisionPx.cone_px && ep.decisionPx.cone_px.length > 2) {{
                    if (ep.decisionPx.scan_cones_px) {{
                        for (const sc of ep.decisionPx.scan_cones_px) {{
                            if (!sc.cone_px || sc.cone_px.length <= 2) continue;
                            const sd = 'M ' + sc.cone_px.map(p => `${{p[0]}} ${{p[1]}}`).join(' L ') + ' Z';
                            vSvg += `<path d="${{sd}}" fill="rgba(45,212,191,0.05)" stroke="rgba(45,212,191,0.35)" stroke-width="1" stroke-dasharray="3 3"/>`;
                        }}
                    }}
                    const cd = 'M ' + ep.decisionPx.cone_px.map(p => `${{p[0]}} ${{p[1]}}`).join(' L ') + ' Z';
                    vSvg += `<path d="${{cd}}" fill="rgba(56,189,248,0.10)" stroke="rgba(56,189,248,0.5)" stroke-width="1.5" stroke-dasharray="6 3"/>`;
                    const [fx, fy] = ep.decisionPx.facing_tip_px;
                    vSvg += `<line x1="${{ax}}" y1="${{ay}}" x2="${{fx}}" y2="${{fy}}" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.7"/>`;
                }}
                if (!vidLanesToggle || vidLanesToggle.checked) {{
                    const ke = (ep.keeperTrackId !== null) ? data.entities.find(e => e.track_id === ep.keeperTrackId) : null;
                    const lax = (ke && ke.screen_uv) ? ke.screen_uv[0] : ax;
                    const lay = (ke && ke.screen_uv) ? ke.screen_uv[1] : ay;
                    for (const lane of ep.decisionPx.lanes) {{
                        const [gx, gy] = lane.target_px;
                        const re = data.entities.find(e => e.track_id === lane.track_id);
                        const tx = (re && re.screen_uv) ? re.screen_uv[0] : gx;
                        const ty = (re && re.screen_uv) ? re.screen_uv[1] : gy;
                        const isSel = selectedTrackId !== null && lane.track_id === selectedTrackId;
                        const tagged = lane.is_optimal || lane.is_bad || lane.is_scan_known || isSel;
                        const w = isSel ? 5.0 : (lane.is_optimal ? 4.0 : ((lane.is_bad || lane.is_scan_known) ? 2.5 : 1.5));
                        const dash = (lane.is_optimal || lane.is_scan_known) ? 'none' : (lane.is_bad ? '6 4' : '2 3');
                        const op = (isSel || lane.is_optimal) ? 1.0 : ((lane.is_bad || lane.is_scan_known) ? 0.9 : 0.45);
                        const glow = (isSel || lane.is_optimal) ? ' style="filter:drop-shadow(0 0 4px #22c55e)"' : '';
                        vSvg += `<line x1="${{lax}}" y1="${{lay}}" x2="${{tx}}" y2="${{ty}}" stroke="${{lane.grade_color}}" stroke-width="${{w}}" stroke-dasharray="${{dash}}" opacity="${{op}}"${{glow}}/>`;
                        vSvg += `<circle cx="${{tx}}" cy="${{ty}}" r="${{lane.is_optimal ? 11 : 8}}" fill="none" stroke="${{lane.grade_color}}" stroke-width="2" opacity="${{op}}"/>`;
                        vSvg += `<circle cx="${{gx}}" cy="${{gy}}" r="4" fill="none" stroke="${{lane.grade_color}}" stroke-width="1" stroke-dasharray="2 2" opacity="0.5"><title>Evaluated lead target</title></circle>`;
                        vSvg += `<line x1="${{lax}}" y1="${{lay}}" x2="${{tx}}" y2="${{ty}}" stroke="#ffffff" stroke-opacity="0" stroke-width="20" class="lane-hit" data-track="${{lane.track_id}}"><title>#${{lane.track_id}} ${{lane.visual_label}}</title></line>`;
                        if (tagged) {{
                            const actionPfx = (lane.action_type === 'PROGRESSIVE_CHANNEL') ? 'CHANNEL ' : ((lane.action_type === 'SHIELDED_POCKET') ? 'SHIELDED ' : '');
                            const tag = (lane.path_rank === 1 && lane.is_optimal) ? `${{actionPfx}}LEAD #${{lane.track_id}}` : `#${{lane.track_id}} ${{lane.visual_label}}`;
                            const ty2 = Math.max(14, ty - 14);
                            vSvg += `<rect x="${{tx + 10}}" y="${{ty2 - 11}}" width="${{Math.min(230, tag.length * 6.4 + 12)}}" height="16" fill="#0a0f1d" opacity="0.85" rx="3"/>`;
                            vSvg += `<text x="${{tx + 16}}" y="${{ty2 + 1}}" fill="${{lane.grade_color}}" font-size="10" font-weight="bold">${{tag}}</text>`;
                        }}
                    }}
                }}
                if (ep.decisionPx.matchups) {{
                    const [bax, bay] = ep.decisionPx.anchor_px;
                    for (const mu of ep.decisionPx.matchups) {{
                        if (mu.is_ball_press) {{
                            const bpe = data.entities.find(e => e.track_id === mu.presser_track_id);
                            if (!bpe || !bpe.screen_uv) continue;
                            const bTitle = mu.task_label ? `#${{mu.presser_track_id}} ${{mu.task_label}}` : `BALL PRESS #${{mu.presser_track_id}}`;
                            vSvg += `<line x1="${{bax}}" y1="${{bay}}" x2="${{bpe.screen_uv[0]}}" y2="${{bpe.screen_uv[1]}}" stroke="#f0abfc" stroke-width="2" stroke-dasharray="6 3" opacity="0.8"><title>${{bTitle}}</title></line>`;
                            continue;
                        }}
                        const pe = data.entities.find(e => e.track_id === mu.presser_track_id);
                        const re = data.entities.find(e => e.track_id === mu.receiver_track_id);
                        if (!pe || !re || !pe.screen_uv || !re.screen_uv) continue;
                        const imminent = !mu.beaten && mu.margin_s < 0.5;
                        const col = mu.beaten ? '#ef4444' : (imminent ? '#f59e0b' : '#e8e8e8');
                        const wd = mu.beaten ? 3.0 : (imminent ? 2.2 : 1.0);
                        const dp = (mu.beaten || imminent) ? 'none' : '3 3';
                        const op = mu.beaten ? 0.9 : (imminent ? 0.85 : 0.35);
                        const glow = (mu.beaten || imminent) ? ' style="filter:drop-shadow(0 0 4px #ef4444)"' : '';
                        const taskTitle = mu.task_label ? `#${{mu.presser_track_id}} ${{mu.task_label}}` : `#${{mu.presser_track_id}}`;
                        vSvg += `<line x1="${{pe.screen_uv[0]}}" y1="${{pe.screen_uv[1]}}" x2="${{re.screen_uv[0]}}" y2="${{re.screen_uv[1]}}" stroke="${{col}}" stroke-width="${{wd}}" stroke-dasharray="${{dp}}" opacity="${{op}}"${{glow}}><title>${{taskTitle}}</title></line>`;
                        if (mu.free_man) {{
                            vSvg += `<circle cx="${{re.screen_uv[0]}}" cy="${{re.screen_uv[1]}}" r="13" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-dasharray="2 2" opacity="0.8"><title>Free man (margin ${{mu.margin_s}}s)</title></circle>`;
                        }}
                    }}
                }}
            }}
            videoOverlay.innerHTML = vSvg;
            videoOverlay.querySelectorAll('.lane-hit').forEach(el => {{
                el.addEventListener('click', () => {{
                    selectTrack(parseInt(el.getAttribute('data-track'), 10));
                }});
            }});

            // 2. 2D Pitch Radar
            let lSvg = '';
            const isKeeper = (e) => /gk|keeper/i.test(e.team_label) || e.color === '#c026d3' || e.color === '#facc15';
            let gk = null;
            if (ep.keeperTrackId !== null) {{
                gk = data.entities.find(e => e.track_id === ep.keeperTrackId) || null;
            }}
            if (!gk) {{
                const keepers = data.entities.filter(isKeeper);
                if (keepers.length > 0) {{
                    gk = keepers.reduce((best, p) => {{
                        const d = (p.pitch_xy[0] - ep.decisionGk[0]) ** 2 + (p.pitch_xy[1] - ep.decisionGk[1]) ** 2;
                        const bd = (best.pitch_xy[0] - ep.decisionGk[0]) ** 2 + (best.pitch_xy[1] - ep.decisionGk[1]) ** 2;
                        return d < bd ? p : best;
                    }}, keepers[0]);
                }}
            }}
            const anchor = gk ? gk.pitch_xy : ep.decisionGk;
            const lanesToggle = document.getElementById('lanes-toggle');
            const showLanes = (!lanesToggle || lanesToggle.checked) && decisionFramesSet.has(frameIdx);

            // 0. Opponent Occlusion Shadows
            let sSvg = '';
            if (showLanes && anchor && ep.opponentsPitch) {{
                for (const op of ep.opponentsPitch) {{
                    const dx = op[0] - anchor[0];
                    const dy = op[1] - anchor[1];
                    const dist = Math.hypot(dx, dy);
                    if (dist >= 2.0 && dist <= 48.0) {{
                        const theta = Math.atan2(dy, dx);
                        const rEff = 1.2;
                        const alpha = Math.asin(Math.min(0.90, rEff / dist));
                        const dFar = dist + 35.0;
                        const t1 = theta - alpha, t2 = theta + alpha;

                        const p1x = (op[0] - rEff * Math.sin(theta)) * 5.0;
                        const p1y = (68.0 - (op[1] + rEff * Math.cos(theta))) * 5.0;
                        const p2x = (op[0] + rEff * Math.sin(theta)) * 5.0;
                        const p2y = (68.0 - (op[1] - rEff * Math.cos(theta))) * 5.0;
                        const f1x = (anchor[0] + dFar * Math.cos(t1)) * 5.0;
                        const f1y = (68.0 - (anchor[1] + dFar * Math.sin(t1))) * 5.0;
                        const f2x = (anchor[0] + dFar * Math.cos(t2)) * 5.0;
                        const f2y = (68.0 - (anchor[1] + dFar * Math.sin(t2))) * 5.0;

                        sSvg += `<polygon points="${{p1x}},${{p1y}} ${{f1x}},${{f1y}} ${{f2x}},${{f2y}} ${{p2x}},${{p2y}}" fill="rgba(239, 68, 68, 0.16)" stroke="rgba(239, 68, 68, 0.35)" stroke-width="1" stroke-dasharray="3 3"/>`;
                    }}
                }}
            }}
            const shadowLayer = document.getElementById('radar-shadows');
            if (shadowLayer) shadowLayer.innerHTML = sSvg;

            if (showLanes && anchor && ep.evalData && ep.evalData.length > 0) {{
                const anchorX = Math.max(0, Math.min(105, anchor[0])) * 5.0;
                const anchorY = (68.0 - Math.max(0, Math.min(68, anchor[1]))) * 5.0;
                const facingSvgAngle = -((ep.gazeRad !== undefined) ? ep.gazeRad : ep.facingRad);
                const torsoSvgAngle = -(ep.facingRad !== undefined ? ep.facingRad : ep.gazeRad);
                const coneRadius = ep.coneRadiusM * 5.0;
                const halfFov = (70.0 * Math.PI) / 180.0;

                // 1. Swept Scan Memory Envelope: union of per-frame gaze cones.
                // Backend ships merged pitch-frame intervals (scanEnvelope); fall back
                // to min-max over scannedAngles only for stale cached payloads.
                const scanIntervals = (ep.scanEnvelope && ep.scanEnvelope.length > 0)
                    ? ep.scanEnvelope
                    : ((ep.scannedAngles && ep.scannedAngles.length > 1)
                        ? [[Math.min(...ep.scannedAngles) - halfFov, Math.max(...ep.scannedAngles) + halfFov]]
                        : []);
                if (scanIntervals.length > 0) {{
                    for (const [sPitch, ePitch] of scanIntervals) {{
                        const sA = -ePitch, eA = -sPitch;
                        const s1x = anchorX + (coneRadius * 1.08) * Math.cos(sA);
                        const s1y = anchorY + (coneRadius * 1.08) * Math.sin(sA);
                        const s2x = anchorX + (coneRadius * 1.08) * Math.cos(eA);
                        const s2y = anchorY + (coneRadius * 1.08) * Math.sin(eA);
                        const largeArc = (eA - sA) > Math.PI ? 1 : 0;
                        lSvg += `<path d="M ${{anchorX}} ${{anchorY}} L ${{s1x}} ${{s1y}} A ${{coneRadius * 1.08}} ${{coneRadius * 1.08}} 0 ${{largeArc}} 1 ${{s2x}} ${{s2y}} Z" fill="rgba(45, 212, 191, 0.07)" stroke="rgba(45, 212, 191, 0.45)" stroke-width="1.2" stroke-dasharray="3 3"/>`;
                    }}
                    lSvg += `<text x="${{anchorX - 35}}" y="${{anchorY - coneRadius - 6}}" fill="#2dd4bf" font-size="8" font-weight="600" letter-spacing="0.5px">SWEPT SCAN MEMORY ENVELOPE</text>`;
                }}

                // 2. Active Instantaneous Gaze Cone
                const a1 = facingSvgAngle - halfFov;
                const a2 = facingSvgAngle + halfFov;
                const p1x = anchorX + coneRadius * Math.cos(a1);
                const p1y = anchorY + coneRadius * Math.sin(a1);
                const p2x = anchorX + coneRadius * Math.cos(a2);
                const p2y = anchorY + coneRadius * Math.sin(a2);

                lSvg += `<path d="M ${{anchorX}} ${{anchorY}} L ${{p1x}} ${{p1y}} A ${{coneRadius}} ${{coneRadius}} 0 0 1 ${{p2x}} ${{p2y}} Z" fill="rgba(56, 189, 248, 0.12)" stroke="rgba(56, 189, 248, 0.45)" stroke-width="1.5" stroke-dasharray="4 2"/>`;
                lSvg += `<text x="${{anchorX - 30}}" y="${{anchorY + coneRadius + 14}}" fill="#38bdf8" font-size="9" font-weight="bold">140° ACTIVE GAZE (12m)</text>`;

                // 3. Torso Alignment Vector (if decoupled from gaze)
                if (Math.abs(facingSvgAngle - torsoSvgAngle) > 0.08) {{
                    const tvLen = coneRadius * 0.55;
                    const tvx = anchorX + tvLen * Math.cos(torsoSvgAngle);
                    const tvy = anchorY + tvLen * Math.sin(torsoSvgAngle);
                    lSvg += `<line x1="${{anchorX}}" y1="${{anchorY}}" x2="${{tvx}}" y2="${{tvy}}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="3 2" opacity="0.85"/>
                             <circle cx="${{tvx}}" cy="${{tvy}}" r="3" fill="#f59e0b"/>
                             <text x="${{tvx + 4}}" y="${{tvy + 3}}" fill="#f59e0b" font-size="8" font-weight="bold">TORSO</text>`;
                }}

                for (const opt of ep.evalData) {{
                    const tx = Math.max(0, Math.min(105, opt.target_pos[0])) * 5.0;
                    const ty = (68.0 - Math.max(0, Math.min(68, opt.target_pos[1]))) * 5.0;
                    const isFeasible = (opt.in_visual_cone || opt.scan_known || opt.path_status === "DISGUISED_OUTLET" || opt.path_status === "SCANNED_OPEN") && !opt.is_los_blocked && !opt.is_approach_cut && !opt.is_inertia_locked;
                    const strokeDash = isFeasible ? "none" : "4 3";
                    const strokeOpacity = isFeasible ? "0.9" : "0.35";

                    // Lead vector from receiver body
                    const re = data.entities.find(e => e.track_id === opt.track_id);
                    if (re && re.pitch_xy) {{
                        const rx = Math.max(0, Math.min(105, re.pitch_xy[0])) * 5.0;
                        const ry = (68.0 - Math.max(0, Math.min(68, re.pitch_xy[1]))) * 5.0;
                        lSvg += `<line x1="${{rx}}" y1="${{ry}}" x2="${{tx}}" y2="${{ty}}" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="2 2" opacity="0.8"/>`;
                    }}

                    // Execution dispersion ring
                    const sig = opt.dispersion_sigma_m || (opt.distance_m * 0.07) || 1.8;
                    lSvg += `<circle cx="${{tx}}" cy="${{ty}}" r="${{sig * 5.0}}" fill="rgba(56, 189, 248, 0.07)" stroke="rgba(56, 189, 248, 0.45)" stroke-width="1.2" stroke-dasharray="3 3"><title>Delivery dispersion ±${{sig.toFixed(1)}}m</title></circle>`;

                    // Pass corridor
                    lSvg += `<line x1="${{anchorX}}" y1="${{anchorY}}" x2="${{tx}}" y2="${{ty}}" stroke="${{opt.grade_color}}" stroke-width="2.2" stroke-dasharray="${{strokeDash}}" opacity="${{strokeOpacity}}"/>
                             <circle cx="${{tx}}" cy="${{ty}}" r="${{opt.is_optimal ? 9 : 7}}" fill="${{opt.grade_color}}" stroke="#ffffff" stroke-width="1.5"/>`;

                    // Target label
                    const actShort = opt.action_type === 'PROGRESSIVE_CHANNEL' ? 'CHANNEL' : (opt.action_type === 'SHIELDED_POCKET' ? 'SHIELD' : 'FEET');
                    lSvg += `<text x="${{tx + 12}}" y="${{ty + 3}}" fill="#f8fafc" font-size="8.5" font-weight="bold">#${{opt.track_id}} ${{actShort}}</text>`;
                }}
                lSvg += `<circle cx="${{anchorX}}" cy="${{anchorY}}" r="5" fill="none" stroke="#e879f9" stroke-width="1.2" stroke-dasharray="2 2"><title>Keeper anchor</title></circle>`;
            }}

            // 4. Live gaze needle: last measured gaze at or before this scrub frame.
            if (anchor && ep.scanFrames) {{
                let needleRad = null;
                for (let f = frameIdx; f >= 0; f--) {{
                    const v = ep.scanFrames[String(f)];
                    if (v !== undefined) {{ needleRad = v; break; }}
                }}
                if (needleRad !== null) {{
                    const nax = Math.max(0, Math.min(105, anchor[0])) * 5.0;
                    const nay = (68.0 - Math.max(0, Math.min(68, anchor[1]))) * 5.0;
                    const nA = -needleRad;
                    const nLen = ep.coneRadiusM * 5.0 * 0.85;
                    const nx = nax + nLen * Math.cos(nA);
                    const ny = nay + nLen * Math.sin(nA);
                    lSvg += `<line x1="${{nax}}" y1="${{nay}}" x2="${{nx}}" y2="${{ny}}" stroke="#38bdf8" stroke-width="2" opacity="0.9"><title>Measured gaze at/before this frame</title></line>
                             <circle cx="${{nx}}" cy="${{ny}}" r="3" fill="#38bdf8" opacity="0.9"/>`;
                }}
            }}
            radarLanes.innerHTML = lSvg;

            let rSvg = '';
            let invalidCount = 0;
            for (const e of data.entities) {{
                const valid = e.pitch_valid !== false;
                if (!valid) {{ invalidCount += 1; continue; }}
                const [px, py] = e.pitch_xy;
                const cx = Math.max(0, Math.min(105, px)) * 5.0;
                const cy = (68.0 - Math.max(0, Math.min(68, py))) * 5.0;
                const strokeColor = '#000000';

                if (e.class_name === 'sports ball') {{
                    rSvg += `<g>
                                <circle cx="${{cx}}" cy="${{cy}}" r="4.0" fill="#facc15" stroke="#000000" stroke-width="1.2"/>
                             </g>`;
                }} else {{
                    rSvg += `<g>
                                <circle cx="${{cx}}" cy="${{cy}}" r="7" fill="${{e.color}}" stroke="${{strokeColor}}" stroke-width="1.8"/>
                                <text x="${{cx}}" y="${{cy-9}}" fill="#f8fafc" font-size="9" font-weight="bold" text-anchor="middle">#${{e.track_id}}</text>
                             </g>`;
                }}
            }}
            radarNodes.innerHTML = rSvg;
            const hullNote = document.getElementById('hull-note');
            if (hullNote) hullNote.textContent = invalidCount > 0 ? `${{invalidCount}} outside calibration hull — hidden on radar, visible in video, excluded from scoring.` : 'All points inside calibration hull.';
        }}

        function frameFromVideo() {{
            const ep = EPISODES[currentEpKey];
            if (!ep) return 0;
            return Math.min(ep.nFrames - 1, Math.max(0, Math.round(video.currentTime * ep.fps)));
        }}
        function syncFromVideo() {{
            const frameIdx = frameFromVideo();
            if (String(scrubber.value) !== String(frameIdx)) scrubber.value = frameIdx;
            renderFrame(frameIdx);
        }}

        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {{
            const tick = () => {{ if (!video.paused) syncFromVideo(); video.requestVideoFrameCallback(tick); }};
            video.requestVideoFrameCallback(tick);
        }} else {{
            setInterval(() => {{ if (!video.paused) syncFromVideo(); }}, 100);
        }}

        scrubber.addEventListener('input', (e) => {{
            const ep = EPISODES[currentEpKey];
            if (!ep) return;
            const frameIdx = parseInt(e.target.value);
            video.pause();
            playBtn.textContent = '▶ Play';
            video.currentTime = frameIdx / ep.fps;
            renderFrame(frameIdx);
        }});

        const lanesToggle = document.getElementById('lanes-toggle');
        if (lanesToggle) lanesToggle.addEventListener('change', () => {{
            renderFrame(parseInt(scrubber.value) || 0);
        }});
        const physicsToggle = document.getElementById('physics-toggle');
        if (physicsToggle) physicsToggle.addEventListener('click', (ev) => {{
            ev.preventDefault();
            const card = document.getElementById('spatial-card');
            if (!card) return;
            const hidden = card.style.display === 'none';
            card.style.display = hidden ? '' : 'none';
            physicsToggle.textContent = hidden ? 'hide 2D physics' : '2D physics (internal)';
        }});
        for (const id of ['vid-lanes-toggle', 'vid-cone-toggle', 'vid-trail-toggle']) {{
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => {{
                renderFrame(parseInt(scrubber.value) || 0);
            }});
        }}

        playBtn.addEventListener('click', () => {{
            if (video.paused) {{
                video.play();
                playBtn.textContent = '⏸ Pause';
            }} else {{
                video.pause();
                playBtn.textContent = '▶ Play';
            }}
        }});

        // Initialize default episode
        switchEpisode("{default_ep_key}");
    </script>
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"✅ Generated Goalkeeper Distribution Valuation Dashboard: {out_html}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Goalkeeper Distribution Valuation Engine (Scene-Neutral)."
    )
    p.add_argument(
        "--episode",
        type=str,
        default="all",
        help="Episode to evaluate: 'episode21', 'episode_2500_2525', or 'all' (default: all)",
    )
    p.add_argument("--video", type=str, default=None, help="Path to input video file.")
    p.add_argument(
        "--calibration", type=str, default=None, help="Path to calibration JSON file."
    )
    p.add_argument(
        "--decision-time-s",
        type=float,
        default=None,
        help="Decision timestamp in seconds.",
    )
    p.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to precomputed tracking records JSON.",
    )
    p.add_argument("--out", type=str, default=None, help="Path to output HTML report.")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh YOLO detection and tracking.",
    )
    return p.parse_args()


def run(args: Optional[argparse.Namespace] = None) -> None:
    opts = args or parse_args()

    configs_to_run: list[MatchConfig] = []
    if opts.episode == "all":
        configs_to_run = [
            episode21_config(),
            episode_2500_2525_config(),
            episode_7818_7828_config(),
        ]
    elif opts.episode in ("episode21", "0708_0720", "ep1"):
        configs_to_run = [episode21_config()]
    elif opts.episode in ("episode_2500_2525", "2500_2525", "ep2"):
        configs_to_run = [episode_2500_2525_config()]
    elif opts.episode in ("episode_7818_7828", "7818_7828", "ep3", "clip3"):
        configs_to_run = [episode_7818_7828_config()]
    else:
        configs_to_run = [get_match_config(opts.episode)]

    evaluated_episodes = []
    for cfg in configs_to_run:
        ep_data = evaluate_match_episode(cfg, opts)
        evaluated_episodes.append(ep_data)

    out_path = (
        Path(opts.out)
        if opts.out
        else Path("reports/goalkeeper_distribution_valuation.html")
    )
    generate_multi_episode_dashboard_html(evaluated_episodes, out_path)


if __name__ == "__main__":
    run()
