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
from typing import Any, Dict, List, Optional
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    evaluate_pressing_approach_cut,
    update_spatial_memory_buffer,
    visible_person_ids,
    facing_to_ball,
    vision_logit_penalty,
    SCAN_KNOWN_THRESHOLD,
    SCAN_KNOWN_PENALTY,
    OUT_OF_VISION_PENALTY,
)
from src.physics.biomechanics import (
    compute_hip_pivot_latency,
    compute_biomechanical_execution_penalty,
    evaluate_effective_press_closure,
    is_pass_inertia_locked,
    compute_projected_closing_velocity,
)
from src.models.distribution.evaluator import DistributionEvaluator
from src.models.distribution.first_touch import (
    find_first_touch,
    decision_possession_frames,
)
from src.models.pressing.types import PressingActor, PressingSnapshot
from src.models.pressing.units import cluster_lines, press_state
from src.models.pressing.arrival import arrival_margin
from src.models.pressing.matchup import assign_pressers
from src.models.pressing.readiness import receiver_readiness
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


def evaluate_distribution_decision(
    passer_pos: PitchPoint,
    receiver_candidates: List[Dict[str, Any]],
    opponents: List[PitchPoint],
    evaluator: DistributionEvaluator,
    passer_facing_angle_rad: Optional[float] = None,
    touch_vel_xy_ms: Optional[tuple[float, float]] = None,
    nearest_presser_vel_ms: Optional[tuple[float, float]] = None,
    attack_dir_x: float = -1.0,
    memory_weights: Optional[Dict[int, float]] = None,
    opponent_actors: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluates candidate pass options taking into account goalkeeper field of view
    (140-deg visual cone), line-of-sight occlusion, first-touch momentum inertia locks,
    and dynamic pressing approach trajectory cuts.
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

    option_evals = []

    for cand in receiver_candidates:
        raw_x, raw_y = cand["pitch_xy"][0], cand["pitch_xy"][1]
        rec_vx = cand.get("pitch_vel_ms", [0.0, 0.0])[0]
        rec_vy = cand.get("pitch_vel_ms", [0.0, 0.0])[1]

        raw_dist = math.hypot(raw_x - passer_pos.x, raw_y - passer_pos.y)
        pass_speed_ms = 19.0
        flight_time_s = min(1.2, raw_dist / max(5.0, pass_speed_ms))

        # Project lead pass into receiver's running path
        lead_x = max(0.0, min(105.0, raw_x + rec_vx * flight_time_s))
        lead_y = max(0.0, min(68.0, raw_y + rec_vy * flight_time_s))
        target_pos = PitchPoint(x=round(lead_x, 2), y=round(lead_y, 2))
        dist_m = float(
            math.hypot(target_pos.x - passer_pos.x, target_pos.y - passer_pos.y)
        )

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
            DistributionType.SHORT_PASS
            if dist_m <= 25.0
            else DistributionType.LONG_PASS
        )
        raw_xp = evaluator.estimate_completion_probability(
            dist_m, min_press_dist, dist_type
        )

        pass_angle = math.atan2(
            target_pos.y - passer_pos.y, target_pos.x - passer_pos.x
        )
        in_cone = True
        diff_deg = 0.0
        is_blocked = False
        prep_latency = 0.15
        exec_mult = 1.0

        if passer_facing_angle_rad is not None:
            in_cone, abs_diff = is_in_visual_cone(
                passer_pos, passer_facing_angle_rad, target_pos, fov_deg=140.0
            )
            diff_deg = round(math.degrees(abs_diff), 0)
            is_blocked, _ = compute_los_occlusion(
                passer_pos, target_pos, obstacles=opponents, obstacle_radius_m=0.75
            )
            prep_latency = compute_hip_pivot_latency(
                passer_facing_angle_rad, pass_angle
            )
            exec_mult = compute_biomechanical_execution_penalty(
                passer_facing_angle_rad, pass_angle
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
                    pass_speed_ms=pass_speed_ms,
                    prep_latency_s=prep_latency,
                    time_horizon_s=1.2,
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

        rec_actor = recv_by_id.get(int(cand["track_id"]))
        arrival = (
            arrival_margin(rec_actor, opp_actors, passer_pos)
            if rec_actor is not None
            else None
        )
        is_margin_trapped = arrival is not None and arrival.margin_s < 0.0
        pairing = pair_by_recv.get(int(cand["track_id"]))
        is_pincer = bool(pairing.is_pincer) if pairing is not None else False
        nearest_presser_actor = None
        if pairing is not None and pairing.presser_track_id is not None:
            nearest_presser_actor = next(
                (o for o in opp_actors if o.track_id == pairing.presser_track_id),
                None,
            )
        presser_closing = False
        if nearest_presser_actor is not None:
            dx = target_pos.x - nearest_presser_actor.pos_m.x
            dy = target_pos.y - nearest_presser_actor.pos_m.y
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                vx, vy = nearest_presser_actor.vel_ms
                presser_closing = (vx * dx + vy * dy) / dist > 1.0
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
        if readiness is not None:
            prep_latency = prep_latency + readiness.turn_latency_s

        effective_press_dist = evaluate_effective_press_closure(
            min_press_dist,
            presser_closing_speed_ms=closing_speed_ms,
            prep_latency_s=prep_latency,
        )

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

        cushion_ratio = float(
            np.clip((effective_press_dist - 4.0) / (12.0 - 4.0), 0.0, 1.0)
        )
        score_safety = 35.0 * effective_xp
        score_cushion = 35.0 * cushion_ratio * effective_xp
        score_evasion = (
            min(30.0, opponents_pressed_count * 10.0) * effective_xp * cushion_ratio
        )

        dist_to_own_goal = (105.0 - target_pos.x) if attack_dir_x < 0 else target_pos.x
        is_defensive_third_tight = (
            effective_press_dist < 8.0 and dist_to_own_goal < 35.0
        )
        turnover_danger = (
            12.0 * (1.0 - effective_xp) if is_defensive_third_tight else 0.0
        )

        raw_path_score = score_safety + score_cushion + score_evasion - turnover_danger

        if is_inertia_locked:
            path_status = "INERTIA_LOCKED"
            grade_color = "#ef4444"
            vis_label = "MOMENTUM LOCKED"
            bottleneck = "First-touch momentum opposes strike angle under press"
            penalty = 35.0
        elif is_approach_cut:
            path_status = "APPROACH_CUT"
            grade_color = "#f97316"
            vis_label = "PRESSER CUTTING LANE"
            bottleneck = "Defender sprint trajectory cuts arrival window"
            penalty = 30.0
        elif is_margin_trapped:
            assert arrival is not None
            path_status = "PRESS_TRAP"
            grade_color = "#ef4444"
            vis_label = f"TRAPPED (press {-arrival.margin_s:.1f}s early)"
            bottleneck = (
                f"Presser #{arrival.best_presser_id} reaches the lead target "
                f"{-arrival.margin_s:.2f}s before the ball"
            )
            penalty = 25.0
        elif is_blocked:
            path_status = "SHADOWED_LANE"
            grade_color = "#ef4444"
            vis_label = "BLOCKED BY STRIKER"
            bottleneck = "Striker cover shadow blocks direct passing corridor"
            penalty = 30.0
        elif effective_press_dist < 4.0:
            path_status = "PRESS_TRAP"
            grade_color = "#ef4444"
            vis_label = f"IN VISION ({diff_deg}°)" if in_cone else "PRESS TRAP"
            bottleneck = "Receiver will be immediately tackled (< 4.0m cushion)"
            penalty = 25.0
        elif effective_press_dist < 8.0:
            path_status = "CONTESTED_POCKET"
            grade_color = "#facc15"
            vis_label = f"IN VISION ({diff_deg}°)" if in_cone else "CONTESTED"
            bottleneck = f"Compressed pocket ({effective_press_dist:.1f}m cushion) requires delayed release to open"
            penalty = 12.0
        elif scan_known:
            path_status = "KNOWN_FROM_SCAN"
            grade_color = "#2dd4bf"
            vis_label = f"KNOWN FROM SCAN ({memory_weight:.0%})"
            bottleneck = "Outside current gaze but scanned within memory horizon"
            penalty = SCAN_KNOWN_PENALTY
        elif not in_cone:
            path_status = "OUT_OF_VISION"
            grade_color = "#94a3b8"
            vis_label = f"OUT OF CONE ({diff_deg}°)"
            bottleneck = "Outside 140° visual field (scanning blindspot)"
            penalty = OUT_OF_VISION_PENALTY
        else:
            path_status = "FEASIBLE_OPEN"
            grade_color = "#22c55e"
            vis_label = f"IN VISION ({diff_deg}°)"
            bottleneck = "Open passing lane with viable receiver separation"
            penalty = 0.0

        if is_pincer and path_status not in ("APPROACH_CUT", "INERTIA_LOCKED"):
            bottleneck = (
                f"{bottleneck} (2nd man #{pairing.second_presser_id} synchronized)"
                if pairing is not None and pairing.second_presser_id is not None
                else f"{bottleneck} (pincer)"
            )

        final_path_score = round(max(0.0, min(100.0, raw_path_score - penalty)), 1)

        option_evals.append(
            {
                "track_id": cand["track_id"],
                "role": cand["team_label"],
                "target_pos": [round(target_pos.x, 1), round(target_pos.y, 1)],
                "distance_m": round(dist_m, 1),
                "nearest_presser_m": round(min_press_dist, 1),
                "effective_press_m": round(effective_press_dist, 1),
                "bypassed_pressers": opponents_pressed_count,
                "xp_completion_prob": effective_xp,
                "raw_xp": round(raw_xp, 3),
                "path_score": final_path_score,
                "score_safety": round(score_safety, 1),
                "score_evasion": round(score_evasion, 1),
                "score_cushion": round(score_cushion, 1),
                "decision_grade": path_status,
                "path_status": path_status,
                "grade_color": grade_color,
                "in_visual_cone": in_cone,
                "is_los_blocked": is_blocked,
                "is_inertia_locked": is_inertia_locked,
                "is_approach_cut": is_approach_cut,
                "memory_weight": round(memory_weight, 3),
                "scan_known": scan_known,
                "margin_s": arrival.margin_s if arrival is not None else 99.0,
                "best_presser_id": arrival.best_presser_id
                if arrival is not None
                else None,
                "is_pincer": is_pincer,
                "second_presser_id": pairing.second_presser_id
                if pairing is not None
                else None,
                "presser_track_id": pairing.presser_track_id
                if pairing is not None
                else None,
                "free_man": bool(pairing.is_free) if pairing is not None else False,
                "readiness_mult": readiness.readiness_mult
                if readiness is not None
                else 1.0,
                "turn_latency_s": readiness.turn_latency_s
                if readiness is not None
                else 0.0,
                "facing_source": readiness.facing_source
                if readiness is not None
                else "unknown",
                "visual_label": vis_label,
                "diff_deg": diff_deg,
                "bottleneck_diagnostic": bottleneck,
                "hip_latency_s": prep_latency,
                "exec_multiplier": exec_mult,
            }
        )

    option_evals.sort(key=lambda x: x["path_score"], reverse=True)
    for rank_idx, opt in enumerate(option_evals, 1):
        opt["path_rank"] = rank_idx
        opt["is_optimal"] = (
            rank_idx == 1
            and opt["path_score"] >= 30.0
            and opt["path_status"] not in BAD_STATUSES
        )
        if rank_idx == 1:
            if opt["path_score"] >= 30.0 and opt["path_status"] not in BAD_STATUSES:
                opt["optimality_class"] = "RECOMMENDED"
                opt["grade_color"] = "#22c55e"
                opt["decision_grade"] = "RECOMMENDED"
            elif opt["path_score"] >= 20.0 and opt["path_status"] not in BAD_STATUSES:
                opt["optimality_class"] = "VIABLE_OUTLET"
                opt["grade_color"] = "#38bdf8"
                opt["decision_grade"] = "VIABLE_OUTLET"
            else:
                opt["optimality_class"] = opt["path_status"]
                opt["grade_color"] = (
                    "#ef4444" if opt["path_status"] in BAD_STATUSES else "#facc15"
                )
                opt["decision_grade"] = opt["path_status"]
        elif opt["path_score"] >= 40.0 and opt["path_status"] not in BAD_STATUSES:
            opt["optimality_class"] = "VIABLE_OUTLET"
            opt["grade_color"] = "#38bdf8"
            opt["decision_grade"] = "VIABLE_OUTLET"
        else:
            opt["optimality_class"] = opt["path_status"]

    return option_evals


BAD_STATUSES = {"INERTIA_LOCKED", "SHADOWED_LANE", "PRESS_TRAP", "APPROACH_CUT"}


def project_decision_to_screen(
    homography: Any,
    passer_pos: PitchPoint,
    decision_eval: List[Dict[str, Any]],
    facing_rad: float,
    decision_frame_idx: int,
    cone_radius_m: float = 12.0,
    cone_steps: int = 16,
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
                "grade_color": opt["grade_color"],
                "is_optimal": bool(opt.get("is_optimal")),
                "is_bad": status in BAD_STATUSES,
                "is_scan_known": status == "KNOWN_FROM_SCAN",
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
        matchups.append(
            {
                "presser_track_id": opt["presser_track_id"],
                "receiver_track_id": opt["track_id"],
                "beaten": bool(opt.get("margin_s", 99.0) < 0.0),
                "free_man": bool(opt.get("free_man", False)),
                "margin_s": opt.get("margin_s", 99.0),
            }
        )
    return {
        "anchor_px": anchor_px,
        "lanes": lanes,
        "cone_px": cone_px,
        "facing_tip_px": facing_tip_px,
        "matchups": matchups,
    }


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
    passer_track_id: int | None = None
    passer_pos = PitchPoint(x=95.0, y=34.0)
    receivers: List[Dict[str, Any]] = []
    opponents: List[PitchPoint] = []
    gk_facing_angle_rad = 1.571

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

                if "bbox" in passer:
                    pose_res = pose_estimator.estimate_pose_in_crop(
                        frame, tuple(passer["bbox"]), frame_idx=frame_idx
                    )
                    if pose_res.is_valid:
                        gk_facing_angle_rad = pose_res.facing_angle_pitch_rad

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
                    if "bbox" in p:
                        cand_pose = pose_estimator.estimate_pose_in_crop(
                            frame, tuple(p["bbox"]), frame_idx=frame_idx
                        )
                        if cand_pose.is_valid:
                            facing_by_track[int(p["track_id"])] = (
                                float(cand_pose.facing_angle_pitch_rad),
                                "pose",
                            )

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
    opponent_actor_list = _actors_from_entities(eval_opp_ents, "opp")

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
    if tracking_records:
        mem_lo = max(0, decision_frame_idx - int(3.5 * fps))
        last_seen: Dict[int, float] = {}
        for fi in range(mem_lo, min(decision_frame_idx + 1, len(tracking_records))):
            ents = tracking_records[fi]["entities"]
            t_now = fi / fps
            gk_xy = _passer_xy(ents, passer_track_id)
            if gk_xy is None:
                continue
            ball = _ball_xy(ents)
            facing = facing_to_ball(
                (gk_xy[0], gk_xy[1]),
                (ball[0], ball[1]) if ball else None,
                gk_facing_angle_rad,
            )
            persons = [
                (e["track_id"], (e["pitch_xy"][0], e["pitch_xy"][1]))
                for e in ents
                if e.get("class_name") == "person"
                and not is_official_label(e.get("team_label", ""))
            ]
            visible = visible_person_ids((gk_xy[0], gk_xy[1]), facing, persons)
            last_seen, memory_weights = update_spatial_memory_buffer(
                last_seen, visible, t_now
            )

    decision_evaluation = evaluate_distribution_decision(
        passer_pos,
        receivers,
        opponents,
        evaluator,
        passer_facing_angle_rad=gk_facing_angle_rad,
        touch_vel_xy_ms=touch_vel_xy,
        nearest_presser_vel_ms=nearest_presser_vel,
        attack_dir_x=float(match.attack_dir_x),
        memory_weights=memory_weights,
        opponent_actors=opponent_actor_list,
    )

    decision_overlay = project_decision_to_screen(
        homography,
        passer_pos,
        decision_evaluation,
        gk_facing_angle_rad,
        int(decision_frame_idx),
        cone_radius_m=12.0,
    )

    opt_rows = []
    for opt in decision_evaluation:
        badge = (
            f"<span class='badge' style='background:{opt['grade_color']}33; "
            f"color:{opt['grade_color']}; border:1px solid {opt['grade_color']}66;'>"
            f"{opt['optimality_class']}</span>"
        )
        opt_tag = (
            " <span class='badge' style='background:#22c55e22; color:#22c55e; border:1px solid #22c55e66; margin-left:4px; font-size:0.7rem;'>PRIMARY</span>"
            if opt.get("is_optimal")
            else ""
        )
        # True Field of View column
        if opt.get("in_visual_cone"):
            fov_label = f"IN VISION ({opt.get('diff_deg', 0):.0f}°)"
            fov_color = "#22c55e"
        elif opt.get("scan_known"):
            fov_label = f"KNOWN FROM SCAN ({opt.get('memory_weight', 0):.0%})"
            fov_color = "#2dd4bf"
        else:
            fov_label = f"OUT OF CONE ({opt.get('diff_deg', 0):.0f}°)"
            fov_color = "#94a3b8"

        opt_rows.append(
            f"<tr>"
            f"<td><strong>#{opt['track_id']}</strong>{opt_tag}</td>"
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
        "coneRadiusM": 12.0,
        "firstTouchNote": first_touch_note,
        "pressNote": press_note,
        "tableTimeNote": f"Release-Point Option Telemetry (t = {decision_frame_idx / fps:.2f}s · {first_touch_note})",
        "decisionFrames": sorted(decision_frames),
        "decisionPx": decision_overlay,
        "evalData": decision_evaluation,
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
        .switcher-bar {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem; background: var(--bg-secondary); padding: 0.75rem 1.25rem; border-radius: 0.75rem; border: 1px solid var(--border); }}
        .grid {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
        .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.25rem; }}
        .card-header {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }}
        .viewport {{ position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 0.5rem; overflow: hidden; border: 1px solid var(--border); }}
        video {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
        .overlay-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
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

        <div class="grid">
            <!-- Left: Match Video with Live Tracking -->
            <div class="card">
                <div class="card-header">
                    <span>1. Match Video with Live Tracking</span>
                    <span id="time-display" style="font-family: monospace; color: var(--accent-blue); font-weight: 700;">0.00s</span>
                </div>
                <div class="viewport">
                    <video id="match-video" src="{default_ep["videoSrc"]}" playsinline muted preload="auto"></video>
                    <svg id="video-overlay" class="overlay-svg" viewBox="0 0 1280 720" preserveAspectRatio="xMidYMid meet"></svg>
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
                    <label><input type="checkbox" id="radar-toggle" style="accent-color:#94a3b8;"> 2D engine view</label>
                    <span><span style="display:inline-block;width:14px;height:2px;background:#ef4444;vertical-align:middle;"></span> beaten matchup</span>
                    <span><span style="display:inline-block;width:10px;height:10px;border:1px dashed #22c55e;border-radius:50%;vertical-align:middle;"></span> free man</span>
                </div>
            </div>

            <!-- Right: 2D Pitch Radar (compute view, hidden by default) -->
            <div class="card" id="radar-card" style="display:none;">
                <div class="card-header">
                    <span>2. 2D Engine (compute view)</span>
                    <span style="display:flex; gap:0.5rem; align-items:center;">
                    <label style="font-size:0.75rem; color:var(--text-muted);"><input type="checkbox" id="lanes-toggle" checked style="accent-color:#22c55e;"> frozen decision lanes</label>
                    <span class="badge" style="background: rgba(192, 38, 211, 0.2); color: #e879f9; border: 1px solid #c026d3;">GK: Maignan</span>
                    </span>
                </div>
                <svg class="pitch-canvas" viewBox="0 0 525 340">
                    <rect x="0" y="0" width="525" height="340" fill="#0f3d1e"/>
                    <line x1="262.5" y1="0" x2="262.5" y2="340" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                    <circle cx="262.5" cy="170" r="45.75" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                    <circle cx="262.5" cy="170" r="2" fill="#ffffff"/>
                    <rect x="0" y="66" width="82.5" height="208" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                    <rect x="0" y="116" width="27.5" height="108" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                    <rect x="442.5" y="66" width="82.5" height="208" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>
                    <rect x="497.5" y="116" width="27.5" height="108" fill="none" stroke="#ffffff" stroke-width="1.2" opacity="0.6"/>

                    <g id="radar-lanes"></g>
                    <g id="radar-nodes"></g>
                </svg>

                <div style="display: flex; gap: 1rem; margin-top: 0.75rem; font-size: 0.85rem; color: var(--text-muted); flex-wrap: wrap;">
                    <div><span style="display:inline-block;width:10px;height:10px;background:#c026d3;border-radius:50%;"></span> Mike Maignan (GK)</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#22c55e;border-radius:50%;"></span> Line-Breaker (High EV)</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:#facc15;border-radius:50%;"></span> Safe Recycle</div>
                    <div><span style="display:inline-block;width:10px;height:10px;background:transparent;border:1px dashed #94a3b8;border-radius:50%;"></span> outside-hull (excluded)</div>
                </div>
                <div id="hull-note" style="font-size:0.75rem; color:var(--text-muted); margin-top:0.35rem;"></div>
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

        const video = document.getElementById('match-video');
        const scrubber = document.getElementById('scrubber');
        const playBtn = document.getElementById('play-btn');
        const timeDisplay = document.getElementById('time-display');
        const frameCounter = document.getElementById('frame-counter');
        const videoOverlay = document.getElementById('video-overlay');
        const radarLanes = document.getElementById('radar-lanes');
        const radarNodes = document.getElementById('radar-nodes');

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
            video.currentTime = ep.decisionIdx / ep.fps;
            renderFrame(ep.decisionIdx);
        }}

        document.querySelectorAll('.ep-pill-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                switchEpisode(btn.getAttribute('data-ep'));
            }});
        }});

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
                        {{pts: trailPts.slice(0, thirds * 2), op: 0.14}},
                        {{pts: trailPts.slice(thirds * 2, thirds * 3), op: 0.28}},
                        {{pts: trailPts.slice(thirds * 3), op: 0.5}},
                    ];
                    for (const s of segs) {{
                        if (s.pts.length < 2) continue;
                        const d = 'M ' + s.pts.map(p => `${{p[0].toFixed(1)}} ${{p[1].toFixed(1)}}`).join(' L ');
                        vSvg += `<path d="${{d}}" fill="none" stroke="#f8fafc" stroke-width="2.5" stroke-linecap="round" opacity="${{s.op}}"/>`;
                    }}
                }}
                if (trailPts.length > 2) {{
                    const last = trailPts[trailPts.length - 1];
                    const prev = trailPts[trailPts.length - 3];
                    const dx = last[0] - prev[0], dy = last[1] - prev[1];
                    const tipX = last[0] + dx * 3, tipY = last[1] + dy * 3;
                    vSvg += `<line x1="${{last[0]}}" y1="${{last[1]}}" x2="${{tipX}}" y2="${{tipY}}" stroke="#f8fafc" stroke-width="2" stroke-dasharray="5 4" opacity="0.55"/>`;
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
                        const tagged = lane.is_optimal || lane.is_bad || lane.is_scan_known;
                        const w = lane.is_optimal ? 4.0 : ((lane.is_bad || lane.is_scan_known) ? 2.5 : 1.5);
                        const dash = (lane.is_optimal || lane.is_scan_known) ? 'none' : (lane.is_bad ? '6 4' : '2 3');
                        const op = lane.is_optimal ? 1.0 : ((lane.is_bad || lane.is_scan_known) ? 0.9 : 0.45);
                        const glow = lane.is_optimal ? ' style="filter:drop-shadow(0 0 4px #22c55e)"' : '';
                        vSvg += `<line x1="${{lax}}" y1="${{lay}}" x2="${{tx}}" y2="${{ty}}" stroke="${{lane.grade_color}}" stroke-width="${{w}}" stroke-dasharray="${{dash}}" opacity="${{op}}"${{glow}}/>`;
                        vSvg += `<circle cx="${{tx}}" cy="${{ty}}" r="${{lane.is_optimal ? 11 : 8}}" fill="none" stroke="${{lane.grade_color}}" stroke-width="2" opacity="${{op}}"/>`;
                        vSvg += `<circle cx="${{gx}}" cy="${{gy}}" r="4" fill="none" stroke="${{lane.grade_color}}" stroke-width="1" stroke-dasharray="2 2" opacity="0.5"><title>Evaluated lead target</title></circle>`;
                        if (tagged) {{
                            const tag = lane.is_optimal ? `RECOMMENDED #${{lane.track_id}}` : `#${{lane.track_id}} ${{lane.visual_label}}`;
                            const ty2 = Math.max(14, ty - 14);
                            vSvg += `<rect x="${{tx + 10}}" y="${{ty2 - 11}}" width="${{Math.min(230, tag.length * 6.4 + 12)}}" height="16" fill="#0a0f1d" opacity="0.85" rx="3"/>`;
                            vSvg += `<text x="${{tx + 16}}" y="${{ty2 + 1}}" fill="${{lane.grade_color}}" font-size="10" font-weight="bold">${{tag}}</text>`;
                        }}
                    }}
                }}
                if (ep.decisionPx.matchups) {{
                    for (const mu of ep.decisionPx.matchups) {{
                        const pe = data.entities.find(e => e.track_id === mu.presser_track_id);
                        const re = data.entities.find(e => e.track_id === mu.receiver_track_id);
                        if (!pe || !re || !pe.screen_uv || !re.screen_uv) continue;
                        const col = mu.beaten ? '#ef4444' : '#e8e8e8';
                        const wd = mu.beaten ? 2.5 : 1.0;
                        const dp = mu.beaten ? 'none' : '3 3';
                        const op = mu.beaten ? 0.85 : 0.35;
                        vSvg += `<line x1="${{pe.screen_uv[0]}}" y1="${{pe.screen_uv[1]}}" x2="${{re.screen_uv[0]}}" y2="${{re.screen_uv[1]}}" stroke="${{col}}" stroke-width="${{wd}}" stroke-dasharray="${{dp}}" opacity="${{op}}"/>`;
                        if (mu.free_man) {{
                            vSvg += `<circle cx="${{re.screen_uv[0]}}" cy="${{re.screen_uv[1]}}" r="13" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-dasharray="2 2" opacity="0.8"><title>Free man (margin ${{mu.margin_s}}s)</title></circle>`;
                        }}
                    }}
                }}
            }}
            videoOverlay.innerHTML = vSvg;

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

            if (showLanes && anchor && ep.evalData && ep.evalData.length > 0) {{
                const anchorX = Math.max(0, Math.min(105, anchor[0])) * 5.0;
                const anchorY = (68.0 - Math.max(0, Math.min(68, anchor[1]))) * 5.0;
                const facingSvgAngle = -ep.facingRad;
                const coneRadius = ep.coneRadiusM * 5.0;
                const halfFov = (70.0 * Math.PI) / 180.0;
                const a1 = facingSvgAngle - halfFov;
                const a2 = facingSvgAngle + halfFov;
                const p1x = anchorX + coneRadius * Math.cos(a1);
                const p1y = anchorY + coneRadius * Math.sin(a1);
                const p2x = anchorX + coneRadius * Math.cos(a2);
                const p2y = anchorY + coneRadius * Math.sin(a2);

                lSvg += `<path d="M ${{anchorX}} ${{anchorY}} L ${{p1x}} ${{p1y}} A ${{coneRadius}} ${{coneRadius}} 0 0 1 ${{p2x}} ${{p2y}} Z" fill="rgba(56, 189, 248, 0.12)" stroke="rgba(56, 189, 248, 0.45)" stroke-width="1.5" stroke-dasharray="4 2"/>`;
                lSvg += `<text x="${{anchorX - 30}}" y="${{anchorY + coneRadius + 14}}" fill="#38bdf8" font-size="9" font-weight="bold">140° VISUAL FIELD (12m)</text>`;

                for (const opt of ep.evalData) {{
                    const tx = Math.max(0, Math.min(105, opt.target_pos[0])) * 5.0;
                    const ty = (68.0 - Math.max(0, Math.min(68, opt.target_pos[1]))) * 5.0;
                    const isFeasible = opt.in_visual_cone && !opt.is_los_blocked;
                    const strokeDash = isFeasible ? "none" : "4 3";
                    const strokeOpacity = isFeasible ? "0.9" : "0.35";
                    lSvg += `<line x1="${{anchorX}}" y1="${{anchorY}}" x2="${{tx}}" y2="${{ty}}" stroke="${{opt.grade_color}}" stroke-width="2.2" stroke-dasharray="${{strokeDash}}" opacity="${{strokeOpacity}}"/>
                             <circle cx="${{tx}}" cy="${{ty}}" r="10" fill="none" stroke="${{opt.grade_color}}" stroke-width="1.5" opacity="${{strokeOpacity}}"/>`;
                }}
                lSvg += `<circle cx="${{anchorX}}" cy="${{anchorY}}" r="5" fill="none" stroke="#e879f9" stroke-width="1.2" stroke-dasharray="2 2"><title>Keeper anchor</title></circle>`;
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
        for (const id of ['vid-lanes-toggle', 'vid-cone-toggle', 'vid-trail-toggle']) {{
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', () => {{
                renderFrame(parseInt(scrubber.value) || 0);
            }});
        }}
        const radarToggle = document.getElementById('radar-toggle');
        const radarCard = document.getElementById('radar-card');
        if (radarToggle && radarCard) radarToggle.addEventListener('change', () => {{
            radarCard.style.display = radarToggle.checked ? '' : 'none';
        }});

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
