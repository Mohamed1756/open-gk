"""Continuous Action Manifold for Goalkeeper Distribution Decisions.

Evaluates continuous reachability landing surfaces across three spatial action vectors:
1. Centroid Lead ('FEET'): standard lead along velocity vector
2. Progressive Channel Lead ('PROGRESSIVE_CHANNEL'): space lead into stride
3. Shielded Pocket ('SHIELDED_POCKET'): lateral step away from marker

Enforces physical boundary buffers, multi-teammate target ambiguity penalties,
and kicking execution dispersion smoothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.config import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from src.core.geometry import PitchPoint
from src.models.pressing.arrival import flight_time_s, presser_arrival_s
from src.models.pressing.types import PressingActor
from src.physics.gk_constraints import (
    KICK_DISPERSION_RADIAL_RATIO,
    MANIFOLD_BURST_SPEED_BOOST_MS,
    MANIFOLD_MAX_BURST_LEAD_M,
    MANIFOLD_MIN_BURST_LEAD_M,
    MANIFOLD_SHIELD_OFFSET_DIST_M,
    MAX_LEAD_DISPLACEMENT_M,
    PITCH_TOUCHLINE_BUFFER_M,
    TARGET_AMBIGUITY_BALL_SYNC_WINDOW_S,
    TARGET_AMBIGUITY_MAX_SEPARATION_M,
    TARGET_AMBIGUITY_THRESHOLD_S,
)


@dataclass(frozen=True)
class ManifoldTarget:
    """A parameterized landing coordinate on a receiver's continuous reachability surface."""

    action_type: str  # "FEET", "PROGRESSIVE_CHANNEL", "SHIELDED_POCKET"
    target_pos: PitchPoint
    flight_time_s: float
    pass_dist_m: float
    offset_dist_m: float
    is_boundary_discounted: bool
    boundary_discount_mult: float
    is_ambiguity_penalized: bool
    ambiguity_partner_id: Optional[int]
    dispersion_sigma_m: float


def clamp_to_pitch_buffer(
    pos: PitchPoint,
    buffer_m: float = PITCH_TOUCHLINE_BUFFER_M,
    dispersion_sigma_m: float = 0.0,
) -> Tuple[PitchPoint, bool, float]:
    """Clamps a landing point inside the physical pitch safety buffer and computes rollout decay.

    When an unconstrained landing position crosses or approaches the boundary, the discount multiplier
    incorporates Gaussian survival probability based on delivery dispersion sigma.
    """
    min_x = buffer_m
    max_x = PITCH_LENGTH_METERS - buffer_m
    min_y = buffer_m
    max_y = PITCH_WIDTH_METERS - buffer_m

    clamped_x = max(min_x, min(max_x, pos.x))
    clamped_y = max(min_y, min(max_y, pos.y))
    is_discounted = (clamped_x != pos.x) or (clamped_y != pos.y)

    dist_to_edge = min(
        clamped_x - 0.0,
        PITCH_LENGTH_METERS - clamped_x,
        clamped_y - 0.0,
        PITCH_WIDTH_METERS - clamped_y,
    )
    if (
        pos.x < 0.0
        or pos.x > PITCH_LENGTH_METERS
        or pos.y < 0.0
        or pos.y > PITCH_WIDTH_METERS
    ):
        dist_out = max(
            0.0 - pos.x,
            pos.x - PITCH_LENGTH_METERS,
            0.0 - pos.y,
            pos.y - PITCH_WIDTH_METERS,
        )
        sigma = max(0.5, dispersion_sigma_m)
        z = dist_out / sigma
        discount_mult = float(max(0.10, 0.50 * math.erfc(z / math.sqrt(2.0))))
        is_discounted = True
    elif dist_to_edge < buffer_m + 1.0:
        decay = float(max(0.50, 1.0 - math.exp(-dist_to_edge / 1.5)))
        if dispersion_sigma_m > 0.0:
            sigma = max(0.5, dispersion_sigma_m)
            z = dist_to_edge / sigma
            p_stay = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            discount_mult = float(min(decay, max(0.40, p_stay)))
        else:
            discount_mult = decay
        is_discounted = True
    else:
        discount_mult = 1.0

    return (
        PitchPoint(x=round(clamped_x, 2), y=round(clamped_y, 2)),
        is_discounted,
        round(discount_mult, 3),
    )


def check_target_ambiguity(
    target: PitchPoint,
    intended_receiver: PressingActor,
    teammates: Sequence[PressingActor],
    flight_s: float,
    threshold_s: float = TARGET_AMBIGUITY_THRESHOLD_S,
    max_separation_m: float = TARGET_AMBIGUITY_MAX_SEPARATION_M,
    ball_sync_window_s: float = TARGET_AMBIGUITY_BALL_SYNC_WINDOW_S,
) -> Tuple[bool, Optional[int]]:
    """Detects if multiple teammates in localized proximity arrive at the landing zone concurrently with the ball."""
    intended_arrival, _ = presser_arrival_s(intended_receiver, target)
    for tm in teammates:
        if tm.track_id == intended_receiver.track_id:
            continue
        # Spatial locality check: teammate must be within localized proximity of target or intended receiver
        d_tm_target = math.hypot(tm.pos_m.x - target.x, tm.pos_m.y - target.y)
        d_tm_rec = math.hypot(
            tm.pos_m.x - intended_receiver.pos_m.x,
            tm.pos_m.y - intended_receiver.pos_m.y,
        )
        if d_tm_target > max_separation_m and d_tm_rec > max_separation_m:
            continue

        tm_arrival, _ = presser_arrival_s(tm, target)
        # Arrival synchronization: both players arrive close together AND near the ball arrival
        is_concurrent_players = abs(tm_arrival - intended_arrival) < threshold_s
        is_near_ball_arrival = (
            abs(intended_arrival - flight_s) <= ball_sync_window_s
            or abs(tm_arrival - flight_s) <= ball_sync_window_s
            or (
                tm_arrival <= flight_s + 0.5
                and intended_arrival <= flight_s + 0.5
                and abs(flight_s - intended_arrival) < 0.8
            )
        )
        if is_concurrent_players and is_near_ball_arrival:
            return True, tm.track_id
    return False, None


def generate_receiver_manifold(
    passer_pos: PitchPoint,
    receiver: PressingActor,
    nearest_presser: Optional[PressingActor],
    teammates: Sequence[PressingActor],
    attack_dir_x: float = 1.0,
) -> List[ManifoldTarget]:
    """Generates continuous landing targets across the receiver's reachability manifold."""
    targets: List[ManifoldTarget] = []

    # 1. Base centroid lead target ("FEET")
    raw_dist = math.hypot(
        receiver.pos_m.x - passer_pos.x, receiver.pos_m.y - passer_pos.y
    )
    t_flight_base = flight_time_s(raw_dist)

    vx, vy = receiver.vel_ms
    lead_dx = vx * t_flight_base
    lead_dy = vy * t_flight_base
    disp = math.hypot(lead_dx, lead_dy)
    if disp > MAX_LEAD_DISPLACEMENT_M and disp > 1e-6:
        scale = MAX_LEAD_DISPLACEMENT_M / disp
        lead_dx *= scale
        lead_dy *= scale

    pos_feet = PitchPoint(
        x=receiver.pos_m.x + lead_dx,
        y=receiver.pos_m.y + lead_dy,
    )
    d_pass_raw_feet = math.hypot(pos_feet.x - passer_pos.x, pos_feet.y - passer_pos.y)
    clamped_feet, disc_feet, mult_feet = clamp_to_pitch_buffer(
        pos_feet,
        dispersion_sigma_m=d_pass_raw_feet * KICK_DISPERSION_RADIAL_RATIO,
    )
    d_pass_feet = math.hypot(
        clamped_feet.x - passer_pos.x, clamped_feet.y - passer_pos.y
    )
    t_flight_feet = flight_time_s(d_pass_feet)
    amb_feet, partner_feet = check_target_ambiguity(
        clamped_feet, receiver, teammates, t_flight_feet
    )

    targets.append(
        ManifoldTarget(
            action_type="FEET",
            target_pos=clamped_feet,
            flight_time_s=t_flight_feet,
            pass_dist_m=round(d_pass_feet, 2),
            offset_dist_m=0.0,
            is_boundary_discounted=disc_feet,
            boundary_discount_mult=round(mult_feet, 3),
            is_ambiguity_penalized=amb_feet,
            ambiguity_partner_id=partner_feet,
            dispersion_sigma_m=round(d_pass_feet * KICK_DISPERSION_RADIAL_RATIO, 2),
        )
    )

    # 2. Progressive channel lead into stride ("PROGRESSIVE_CHANNEL")
    # Dynamic kinematic lead scaling with flight time
    dynamic_lead_m = float(
        max(
            MANIFOLD_MIN_BURST_LEAD_M,
            min(
                MANIFOLD_MAX_BURST_LEAD_M,
                MANIFOLD_MIN_BURST_LEAD_M
                + MANIFOLD_BURST_SPEED_BOOST_MS * t_flight_base,
            ),
        )
    )

    # Attacking direction constraint: non-regressive progression
    atk_u = 1.0 if attack_dir_x >= 0 else -1.0
    v_rec_speed = math.hypot(vx, vy)
    if v_rec_speed >= 0.8:
        v_prog = vx * atk_u
        if v_prog >= 0.0:
            dir_x, dir_y = vx / v_rec_speed, vy / v_rec_speed
        else:
            # Tracking backward: progressive channel leads forward into turning space
            dir_x = atk_u
            dir_y = vy / v_rec_speed if abs(vy) > 0.5 else 0.0
            dir_norm = math.hypot(dir_x, dir_y)
            dir_x /= dir_norm
            dir_y /= dir_norm
    else:
        dir_x = atk_u
        dir_y = 0.0

    pos_channel = PitchPoint(
        x=clamped_feet.x + dir_x * dynamic_lead_m,
        y=clamped_feet.y + dir_y * dynamic_lead_m,
    )
    d_pass_raw_ch = math.hypot(
        pos_channel.x - passer_pos.x, pos_channel.y - passer_pos.y
    )
    clamped_ch, disc_ch, mult_ch = clamp_to_pitch_buffer(
        pos_channel,
        dispersion_sigma_m=d_pass_raw_ch * KICK_DISPERSION_RADIAL_RATIO,
    )
    d_pass_ch = math.hypot(clamped_ch.x - passer_pos.x, clamped_ch.y - passer_pos.y)
    t_flight_ch = flight_time_s(d_pass_ch)
    amb_ch, partner_ch = check_target_ambiguity(
        clamped_ch, receiver, teammates, t_flight_ch
    )

    targets.append(
        ManifoldTarget(
            action_type="PROGRESSIVE_CHANNEL",
            target_pos=clamped_ch,
            flight_time_s=t_flight_ch,
            pass_dist_m=round(d_pass_ch, 2),
            offset_dist_m=round(dynamic_lead_m, 2),
            is_boundary_discounted=disc_ch,
            boundary_discount_mult=round(mult_ch, 3),
            is_ambiguity_penalized=amb_ch,
            ambiguity_partner_id=partner_ch,
            dispersion_sigma_m=round(d_pass_ch * KICK_DISPERSION_RADIAL_RATIO, 2),
        )
    )

    # 3. Shielded pocket away from presser ("SHIELDED_POCKET")
    if nearest_presser is not None:
        dx_press = receiver.pos_m.x - nearest_presser.pos_m.x
        dy_press = receiver.pos_m.y - nearest_presser.pos_m.y
        press_dist = math.hypot(dx_press, dy_press)
        if press_dist < 8.0 and press_dist > 1e-3:
            dx_pass = clamped_feet.x - passer_pos.x
            dy_pass = clamped_feet.y - passer_pos.y
            pass_len = math.hypot(dx_pass, dy_pass)
            if pass_len > 1e-2:
                u_pass_x = dx_pass / pass_len
                u_pass_y = dy_pass / pass_len
                # Normal vectors perpendicular to pass corridor
                n1_x, n1_y = -u_pass_y, u_pass_x
                n2_x, n2_y = u_pass_y, -u_pass_x
                # Choose normal directing away from pressing defender
                dot1 = n1_x * dx_press + n1_y * dy_press
                dot2 = n2_x * dx_press + n2_y * dy_press
                if dot1 >= dot2:
                    u_shield_x, u_shield_y = n1_x, n1_y
                else:
                    u_shield_x, u_shield_y = n2_x, n2_y

                # Blend 70% orthogonal body-shielding with 30% direct displacement
                u_away_x = dx_press / press_dist
                u_away_y = dy_press / press_dist
                u_comb_x = 0.70 * u_shield_x + 0.30 * u_away_x
                u_comb_y = 0.70 * u_shield_y + 0.30 * u_away_y
                comb_norm = math.hypot(u_comb_x, u_comb_y)
                if comb_norm > 1e-3:
                    u_comb_x /= comb_norm
                    u_comb_y /= comb_norm
                else:
                    u_comb_x, u_comb_y = u_away_x, u_away_y
            else:
                u_comb_x = dx_press / press_dist
                u_comb_y = dy_press / press_dist

            pos_shield = PitchPoint(
                x=clamped_feet.x + u_comb_x * MANIFOLD_SHIELD_OFFSET_DIST_M,
                y=clamped_feet.y + u_comb_y * MANIFOLD_SHIELD_OFFSET_DIST_M,
            )
            d_pass_raw_sh = math.hypot(
                pos_shield.x - passer_pos.x, pos_shield.y - passer_pos.y
            )
            clamped_sh, disc_sh, mult_sh = clamp_to_pitch_buffer(
                pos_shield,
                dispersion_sigma_m=d_pass_raw_sh * KICK_DISPERSION_RADIAL_RATIO,
            )
            d_pass_sh = math.hypot(
                clamped_sh.x - passer_pos.x, clamped_sh.y - passer_pos.y
            )
            t_flight_sh = flight_time_s(d_pass_sh)
            amb_sh, partner_sh = check_target_ambiguity(
                clamped_sh, receiver, teammates, t_flight_sh
            )

            targets.append(
                ManifoldTarget(
                    action_type="SHIELDED_POCKET",
                    target_pos=clamped_sh,
                    flight_time_s=t_flight_sh,
                    pass_dist_m=round(d_pass_sh, 2),
                    offset_dist_m=MANIFOLD_SHIELD_OFFSET_DIST_M,
                    is_boundary_discounted=disc_sh,
                    boundary_discount_mult=round(mult_sh, 3),
                    is_ambiguity_penalized=amb_sh,
                    ambiguity_partner_id=partner_sh,
                    dispersion_sigma_m=round(
                        d_pass_sh * KICK_DISPERSION_RADIAL_RATIO, 2
                    ),
                )
            )

    return targets


__all__ = [
    "ManifoldTarget",
    "clamp_to_pitch_buffer",
    "check_target_ambiguity",
    "generate_receiver_manifold",
]
