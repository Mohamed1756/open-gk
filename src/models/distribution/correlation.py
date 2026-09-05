"""Outlet independence: do two options share one fate?

Two outlets down the same lane are one trigger away from dying together, no
matter how their individual scores read. A pair counts as correlated only when
geometry AND threat agree (narrow angle plus the same covering defender), so
geometrically close but independently marked outlets stay independent. Notes
only: correlation never moves a score (decision quality is judged at release,
and whether a shared trigger actually jumps is execution/luck).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.physics.gk_constraints import DUEL_CONTROL_RADIUS_M

# Majority-of-window rule for shared fate (a counting rule, not calibrated).
SHARED_FATE_MAJORITY = 0.5


def angular_gap_rad(
    passer_xy: Tuple[float, float],
    lead_a_xy: Tuple[float, float],
    lead_b_xy: Tuple[float, float],
) -> float:
    """Absolute angle between two outlet corridors from the passer."""
    ax = lead_a_xy[0] - passer_xy[0]
    ay = lead_a_xy[1] - passer_xy[1]
    bx = lead_b_xy[0] - passer_xy[0]
    by = lead_b_xy[1] - passer_xy[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.acos(cosang)


def correlation_threshold_rad(dist_m: float) -> float:
    """Angle subtended by one control diameter at outlet distance.

    Two corridors closer than this share a defender's control reach: the
    threshold shrinks with distance instead of being a fixed magic angle.
    """
    if dist_m < 1e-6:
        return math.pi
    return (2.0 * DUEL_CONTROL_RADIUS_M) / dist_m


def frame_correlated(
    passer_xy: Tuple[float, float],
    lead_a_xy: Tuple[float, float],
    lead_b_xy: Tuple[float, float],
    presser_a_id: Optional[int],
    presser_b_id: Optional[int],
) -> bool:
    """One frame: narrow angle AND the same covering defender."""
    if presser_a_id is None or presser_b_id is None:
        return False
    if presser_a_id != presser_b_id:
        return False
    dist = max(
        math.hypot(lead_a_xy[0] - passer_xy[0], lead_a_xy[1] - passer_xy[1]),
        math.hypot(lead_b_xy[0] - passer_xy[0], lead_b_xy[1] - passer_xy[1]),
    )
    return angular_gap_rad(passer_xy, lead_a_xy, lead_b_xy) < correlation_threshold_rad(
        dist
    )


def shared_fate_ratio(
    passer_xy: Tuple[float, float],
    leads_a: List[Tuple[float, float]],
    leads_b: List[Tuple[float, float]],
    pressers_a: List[Optional[int]],
    pressers_b: List[Optional[int]],
) -> float:
    """Fraction of aligned frames where the pair shares one fate."""
    n = len(leads_a)
    if n == 0 or not (len(leads_b) == len(pressers_a) == len(pressers_b) == n):
        raise ValueError("correlation sequences must align in length")
    hits = sum(
        1
        for la, lb, pa, pb in zip(leads_a, leads_b, pressers_a, pressers_b)
        if frame_correlated(passer_xy, la, lb, pa, pb)
    )
    return hits / n


def find_shared_fate(
    outlet_ids: List[int],
    passer_xy: Tuple[float, float],
    leads: Dict[int, List[Tuple[float, float]]],
    pressers: Dict[int, List[Optional[int]]],
    majority: float = SHARED_FATE_MAJORITY,
) -> Dict[int, List[int]]:
    """Outlets sharing fate with each outlet over the window (majority rule)."""
    shared: Dict[int, List[int]] = {tid: [] for tid in outlet_ids}
    for i, aid in enumerate(outlet_ids):
        for bid in outlet_ids[i + 1 :]:
            ratio = shared_fate_ratio(
                passer_xy, leads[aid], leads[bid], pressers[aid], pressers[bid]
            )
            if ratio >= majority:
                shared[aid].append(bid)
                shared[bid].append(aid)
    return shared
