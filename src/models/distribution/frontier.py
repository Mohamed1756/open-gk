"""
Multi-Objective Decision Frontier for Goalkeeper Distribution Valuation (Module M7).

Computes the Pareto frontier over candidate distribution actions across
Defensive Cushion (safety) and Net Expected Value (progression), enforcing
edge-case constraints for saturation, baseline clearance, intra-player pruning,
and corridor occlusion.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.physics.gk_constraints import (
    CLEARANCE_BASELINE_CUSHION_S,
    CLEARANCE_BASELINE_NET_EV,
    CUSHION_SATURATION_CAP_S,
    MIN_POST_RECEIPT_CUSHION_S,
)


@dataclass(frozen=True)
class DecisionFrontierPoint:
    """Represents a single candidate distribution action in decision space."""

    candidate_id: int
    action_type: str
    target_pos: Tuple[float, float]
    raw_cushion_s: float
    effective_cushion_s: float
    net_ev: float
    path_score: float
    path_status: str
    is_occluded: bool
    is_aerial: bool
    is_intra_dominant: bool
    is_pareto_optimal: bool
    is_clearance_baseline: bool
    dispersion_sigma_m: float = 0.0
    xp_completion_prob: float = 0.0
    turnover_hazard: float = 0.0
    sub_manifold_points: Tuple["DecisionFrontierPoint", ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the frontier point to a plain dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "target_pos": [round(self.target_pos[0], 2), round(self.target_pos[1], 2)],
            "raw_cushion_s": round(self.raw_cushion_s, 2),
            "effective_cushion_s": round(self.effective_cushion_s, 2),
            "net_ev": round(self.net_ev, 4),
            "path_score": round(self.path_score, 1),
            "path_status": self.path_status,
            "is_occluded": self.is_occluded,
            "is_aerial": self.is_aerial,
            "is_intra_dominant": self.is_intra_dominant,
            "is_pareto_optimal": self.is_pareto_optimal,
            "is_clearance_baseline": self.is_clearance_baseline,
            "dispersion_sigma_m": round(self.dispersion_sigma_m, 2),
            "xp_completion_prob": round(self.xp_completion_prob, 3),
            "turnover_hazard": round(self.turnover_hazard, 4),
            "sub_manifold_points": [p.to_dict() for p in self.sub_manifold_points],
        }


@dataclass(frozen=True)
class DecisionFrontierResult:
    """Complete multi-objective decision landscape at release commitment."""

    all_points: List[DecisionFrontierPoint]
    team_points: List[DecisionFrontierPoint]
    pareto_frontier: List[DecisionFrontierPoint]
    recommended_point: Optional[DecisionFrontierPoint]
    is_press_collapse: bool
    clearance_baseline: DecisionFrontierPoint
    cushion_saturation_cap_s: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the frontier result to a JSON-compatible dictionary."""
        return {
            "all_points": [p.to_dict() for p in self.all_points],
            "team_points": [p.to_dict() for p in self.team_points],
            "pareto_frontier": [p.to_dict() for p in self.pareto_frontier],
            "recommended_point": (
                self.recommended_point.to_dict()
                if self.recommended_point is not None
                else None
            ),
            "is_press_collapse": self.is_press_collapse,
            "clearance_baseline": self.clearance_baseline.to_dict(),
            "cushion_saturation_cap_s": self.cushion_saturation_cap_s,
        }


def compute_effective_cushion(
    raw_cushion_s: float,
    cap_s: float = CUSHION_SATURATION_CAP_S,
) -> float:
    """Applies defensive arrival cushion saturation cap (tau_safe)."""
    if raw_cushion_s <= 0.0:
        return raw_cushion_s
    return min(raw_cushion_s, cap_s)


def is_strictly_dominated(
    candidate: DecisionFrontierPoint,
    competitors: Sequence[DecisionFrontierPoint],
) -> bool:
    """Evaluates if candidate is strictly dominated in cushion and Net EV."""
    if candidate.is_occluded:
        return True

    c_x = candidate.effective_cushion_s
    c_y = candidate.net_ev

    for other in competitors:
        if (
            other.candidate_id == candidate.candidate_id
            and other.action_type == candidate.action_type
        ):
            continue
        if other.is_occluded:
            continue

        o_x = other.effective_cushion_s
        o_y = other.net_ev

        # Dominance: at least as good in both dimensions, strictly better in one
        if o_x >= c_x and o_y >= c_y and (o_x > c_x or o_y > c_y):
            return True

    return False


def create_clearance_baseline_point() -> DecisionFrontierPoint:
    """Creates the permanent reference floor for direct touchline clearance."""
    return DecisionFrontierPoint(
        candidate_id=-9999,
        action_type="CLEARANCE",
        target_pos=(0.0, 0.0),
        raw_cushion_s=CLEARANCE_BASELINE_CUSHION_S,
        effective_cushion_s=CLEARANCE_BASELINE_CUSHION_S,
        net_ev=CLEARANCE_BASELINE_NET_EV,
        path_score=15.0,
        path_status="CLEARANCE_BASELINE",
        is_occluded=False,
        is_aerial=True,
        is_intra_dominant=True,
        is_pareto_optimal=False,
        is_clearance_baseline=True,
        dispersion_sigma_m=0.0,
        xp_completion_prob=0.99,
        turnover_hazard=abs(CLEARANCE_BASELINE_NET_EV),
        sub_manifold_points=(),
    )


def compute_decision_frontier(
    evaluated_options: Sequence[Dict[str, Any]],
    saturation_cap_s: float = CUSHION_SATURATION_CAP_S,
) -> DecisionFrontierResult:
    """
    Computes the multi-objective Pareto decision frontier from evaluated options.

    Performs:
    1. Cushion saturation capping to eliminate dead-space backpass skew.
    2. Intra-player manifold dominance selection (best manifold per teammate).
    3. Corridor occlusion exclusion.
    4. Clearance reference baseline injection.
    5. Press collapse detection under all-negative options.
    6. Non-dominated Pareto frontier extraction and optimal recommendation.
    """
    clearance_pt = create_clearance_baseline_point()

    # 1. Parse all options into DecisionFrontierPoint instances
    parsed_points_by_rec: Dict[int, List[DecisionFrontierPoint]] = {}

    for opt in evaluated_options:
        track_id = int(opt.get("target_track_id", opt.get("track_id", 0)))
        raw_cush = float(opt.get("t_post_cushion_s", 0.0))
        eff_cush = compute_effective_cushion(raw_cush, saturation_cap_s)
        net_ev = float(opt.get("net_ev", 0.0))
        path_score = float(opt.get("path_score", 0.0))
        path_status = str(opt.get("path_status", "UNKNOWN"))
        is_occluded = bool(
            opt.get("is_occluded", False)
            or opt.get("is_los_blocked", False)
            or path_status == "OCCLUDED"
        )
        is_aerial = bool(
            opt.get("is_aerial", False) or float(opt.get("pass_dist_m", 0.0)) >= 35.0
        )
        target_pos_raw = opt.get("target_pos", (0.0, 0.0))
        target_pos = (float(target_pos_raw[0]), float(target_pos_raw[1]))
        action_type = str(opt.get("action_type", "FEET"))

        pt = DecisionFrontierPoint(
            candidate_id=track_id,
            action_type=action_type,
            target_pos=target_pos,
            raw_cushion_s=raw_cush,
            effective_cushion_s=eff_cush,
            net_ev=net_ev,
            path_score=path_score,
            path_status=path_status,
            is_occluded=is_occluded,
            is_aerial=is_aerial,
            is_intra_dominant=False,
            is_pareto_optimal=False,
            is_clearance_baseline=False,
            dispersion_sigma_m=float(opt.get("dispersion_sigma_m", 0.0)),
            xp_completion_prob=float(opt.get("xp_completion_prob", 0.0)),
            turnover_hazard=float(opt.get("turnover_hazard", 0.0)),
        )

        parsed_points_by_rec.setdefault(track_id, []).append(pt)

    # 2. Intra-player dominance selection
    team_points: List[DecisionFrontierPoint] = []
    all_points: List[DecisionFrontierPoint] = []

    for track_id, variants in parsed_points_by_rec.items():
        # Pick best variant by path_score
        best_variant = max(variants, key=lambda v: v.path_score)
        subs = tuple(v for v in variants if v.action_type != best_variant.action_type)
        dominant_pt = DecisionFrontierPoint(
            candidate_id=best_variant.candidate_id,
            action_type=best_variant.action_type,
            target_pos=best_variant.target_pos,
            raw_cushion_s=best_variant.raw_cushion_s,
            effective_cushion_s=best_variant.effective_cushion_s,
            net_ev=best_variant.net_ev,
            path_score=best_variant.path_score,
            path_status=best_variant.path_status,
            is_occluded=best_variant.is_occluded,
            is_aerial=best_variant.is_aerial,
            is_intra_dominant=True,
            is_pareto_optimal=False,
            is_clearance_baseline=False,
            dispersion_sigma_m=best_variant.dispersion_sigma_m,
            xp_completion_prob=best_variant.xp_completion_prob,
            turnover_hazard=best_variant.turnover_hazard,
            sub_manifold_points=subs,
        )
        team_points.append(dominant_pt)
        all_points.extend(variants)

    # 3. Press Collapse Detection
    # Viable options: unoccluded, non-negative cushion, Net EV > clearance floor
    viable_options = [
        p
        for p in team_points
        if not p.is_occluded
        and p.effective_cushion_s >= MIN_POST_RECEIPT_CUSHION_S
        and p.net_ev > clearance_pt.net_ev
    ]
    is_press_collapse = len(viable_options) == 0

    # 4. Pareto Frontier Extraction among team points
    # Competitor pool for non-dominated check
    competitors = [p for p in team_points if not p.is_occluded]

    pareto_pts: List[DecisionFrontierPoint] = []
    updated_team_points: List[DecisionFrontierPoint] = []

    for pt in team_points:
        is_opt = not pt.is_occluded and not is_strictly_dominated(pt, competitors)
        if is_opt:
            pt_opt = DecisionFrontierPoint(
                candidate_id=pt.candidate_id,
                action_type=pt.action_type,
                target_pos=pt.target_pos,
                raw_cushion_s=pt.raw_cushion_s,
                effective_cushion_s=pt.effective_cushion_s,
                net_ev=pt.net_ev,
                path_score=pt.path_score,
                path_status=pt.path_status,
                is_occluded=pt.is_occluded,
                is_aerial=pt.is_aerial,
                is_intra_dominant=True,
                is_pareto_optimal=True,
                is_clearance_baseline=False,
                dispersion_sigma_m=pt.dispersion_sigma_m,
                xp_completion_prob=pt.xp_completion_prob,
                turnover_hazard=pt.turnover_hazard,
                sub_manifold_points=pt.sub_manifold_points,
            )
            pareto_pts.append(pt_opt)
            updated_team_points.append(pt_opt)
        else:
            updated_team_points.append(pt)

    # Sort Pareto frontier by effective cushion ascending to form curve
    pareto_pts.sort(key=lambda p: p.effective_cushion_s)

    # 5. Recommendation
    if is_press_collapse:
        recommended = clearance_pt
    else:
        # Highest scoring Pareto optimal point that meets viability floor
        viable_pareto = [
            p
            for p in pareto_pts
            if p.effective_cushion_s >= MIN_POST_RECEIPT_CUSHION_S
            and p.net_ev > clearance_pt.net_ev
        ]
        recommended = (
            max(viable_pareto, key=lambda p: p.path_score) if viable_pareto else None
        )

    return DecisionFrontierResult(
        all_points=all_points,
        team_points=updated_team_points,
        pareto_frontier=pareto_pts,
        recommended_point=recommended,
        is_press_collapse=is_press_collapse,
        clearance_baseline=clearance_pt,
        cushion_saturation_cap_s=saturation_cap_s,
    )
