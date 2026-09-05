"""
Goalkeeper Distribution Valuation Engine (Module M7).
Evaluates passing risk/reward conditioned on opponent press intensity and the Buildup +1 dynamic.
"""

import math
from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field

from src.config import PITCH_LENGTH_METERS, GOAL_Y_CENTER
from src.core.geometry import PitchPoint, compute_cover_shadow
from src.core.schemas import DistributionSituation
from src.core.types import DistributionType, DistributionOutcome
from src.physics.gk_constraints import (
    CENTRAL_TURNOVER_PEAK_XG,
    CENTRAL_ZONE_SIGMA_X_M,
    CENTRAL_ZONE_SIGMA_Y_M,
    TURNOVER_HAZARD_FLOOR_XG,
    EXIT_AFFORDANCE_PINNED_MULT,
    EXIT_AFFORDANCE_BASE_MULT,
    EXIT_AFFORDANCE_PER_LANE,
    EXIT_AFFORDANCE_MAX_MULT,
    GK_PRESS_URGENCY_RADIUS_M,
    GK_PRESS_CRITICAL_RADIUS_M,
)


class PassOptionEvaluation(BaseModel):
    """Evaluation scorecard for an individual pass option."""

    receiver_name: str
    receiver_role: str
    target_pos: PitchPoint
    distance_m: float
    xp_completion_prob: float = Field(
        ..., description="Pass completion probability [0, 1]"
    )
    progression_xt: float = Field(
        ..., description="Expected threat progression gained [0, 1]"
    )
    turnover_risk_cost: float = Field(
        ..., description="Expected conceded threat if turned over [0, 1]"
    )
    expected_value: float = Field(
        ..., description="Net Expected Value = xP*xT - (1-xP)*Cost"
    )
    is_optimal_pass: bool = False
    decision_grade: str = "NEUTRAL"  # LINE_BREAKER, SAFE_RECYCLE, PRESS_TRAP


class DistributionEvaluator:
    """
    Evaluates goalkeeper distribution decisions and passes under press.
    Quantifies press-evasion reward and turnover danger counterfactually.
    """

    def __init__(
        self,
        press_danger_radius_m: float = 6.0,
    ):
        self.press_danger_radius_m = press_danger_radius_m

    def estimate_completion_probability(
        self,
        pass_length_m: float,
        nearest_presser_dist_m: float,
        dist_type: DistributionType,
    ) -> float:
        """
        Calculates expected completion probability (xP) given distance and press pressure.
        """
        # Baseline logistic decay with length
        if dist_type in [DistributionType.SHORT_PASS, DistributionType.THROW]:
            base_logit = 3.6 - 0.075 * min(pass_length_m, 35.0)
        elif dist_type == DistributionType.GOAL_KICK:
            base_logit = 2.0 - 0.045 * min(pass_length_m, 70.0)
        else:  # PUNT or LONG_PASS
            base_logit = 1.2 - 0.038 * min(pass_length_m, 70.0)

        # Press pressure penalty if opponent is close
        if nearest_presser_dist_m < self.press_danger_radius_m:
            pressure_penalty = (
                (self.press_danger_radius_m - nearest_presser_dist_m)
                / self.press_danger_radius_m
            ) * 1.2
            base_logit -= pressure_penalty

        prob = 1.0 / (1.0 + np.exp(-base_logit))
        return float(np.clip(prob, 0.15, 0.99))

    def compute_progression_threat(
        self,
        origin: PitchPoint,
        target: PitchPoint,
        opponents_pressed_count: int,
        attack_dir_x: float = 1.0,
        exit_lanes_count: int = 0,
    ) -> float:
        """
        Computes expected progression threat (xT / possession value) created by the pass,
        scaling by bypassed pressers and downstream receiver exit affordance lanes.
        """
        if attack_dir_x >= 0:
            delta_x_m = max(0.0, target.x - origin.x)
        else:
            delta_x_m = max(0.0, origin.x - target.x)

        base_threat = (delta_x_m / PITCH_LENGTH_METERS) * 0.05
        press_bonus = min(0.04, opponents_pressed_count * 0.012)

        # Exit affordance multiplier: more unblocked progressive exit lanes amplify value
        if exit_lanes_count == 0:
            exit_mult = EXIT_AFFORDANCE_PINNED_MULT
        else:
            exit_mult = min(
                EXIT_AFFORDANCE_MAX_MULT,
                EXIT_AFFORDANCE_BASE_MULT + EXIT_AFFORDANCE_PER_LANE * exit_lanes_count,
            )

        total_threat = (base_threat + press_bonus) * exit_mult
        return float(np.clip(total_threat, 0.001, 0.15))

    def compute_turnover_risk_cost(
        self,
        target: PitchPoint,
        attack_dir_x: float = 1.0,
    ) -> float:
        """
        Computes continuous 2D spatial turnover hazard (conceded xG).
        Centrally aligned turnovers near own goal carry maximum hazard (up to 0.55 xG),
        decaying smoothly via a 2D Gaussian density toward the touchline and upfield.
        """
        if attack_dir_x >= 0:
            d_x_m = max(0.0, target.x)
        else:
            d_x_m = max(0.0, PITCH_LENGTH_METERS - target.x)

        d_y_m = abs(target.y - GOAL_Y_CENTER)

        exponent = -(
            (d_x_m**2) / (2.0 * (CENTRAL_ZONE_SIGMA_X_M**2))
            + (d_y_m**2) / (2.0 * (CENTRAL_ZONE_SIGMA_Y_M**2))
        )
        hazard = CENTRAL_TURNOVER_PEAK_XG * math.exp(exponent)
        return float(max(TURNOVER_HAZARD_FLOOR_XG, min(0.65, hazard)))

    def compute_pressure_relief_value(
        self,
        origin_hazard: float,
        target_hazard: float,
        gk_press_dist_m: float,
        closing_speed_ms: float = 0.0,
    ) -> float:
        """
        Computes the possession safety relief (conceded xG delta saved) achieved by releasing
        the ball away from goalkeeper pressure into a safer pitch zone.
        """
        d_eff = max(0.0, gk_press_dist_m - closing_speed_ms * 0.35)
        if d_eff >= GK_PRESS_URGENCY_RADIUS_M:
            urgency = 0.0
        elif d_eff <= GK_PRESS_CRITICAL_RADIUS_M:
            urgency = 1.0
        else:
            urgency = (GK_PRESS_URGENCY_RADIUS_M - d_eff) / (
                GK_PRESS_URGENCY_RADIUS_M - GK_PRESS_CRITICAL_RADIUS_M
            )

        baseline_hazard = origin_hazard * urgency
        relief = max(0.0, baseline_hazard - target_hazard)
        return float(relief)

    def compute_net_distribution_ev(
        self,
        retention_xp: float,
        prog_threat: float,
        target_turnover_cost: float,
        relief_value: float = 0.0,
    ) -> float:
        """
        Computes net expected possession value delta (action value relative to baseline risk).
        """
        ev = (retention_xp * (prog_threat + relief_value)) - (
            (1.0 - retention_xp) * target_turnover_cost
        )
        return float(ev)

    def evaluate_distribution(
        self,
        situation: DistributionSituation,
    ) -> DistributionSituation:
        """
        Evaluates a single goalkeeper distribution action.
        """
        xp = self.estimate_completion_probability(
            pass_length_m=situation.pass_length_m,
            nearest_presser_dist_m=situation.nearest_presser_dist_m,
            dist_type=situation.distribution_type,
        )
        prog_threat = self.compute_progression_threat(
            origin=situation.pass_origin,
            target=situation.pass_target,
            opponents_pressed_count=situation.press_opponents_count,
        )
        turnover_cost = self.compute_turnover_risk_cost(
            target=situation.pass_target,
        )

        situation.expected_completion_prob = round(xp, 3)
        situation.progression_threat_added = round(prog_threat, 4)
        situation.turnover_risk_cost = round(turnover_cost, 4)

        # Net value computation
        is_successful = situation.outcome == DistributionOutcome.SUCCESS_RETAINED
        if is_successful:
            # Reward for completed pass scaling with difficulty (1 - xP) and progression
            val = prog_threat * (0.5 + 0.5 * (1.0 - xp))
        else:
            # Penalty for turnover scaling with likelihood of failure and turnover cost
            val = -turnover_cost * (0.5 + 0.5 * xp)

        situation.distribution_value = round(float(val), 5)
        return situation

    def aggregate_distribution_performance(
        self,
        situations: List[DistributionSituation],
    ) -> Dict[str, Any]:
        """
        Aggregates distribution and press-evasion performance.
        """
        if not situations:
            return {
                "total_distributions": 0,
                "completion_rate": 0.0,
                "expected_completion_rate": 0.0,
                "completion_over_expected": 0.0,
                "avg_nearest_presser_m": 0.0,
                "total_distribution_value": 0.0,
                "press_resistance_grade": "N/A",
            }

        evaluated = [self.evaluate_distribution(s) for s in situations]
        completed = sum(
            1 for s in evaluated if s.outcome == DistributionOutcome.SUCCESS_RETAINED
        )
        comp_rate = completed / len(evaluated)
        exp_comp = np.mean([s.expected_completion_prob or 0.8 for s in evaluated])
        val_sum = sum(s.distribution_value or 0.0 for s in evaluated)
        press_dist_mean = np.mean([s.nearest_presser_dist_m for s in evaluated])

        # Press resistance grade
        coe = comp_rate - exp_comp
        if coe >= 0.05:
            grade = "Elite (Press-Resistant)"
        elif coe >= -0.02:
            grade = "Secure"
        else:
            grade = "Press-Vulnerable"

        return {
            "total_distributions": len(evaluated),
            "completion_rate": round(float(comp_rate), 3),
            "expected_completion_rate": round(float(exp_comp), 3),
            "completion_over_expected": round(float(coe), 3),
            "avg_nearest_presser_m": round(float(press_dist_mean), 2),
            "total_distribution_value": round(float(val_sum), 4),
            "press_resistance_grade": grade,
        }

    def estimate_xp(
        self,
        pass_length_m: float,
        nearest_presser_dist_m: float,
        cover_shadow: float = 0.0,
    ) -> float:
        """
        Calculates expected completion probability (xP) given distance, receiver pressure,
        and cover shadow obstruction along the passing lane.
        """
        # Baseline logistic decay
        base_logit = 3.4 - 0.075 * min(pass_length_m, 60.0)

        # Pressure penalty at receiver
        if nearest_presser_dist_m < self.press_danger_radius_m:
            press_penalty = (
                (self.press_danger_radius_m - nearest_presser_dist_m)
                / self.press_danger_radius_m
            ) * 1.4
            base_logit -= press_penalty

        # Cover shadow obstruction penalty
        base_logit -= cover_shadow * 2.5

        prob = 1.0 / (1.0 + np.exp(-base_logit))
        return float(np.clip(prob, 0.10, 0.98))

    def compute_progression_xt(self, origin: PitchPoint, target: PitchPoint) -> float:
        """
        Computes expected progression threat (xT / territory gain).
        """
        delta_x = max(0.0, target.x - origin.x)
        # Base pitch progression
        xt = (delta_x / 105.0) * 0.08

        # Line-breaking bonus for passes reaching central midfield or final third
        if target.x >= 35.0:
            xt += 0.04
        if 20.0 <= target.y <= 48.0 and target.x >= 30.0:
            xt += 0.03  # Central corridor premium

        return float(np.clip(xt, 0.01, 0.18))

    def evaluate_all_options(
        self,
        gk_pos: PitchPoint,
        teammates: List[Dict[str, Any]],
        opponents: List[Dict[str, Any]],
    ) -> List[PassOptionEvaluation]:
        """
        Evaluates all available passing lanes from the goalkeeper to outfield teammates.
        """
        evaluations: List[PassOptionEvaluation] = []

        for tm in teammates:
            target = PitchPoint(x=tm["x"], y=tm["y"])
            dist = gk_pos.distance_to(target)

            # Nearest opponent to receiver
            min_opp_dist = 999.0
            max_shadow = 0.0

            for opp in opponents:
                opp_pt = PitchPoint(x=opp["x"], y=opp["y"])
                d = target.distance_to(opp_pt)
                if d < min_opp_dist:
                    min_opp_dist = d

                # Cover shadow along pass lane
                shadow = compute_cover_shadow(gk_pos, target, opp_pt)
                if shadow > max_shadow:
                    max_shadow = shadow

            xp = self.estimate_xp(
                pass_length_m=dist,
                nearest_presser_dist_m=min_opp_dist,
                cover_shadow=max_shadow,
            )
            xt = self.compute_progression_xt(gk_pos, target)
            turnover_cost = self.compute_turnover_risk_cost(target)

            # Net Expected Value calculation
            ev = (xp * xt) - ((1.0 - xp) * turnover_cost)

            # Decision grading
            if ev > 0.06 and xt > 0.05:
                grade = "LINE_BREAKER"
            elif ev >= 0.0 and xp >= 0.85:
                grade = "SAFE_RECYCLE"
            else:
                grade = "PRESS_TRAP"

            evaluations.append(
                PassOptionEvaluation(
                    receiver_name=tm.get("name", "Teammate"),
                    receiver_role=tm.get("role", "Field Player"),
                    target_pos=target,
                    distance_m=round(dist, 1),
                    xp_completion_prob=round(xp, 3),
                    progression_xt=round(xt, 3),
                    turnover_risk_cost=round(turnover_cost, 3),
                    expected_value=round(ev, 4),
                    decision_grade=grade,
                )
            )

        # Mark optimal pass (highest Expected Value)
        if evaluations:
            best_idx = max(
                range(len(evaluations)), key=lambda i: evaluations[i].expected_value
            )
            evaluations[best_idx].is_optimal_pass = True

        return evaluations
