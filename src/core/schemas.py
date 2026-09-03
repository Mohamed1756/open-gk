"""
Minimal data schemas for distribution evaluation.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

from src.core.geometry import PitchPoint
from src.core.types import DistributionType, DistributionOutcome


class DistributionSituation(BaseModel):
    """Context for a goalkeeper distribution action."""

    pass_origin: PitchPoint
    pass_target: PitchPoint
    pass_length_m: float
    distribution_type: DistributionType
    nearest_presser_dist_m: float
    press_opponents_count: int
    outcome: DistributionOutcome = DistributionOutcome.SUCCESS_RETAINED

    # Computed fields
    expected_completion_prob: Optional[float] = None
    progression_threat_added: Optional[float] = None
    turnover_risk_cost: Optional[float] = None
    distribution_value: Optional[float] = None
