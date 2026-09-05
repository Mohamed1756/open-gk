"""
Unit tests for Multi-Objective Decision Frontier (Module M7).
"""

from src.models.distribution.frontier import (
    compute_decision_frontier,
    compute_effective_cushion,
)
from src.physics.gk_constraints import (
    CLEARANCE_BASELINE_NET_EV,
    CUSHION_SATURATION_CAP_S,
)


def test_cushion_saturation_cap():
    # 1. Excess cushion > 2.5s saturates at cap
    assert compute_effective_cushion(5.0) == CUSHION_SATURATION_CAP_S
    assert compute_effective_cushion(3.2) == CUSHION_SATURATION_CAP_S

    # 2. Moderate cushion remains unchanged
    assert compute_effective_cushion(1.5) == 1.5
    assert compute_effective_cushion(0.8) == 0.8

    # 3. Negative cushion preserved
    assert compute_effective_cushion(-1.2) == -1.2


def test_clearance_baseline_and_press_collapse():
    # High-press episode with all hospital / negative passes
    options = [
        {
            "target_track_id": 1,
            "action_type": "FEET",
            "t_post_cushion_s": -1.2,
            "net_ev": -0.15,
            "path_score": 1.0,
            "path_status": "PRESS_TRAP",
            "is_occluded": False,
        },
        {
            "target_track_id": 2,
            "action_type": "PROGRESSIVE_CHANNEL",
            "t_post_cushion_s": -0.8,
            "net_ev": -0.09,
            "path_score": 2.5,
            "path_status": "CONTROL_TRAP",
            "is_occluded": False,
        },
    ]

    res = compute_decision_frontier(options)

    # Must detect press collapse
    assert res.is_press_collapse
    assert res.recommended_point is not None
    assert res.recommended_point.action_type == "CLEARANCE"
    assert res.recommended_point.net_ev == CLEARANCE_BASELINE_NET_EV


def test_occlusion_pruned_from_frontier():
    # Option A has massive cushion and progression, but corridor is occluded
    options = [
        {
            "target_track_id": 10,
            "action_type": "FEET",
            "t_post_cushion_s": 3.0,
            "net_ev": 0.25,
            "path_score": 90.0,
            "path_status": "OCCLUDED",
            "is_occluded": True,
        },
        {
            "target_track_id": 11,
            "action_type": "FEET",
            "t_post_cushion_s": 1.5,
            "net_ev": 0.08,
            "path_score": 70.0,
            "path_status": "FEASIBLE_OPEN",
            "is_occluded": False,
        },
    ]

    res = compute_decision_frontier(options)

    # Occluded option must NOT be Pareto optimal
    occluded_pt = next(p for p in res.team_points if p.candidate_id == 10)
    assert not occluded_pt.is_pareto_optimal

    # Unoccluded viable option is on the frontier
    open_pt = next(p for p in res.team_points if p.candidate_id == 11)
    assert open_pt.is_pareto_optimal
    assert len(res.pareto_frontier) == 1
    assert res.pareto_frontier[0].candidate_id == 11


def test_intra_player_manifold_pruning():
    # Single player with 2 manifold evaluations
    options = [
        {
            "target_track_id": 18,
            "action_type": "FEET",
            "t_post_cushion_s": 1.0,
            "net_ev": 0.02,
            "path_score": 45.0,
            "path_status": "FEASIBLE_OPEN",
            "is_occluded": False,
        },
        {
            "target_track_id": 18,
            "action_type": "PROGRESSIVE_CHANNEL",
            "t_post_cushion_s": 2.2,
            "net_ev": 0.09,
            "path_score": 82.0,
            "path_status": "FEASIBLE_OPEN",
            "is_occluded": False,
        },
    ]

    res = compute_decision_frontier(options)

    # In team_points, only 1 representative for player 18
    assert len(res.team_points) == 1
    dominant = res.team_points[0]
    assert dominant.candidate_id == 18
    assert dominant.action_type == "PROGRESSIVE_CHANNEL"
    assert dominant.is_intra_dominant

    # Sub-manifold contains FEET variant
    assert len(dominant.sub_manifold_points) == 1
    assert dominant.sub_manifold_points[0].action_type == "FEET"


def test_pareto_dominance_sorting():
    # Option 1: Cushion 1.0s, Net EV +0.15 (high progression)
    # Option 2: Cushion 2.2s, Net EV +0.08 (high safety)
    # Option 3: Cushion 0.8s, Net EV +0.05 (dominated by Option 1 in both dimensions)
    options = [
        {
            "target_track_id": 1,
            "action_type": "FEET",
            "t_post_cushion_s": 1.0,
            "net_ev": 0.15,
            "path_score": 85.0,
            "path_status": "FEASIBLE_OPEN",
            "is_occluded": False,
        },
        {
            "target_track_id": 2,
            "action_type": "FEET",
            "t_post_cushion_s": 2.2,
            "net_ev": 0.08,
            "path_score": 80.0,
            "path_status": "FEASIBLE_OPEN",
            "is_occluded": False,
        },
        {
            "target_track_id": 3,
            "action_type": "FEET",
            "t_post_cushion_s": 0.8,
            "net_ev": 0.05,
            "path_score": 50.0,
            "path_status": "FEASIBLE_OPEN",
            "is_occluded": False,
        },
    ]

    res = compute_decision_frontier(options)

    frontier_ids = [p.candidate_id for p in res.pareto_frontier]
    assert 1 in frontier_ids
    assert 2 in frontier_ids
    # Option 3 is dominated and must NOT be in the frontier
    assert 3 not in frontier_ids

    # Sorted by cushion ascending: 1.0s, then 2.2s
    assert res.pareto_frontier[0].candidate_id == 1
    assert res.pareto_frontier[1].candidate_id == 2
