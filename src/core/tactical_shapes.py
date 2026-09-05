"""Tactical shape presets and standard positional buildup configurations."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class TacticalPressingShape(BaseModel):
    name: str
    description: str
    opponents: List[Dict[str, Any]]
    shape_edges: List[List[int]] = Field(default_factory=list)


def get_standard_buildup_teammates() -> List[Dict[str, Any]]:
    return [
        {"name": "Laporte", "role": "Left Center-Back", "x": 14.0, "y": 20.0},
        {"name": "Le Normand", "role": "Right Center-Back", "x": 14.0, "y": 48.0},
        {"name": "Cucurella", "role": "Left Back", "x": 28.0, "y": 9.0},
        {"name": "Carvajal", "role": "Right Back", "x": 28.0, "y": 59.0},
        {"name": "Rodri", "role": "Pivot #6", "x": 26.0, "y": 34.0},
        {"name": "Ruiz", "role": "Left #8", "x": 42.0, "y": 22.0},
        {"name": "Olmo", "role": "Attacking Mid #10", "x": 50.0, "y": 34.0},
        {"name": "Williams", "role": "Left Winger", "x": 55.0, "y": 10.0},
        {"name": "Yamal", "role": "Right Winger", "x": 55.0, "y": 58.0},
        {"name": "Morata", "role": "Striker #9", "x": 62.0, "y": 34.0},
    ]


def create_high_433_press() -> TacticalPressingShape:
    opps = [
        {"name": "Foden (LW)", "role": "Pressing Winger", "x": 18.0, "y": 24.0},
        {"name": "Kane (CF)", "role": "Central Presser", "x": 19.0, "y": 34.0},
        {"name": "Saka (RW)", "role": "Pressing Winger", "x": 18.0, "y": 44.0},
        {"name": "Bellingham (LCM)", "role": "Pressing Mid", "x": 32.0, "y": 26.0},
        {"name": "Rice (DM)", "role": "Cover Pivot", "x": 30.0, "y": 34.0},
        {"name": "Mainoo (RCM)", "role": "Pressing Mid", "x": 32.0, "y": 42.0},
        {"name": "Shaw (LB)", "role": "Defender", "x": 50.0, "y": 16.0},
        {"name": "Guéhi (LCB)", "role": "Defender", "x": 52.0, "y": 28.0},
        {"name": "Stones (RCB)", "role": "Defender", "x": 52.0, "y": 40.0},
        {"name": "Walker (RB)", "role": "Defender", "x": 50.0, "y": 52.0},
    ]
    edges = [
        [0, 1],
        [1, 2],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [8, 9],
    ]
    return TacticalPressingShape(
        name="High 4-3-3 Gegenpress",
        description="Front 3 aggressively press 18-yard box; central passing lane is tightly contested.",
        opponents=opps,
        shape_edges=edges,
    )


def create_mid_442_block() -> TacticalPressingShape:
    opps = [
        {"name": "Striker 1", "role": "First Pressing Line", "x": 24.0, "y": 28.0},
        {"name": "Striker 2", "role": "First Pressing Line", "x": 24.0, "y": 40.0},
        {"name": "Left Mid", "role": "Midfield Bank", "x": 42.0, "y": 14.0},
        {"name": "Central Mid 1", "role": "Midfield Bank", "x": 40.0, "y": 26.0},
        {"name": "Central Mid 2", "role": "Midfield Bank", "x": 40.0, "y": 42.0},
        {"name": "Right Mid", "role": "Midfield Bank", "x": 42.0, "y": 54.0},
        {"name": "Left Back", "role": "Back Line", "x": 58.0, "y": 14.0},
        {"name": "Center Back 1", "role": "Back Line", "x": 56.0, "y": 28.0},
        {"name": "Center Back 2", "role": "Back Line", "x": 56.0, "y": 40.0},
        {"name": "Right Back", "role": "Back Line", "x": 58.0, "y": 54.0},
    ]
    edges = [
        [0, 1],
        [2, 3],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [8, 9],
    ]
    return TacticalPressingShape(
        name="Compact 4-4-2 Mid-Block",
        description="Concedes wide CBs, but creates a tight cover shadow trap on central pivot.",
        opponents=opps,
        shape_edges=edges,
    )


def create_man_to_man_lock() -> TacticalPressingShape:
    opps = [
        {"name": "Presser 1", "role": "Shadow CB1", "x": 16.0, "y": 21.0},
        {"name": "Presser 2", "role": "Shadow CB2", "x": 16.0, "y": 47.0},
        {"name": "Presser 3", "role": "Shadow LB", "x": 29.0, "y": 11.0},
        {"name": "Presser 4", "role": "Shadow RB", "x": 29.0, "y": 57.0},
        {"name": "Presser 5", "role": "Shadow #6", "x": 27.5, "y": 34.5},
        {"name": "Presser 6", "role": "Shadow #8", "x": 43.0, "y": 23.0},
        {"name": "Presser 7", "role": "Shadow #10", "x": 51.0, "y": 33.5},
        {"name": "Presser 8", "role": "Shadow LW", "x": 56.0, "y": 12.0},
        {"name": "Presser 9", "role": "Shadow RW", "x": 56.0, "y": 56.0},
        {"name": "Presser 10", "role": "Shadow ST", "x": 63.0, "y": 33.5},
    ]
    edges: List[List[int]] = []
    return TacticalPressingShape(
        name="Full Man-to-Man Press Lock",
        description="All 10 teammates closely shadowed. Goalkeeper step-out or long aerial chip is required.",
        opponents=opps,
        shape_edges=edges,
    )
