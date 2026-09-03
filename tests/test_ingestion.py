"""
Unit tests for data normalization and StatsBomb loaders.
"""

import pytest
from src.ingestion.normalizer import CoordinateNormalizer
from src.ingestion.statsbomb import StatsBombLoader
from src.core.geometry import PitchPoint


def test_coordinate_normalizer_statsbomb():
    # Center of pitch in StatsBomb: (60, 40) -> Canonical: (52.5, 34.0)
    pt = CoordinateNormalizer.from_statsbomb(60.0, 40.0)
    assert pt.x == pytest.approx(52.5)
    assert pt.y == pytest.approx(34.0)

    # Opponent goal in StatsBomb: (120, 40) -> Canonical: (105.0, 34.0)
    goal_pt = CoordinateNormalizer.from_statsbomb(120.0, 40.0)
    assert goal_pt.x == pytest.approx(105.0)
    assert goal_pt.y == pytest.approx(34.0)


def test_align_to_gk_defending_end():
    pt = PitchPoint(100.0, 40.0)
    aligned = CoordinateNormalizer.align_to_gk_defending_end(
        pt, gk_defending_right_to_left=True
    )
    assert aligned.x == pytest.approx(5.0)
    assert aligned.y == pytest.approx(28.0)


def test_statsbomb_loader_360_mock(tmp_path):
    mock_360 = [
        {
            "event_uuid": "event-1234",
            "freeze_frame": [
                {
                    "teammate": True,
                    "actor": False,
                    "keeper": True,
                    "location": [10.0, 40.0],
                },
                {
                    "teammate": False,
                    "actor": True,
                    "keeper": False,
                    "location": [30.0, 35.0],
                },
            ],
        }
    ]
    file_path = tmp_path / "sample_360.json"
    import json

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mock_360, f)

    loader = StatsBombLoader()
    frames = loader.load_360_frames("sample", local_path=file_path)
    assert "event-1234" in frames
    ctx = frames["event-1234"]
    assert len(ctx.players) == 2
    assert ctx.gk_location is not None
    assert (
        ctx.teammate_count == 0
    )  # 1 teammate who is keeper, so non-keeper teammate count is 0
    assert ctx.opponent_count == 1
