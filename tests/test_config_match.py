from pathlib import Path
import pytest
from src.config_match import (
    MatchConfig,
    episode21_config,
    episode_2500_2525_config,
    episode_7818_7828_config,
    get_match_config,
)


def test_episode21_config():
    cfg = episode21_config()
    assert isinstance(cfg, MatchConfig)
    assert cfg.match_id == "torino_milan_0708_0720"
    assert cfg.possession_team_id == "milan"
    assert cfg.keeper_name == "Mike Maignan"
    assert cfg.calibration_path == Path("data/calibration/episode21.json")
    assert cfg.keeper_kit() is not None
    assert cfg.keeper_kit().is_keeper is True


def test_episode_2500_2525_config():
    cfg = episode_2500_2525_config()
    assert isinstance(cfg, MatchConfig)
    assert cfg.match_id == "torino_milan_2500_2525"
    assert cfg.possession_team_id == "milan"
    assert cfg.decision_time_s == 11.92
    assert cfg.calibration_path == Path("data/calibration/episode_2500_2525.json")
    assert cfg.keeper_kit() is not None
    assert cfg.keeper_kit().is_keeper is True


def test_episode_7818_7828_config():
    cfg = episode_7818_7828_config()
    assert isinstance(cfg, MatchConfig)
    assert cfg.match_id == "torino_milan_7818_7828"
    assert cfg.possession_team_id == "milan"
    assert cfg.attack_dir_x == 1.0
    assert cfg.decision_time_s == 2.80
    assert cfg.calibration_path == Path("data/calibration/episode_7818_7828.json")
    assert cfg.keeper_kit() is not None


def test_get_match_config_resolver():
    cfg1 = get_match_config("episode21")
    assert cfg1.match_id == "torino_milan_0708_0720"

    cfg2 = get_match_config("episode_2500_2525")
    assert cfg2.match_id == "torino_milan_2500_2525"

    cfg3 = get_match_config("clip3")
    assert cfg3.match_id == "torino_milan_7818_7828"

    with pytest.raises(ValueError, match="Unknown match configuration"):
        get_match_config("invalid_match_xyz")
