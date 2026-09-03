"""Transferable per-match configuration (no team/kit/video literals in code)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KitSpec:
    team_id: str
    label: str
    color_hex: str
    is_keeper: bool = False
    is_official: bool = False


@dataclass(frozen=True)
class MatchConfig:
    match_id: str
    video_path: Path
    possession_team_id: str
    keeper_name: str = "GK"
    keeper_team_id: str = ""
    attack_dir_x: float = 1.0
    fps: float = 25.0
    decision_time_s: float = 4.0
    out_html: Path = Path("reports/distribution_valuation.html")
    calibration_path: Path = Path("data/calibration/episode21.json")
    kits: tuple[KitSpec, ...] = ()
    title: str = ""
    tactical_context: str = ""

    def canonical_x(self, x_m: float) -> float:
        return float(x_m) if self.attack_dir_x >= 0 else 105.0 - float(x_m)

    def to_canonical(self, x_m: float, y_m: float) -> tuple[float, float]:
        if self.attack_dir_x >= 0:
            return float(x_m), float(y_m)
        return 105.0 - float(x_m), float(y_m)

    def team_ids(self) -> tuple[str, ...]:
        return tuple(k.team_id for k in self.kits if not k.is_official)

    def keeper_kit(self) -> KitSpec | None:
        for k in self.kits:
            if k.is_keeper and (
                not self.keeper_team_id or k.team_id == self.keeper_team_id
            ):
                return k
        for k in self.kits:
            if k.is_keeper:
                return k
        return None


TORINO_MILAN_KITS: tuple[KitSpec, ...] = (
    KitSpec("milan", "AC Milan (White)", "#ffffff"),
    KitSpec("torino", "Torino (Maroon)", "#881337"),
    KitSpec("milan_gk", "AC Milan GK (Maignan - Purple)", "#c026d3", is_keeper=True),
    KitSpec("opp_gk", "Goalkeeper (Yellow)", "#facc15", is_keeper=True),
    KitSpec("official", "Match Official (Cyan)", "#06b6d4", is_official=True),
    KitSpec("unknown", "Unknown-Outfield", "#94a3b8"),
)

DEFAULT_EPISODE21_KITS = TORINO_MILAN_KITS


def episode21_config() -> MatchConfig:
    clean_video = Path("data/video_raw/clips/clip_maignan_pass_0708_0720_clean.mp4")
    raw_video = Path("data/video_raw/clips/clip_maignan_pass_0708_0720.mp4")
    video_path = clean_video if clean_video.exists() else raw_video
    return MatchConfig(
        match_id="torino_milan_0708_0720",
        video_path=video_path,
        possession_team_id="milan",
        keeper_name="Mike Maignan",
        keeper_team_id="milan_gk",
        attack_dir_x=-1.0,
        out_html=Path("reports/goalkeeper_distribution_valuation.html"),
        calibration_path=Path("data/calibration/episode21.json"),
        kits=TORINO_MILAN_KITS,
        title="Match Clock: 07:08 - 07:20",
        tactical_context="Torino Medium-High Press",
    )


def episode_2500_2525_config() -> MatchConfig:
    clean_video = Path("data/video_raw/clips/clip_maignan_pass_2500_2525_clean.mp4")
    return MatchConfig(
        match_id="torino_milan_2500_2525",
        video_path=clean_video,
        possession_team_id="milan",
        keeper_name="Mike Maignan",
        keeper_team_id="milan_gk",
        attack_dir_x=-1.0,
        decision_time_s=11.92,
        out_html=Path("reports/goalkeeper_distribution_valuation_2500_2525.html"),
        calibration_path=Path("data/calibration/episode_2500_2525.json"),
        kits=TORINO_MILAN_KITS,
        title="Match Clock: 25:00 - 25:25",
        tactical_context="Torino Aggressive High Press",
    )


def episode_7818_7828_config() -> MatchConfig:
    clean_video = Path("data/video_raw/clips/clip_maignan_pass_7818_7828_clean.mp4")
    return MatchConfig(
        match_id="torino_milan_7818_7828",
        video_path=clean_video,
        possession_team_id="milan",
        keeper_name="Mike Maignan",
        keeper_team_id="milan_gk",
        attack_dir_x=1.0,
        decision_time_s=2.80,
        out_html=Path("reports/goalkeeper_distribution_valuation_7818_7828.html"),
        calibration_path=Path("data/calibration/episode_7818_7828.json"),
        kits=TORINO_MILAN_KITS,
        title="Match Clock: 78:18 - 78:28",
        tactical_context="Torino Desperation Press (Down 0-2)",
    )


def get_match_config(name_or_alias: str) -> MatchConfig:
    key = name_or_alias.lower().strip().replace("-", "_")
    if key in ("episode21", "0708_0720", "ep1", "episode_21"):
        return episode21_config()
    if key in ("episode_2500_2525", "2500_2525", "ep2", "episode2500_2525"):
        return episode_2500_2525_config()
    if key in ("episode_7818_7828", "7818_7828", "ep3", "episode7818_7828", "clip3"):
        return episode_7818_7828_config()
    raise ValueError(f"Unknown match configuration: {name_or_alias}")
