"""First-touch anchor: the kick happens at the ball, not the body."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

XY_M = Tuple[float, float]


@dataclass(frozen=True)
class FirstTouchResult:
    frame_offset: int
    ball_xy_m: XY_M
    gap_m: float
    gap_at_decision_m: Optional[float]


def find_first_touch(
    ball_xy: List[Optional[XY_M]],
    passer_xy: List[Optional[XY_M]],
    decision_offset: int,
) -> Optional[FirstTouchResult]:
    best: Optional[Tuple[int, float]] = None
    gap_dec: Optional[float] = None
    for i, (b, g) in enumerate(zip(ball_xy, passer_xy)):
        if b is None or g is None:
            continue
        gap = math.hypot(b[0] - g[0], b[1] - g[1])
        if i == decision_offset:
            gap_dec = round(gap, 2)
        if best is None or gap < best[1]:
            best = (i, gap)
    if best is None:
        return None
    i, gap = best
    ball = ball_xy[i]
    assert ball is not None
    return FirstTouchResult(
        frame_offset=i,
        ball_xy_m=(round(ball[0], 2), round(ball[1], 2)),
        gap_m=round(gap, 2),
        gap_at_decision_m=gap_dec,
    )


def decision_possession_frames(
    gaps_m: List[Optional[float]],
    gap_threshold_m: float = 3.0,
    pre_frames: int = 12,
    post_frames: int = 50,
) -> List[int]:
    hold: set[int] = set()
    n = len(gaps_m)
    for i, gap in enumerate(gaps_m):
        if gap is not None and gap <= gap_threshold_m:
            for f in range(i - pre_frames, i + post_frames + 1):
                if 0 <= f < n:
                    hold.add(f)
    return sorted(hold)
