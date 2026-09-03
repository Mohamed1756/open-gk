# GK Decision-Value Engine

Early-stage exploration of goalkeeper distribution decision-making using computer vision.

## Images

![Tracking & bounding boxes](tracking_bounding_boxes.png)

![Keeper optimal distribution path](keeper_optimal_distribution_path.png)

## The idea

Most GK stats (distribution % or pass completion) don't account for context—specifically, the positioning and intensity of the press. A perfectly safe sideways pass looks the same as a risky penetrating one. **The question:** Can we score a keeper's decision-making against the actual problem they face—the press—and separate "good at distribution" from "positioned well"?

## What's here

- **Video-based CV pipeline**: YOLO detection, multi-object tracking, pose estimation
- **Press modeling**: detects pressuring players and their geometry relative to the ball carrier
- **Decision scoring**: evaluates keeper actions against press profile and biomechanics
- **Interactive dashboard**: HTML report with video sync and multi-episode switching

## Quick Start

```bash
clone this repo
pip install -r requirements.txt
python3 scripts/evaluate_distribution.py --episode all
# Open reports/goalkeeper_distribution_valuation.html
```

## Status

**Phase 0.** Proof-of-concept. Code is exploratory and rough. The goal is to validate whether the signal exists—can we measure keeper decision-making against the press?


