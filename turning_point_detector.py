"""
Turning Point Detection Engine
================================
Identifies high-impact match moments using a momentum-based algorithm.

Algorithm:
  - Momentum = Σ(weighted events) over a sliding window
  - Weights: boundary=+2, single=+0.3, dot=-0.1, wicket=-5, wide/nb=-0.2
  - Exponential decay factor (α=0.7) smooths momentum over time
  - Turning point = |Δmomentum| > threshold AND context confirms match impact
  - Minimum gap between turning points: 6 balls (anti-clustering)

Evidence generation uses template-based NLG with match context injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

from app.nlp.commentary_parser import ParsedBall

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

WINDOW_SIZE   = 6      # deliveries in sliding window
DECAY_ALPHA   = 0.7    # momentum decay (recent weight)
THRESHOLD     = 8.0    # min |Δmomentum| to flag turning point
MIN_GAP_BALLS = 6      # minimum balls between consecutive turning points

# Event momentum weights
WEIGHTS = {
    "BOUNDARY_6":   6.0,
    "BOUNDARY_4":   4.0,
    "NORMAL_RUN":   0.4,
    "DOT_BALL":    -0.3,
    "WICKET":      -5.0,
    "EXTRA":       -0.5,
    "MILESTONE":    3.0,
}


class ImpactLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class TurningPoint:
    over_ball: str
    title: str
    description: str
    evidence: List[str]
    impact_score: float           # 0–100
    impact_level: ImpactLevel
    momentum_before: float
    momentum_after: float
    momentum_shift: float
    event_type: str
    category: str                 # "batting_collapse", "boundary_burst", etc.
    affected_team: Optional[str] = None
    ball_index: int = 0


@dataclass
class MomentumPoint:
    over_ball: str
    ball_index: int
    momentum: float
    cumulative_runs: int
    wickets: int
    phase: str
    event_type: str


# ──────────────────────────────────────────────────────────
# Momentum calculator
# ──────────────────────────────────────────────────────────

def _event_momentum(ball: ParsedBall) -> float:
    """Calculate the instantaneous momentum contribution of a single ball."""
    base = WEIGHTS.get(ball.event_type, 0.2)
    if ball.is_milestone:
        base += 2.0
    if ball.is_wicket and ball.wicket_type in ("Caught", "Bowled"):
        base -= 1.0   # clean dismissal weighs heavier
    return base


def compute_momentum_timeline(events: List[ParsedBall]) -> List[MomentumPoint]:
    """
    Compute smoothed momentum for every delivery using
    exponential weighted moving average.
    """
    timeline: List[MomentumPoint] = []
    momentum = 0.0

    for i, ball in enumerate(events):
        instant = _event_momentum(ball)
        momentum = DECAY_ALPHA * instant + (1 - DECAY_ALPHA) * momentum

        timeline.append(MomentumPoint(
            over_ball=ball.over_ball,
            ball_index=i,
            momentum=round(momentum, 3),
            cumulative_runs=ball.cumulative_score,
            wickets=sum(1 for b in events[:i+1] if b.is_wicket),
            phase=ball.phase,
            event_type=ball.event_type,
        ))

    return timeline


# ──────────────────────────────────────────────────────────
# Turning point detector
# ──────────────────────────────────────────────────────────

class TurningPointDetector:
    """
    Detects high-impact match turning points from a sequence of ball events.

    Uses a two-pass algorithm:
      Pass 1 — compute momentum timeline
      Pass 2 — sliding window analysis to detect significant shifts
    """

    def __init__(
        self,
        batting_team: str = "Batting Team",
        bowling_team: str = "Bowling Team",
    ):
        self.batting_team = batting_team
        self.bowling_team = bowling_team

    def detect(self, events: List[ParsedBall]) -> Tuple[List[TurningPoint], List[MomentumPoint]]:
        """
        Returns:
            turning_points: ranked list of high-impact moments
            momentum_timeline: full per-ball momentum data
        """
        if len(events) < WINDOW_SIZE:
            return [], compute_momentum_timeline(events)

        timeline = compute_momentum_timeline(events)
        turning_points: List[TurningPoint] = []
        last_tp_ball = -MIN_GAP_BALLS - 1   # allow first detection

        for i in range(WINDOW_SIZE, len(events)):
            if i - last_tp_ball < MIN_GAP_BALLS:
                continue

            window_prev = events[i - WINDOW_SIZE : i - WINDOW_SIZE // 2]
            window_curr = events[i - WINDOW_SIZE // 2 : i]

            m_before = timeline[i - WINDOW_SIZE // 2 - 1].momentum
            m_after  = timeline[i - 1].momentum
            shift    = abs(m_after - m_before)

            if shift < THRESHOLD:
                continue

            tp = self._build_turning_point(
                anchor_ball=events[i - 1],
                prev_window=window_prev,
                curr_window=window_curr,
                momentum_before=m_before,
                momentum_after=m_after,
                momentum_shift=shift,
                ball_index=i,
            )

            if tp is not None:
                turning_points.append(tp)
                last_tp_ball = i

        # Sort by impact score descending
        turning_points.sort(key=lambda t: t.impact_score, reverse=True)
        logger.info(f"Detected {len(turning_points)} turning points")
        return turning_points, timeline

    # ----------------------------------------------------------
    # Turning point builder
    # ----------------------------------------------------------

    def _build_turning_point(
        self,
        anchor_ball: ParsedBall,
        prev_window: List[ParsedBall],
        curr_window: List[ParsedBall],
        momentum_before: float,
        momentum_after: float,
        momentum_shift: float,
        ball_index: int,
    ) -> Optional[TurningPoint]:

        prev_wickets  = sum(1 for b in prev_window if b.is_wicket)
        curr_wickets  = sum(1 for b in curr_window if b.is_wicket)
        prev_runs     = sum(b.runs for b in prev_window)
        curr_runs     = sum(b.runs for b in curr_window)
        prev_bounds   = sum(1 for b in prev_window if b.is_boundary)
        curr_bounds   = sum(1 for b in curr_window if b.is_boundary)
        prev_dots     = sum(1 for b in prev_window if b.event_type == "DOT_BALL")
        curr_dots     = sum(1 for b in curr_window if b.event_type == "DOT_BALL")

        # Classify the turning point category
        category, title, description, evidence, affected = self._classify_shift(
            prev_wickets, curr_wickets,
            prev_runs, curr_runs,
            prev_bounds, curr_bounds,
            prev_dots, curr_dots,
            anchor_ball, momentum_shift,
        )

        if category is None:
            return None

        raw_impact = min(100, momentum_shift * 4.5)
        impact_score = round(raw_impact, 1)
        impact_level = (
            ImpactLevel.CRITICAL if impact_score >= 80 else
            ImpactLevel.HIGH     if impact_score >= 60 else
            ImpactLevel.MEDIUM   if impact_score >= 35 else
            ImpactLevel.LOW
        )

        return TurningPoint(
            over_ball=anchor_ball.over_ball,
            title=title,
            description=description,
            evidence=evidence,
            impact_score=impact_score,
            impact_level=impact_level,
            momentum_before=round(momentum_before, 2),
            momentum_after=round(momentum_after, 2),
            momentum_shift=round(momentum_shift, 2),
            event_type=anchor_ball.event_type,
            category=category,
            affected_team=affected,
            ball_index=ball_index,
        )

    # ----------------------------------------------------------
    # Shift classification with NLG evidence
    # ----------------------------------------------------------

    def _classify_shift(
        self,
        prev_wkts, curr_wkts,
        prev_runs, curr_runs,
        prev_bounds, curr_bounds,
        prev_dots, curr_dots,
        anchor: ParsedBall,
        shift: float,
    ):
        n = WINDOW_SIZE // 2  # half-window size

        # ── Batting collapse ─────────────────────────────────
        if curr_wkts >= 2 and curr_runs < 8:
            title = f"Batting collapse — {curr_wkts} wickets in {n} balls"
            desc  = (
                f"{self.bowling_team} struck {curr_wkts} times in just {n} deliveries "
                f"conceding only {curr_runs} runs. {self.batting_team} lost momentum rapidly."
            )
            evidence = [
                f"{curr_wkts} wickets fell for {curr_runs} runs (balls {anchor.over_ball})",
                f"Run scoring rate dropped from {prev_runs}/{n} to {curr_runs}/{n} balls",
                f"Bowling team seized the initiative with pressure bowling",
            ]
            return "batting_collapse", title, desc, evidence, self.bowling_team

        # ── Wicket of key batsman ────────────────────────────
        if anchor.is_wicket and prev_runs >= 12:
            name = anchor.dismissed_batsman or "key batsman"
            title = f"Key wicket: {name} dismissed at {anchor.over_ball}"
            desc  = (
                f"{name} was dismissed after {self.batting_team} had scored {prev_runs} "
                f"in the previous {n} balls. A significant momentum check."
            )
            evidence = [
                f"{name} dismissed ({anchor.wicket_type or 'unknown mode'})",
                f"Batting team was in flow: {prev_runs} runs from last {n} balls",
                f"Wicket taken at over {anchor.over_ball} mid-acceleration phase",
            ]
            return "key_wicket", title, desc, evidence, self.bowling_team

        # ── Boundary burst ───────────────────────────────────
        if curr_bounds >= 3 and curr_runs >= 20:
            title = f"Boundary burst — {curr_bounds} boundaries in {n} balls"
            desc  = (
                f"{self.batting_team} exploded with {curr_bounds} boundaries, "
                f"scoring {curr_runs} off just {n} deliveries. Attack mode engaged."
            )
            evidence = [
                f"{curr_bounds} boundaries struck ({curr_runs} runs from {n} balls)",
                f"Run rate surged from {round(prev_runs*6/n,1)} to {round(curr_runs*6/n,1)}",
                f"Bowling team conceded {curr_bounds} boundary deliveries in quick succession",
            ]
            return "boundary_burst", title, desc, evidence, self.batting_team

        # ── Bowling stranglehold ─────────────────────────────
        if curr_dots >= 4 and curr_runs <= 4 and curr_wkts == 0:
            title = f"Bowling stranglehold at {anchor.over_ball}"
            desc  = (
                f"{self.bowling_team} bowled {curr_dots} dot balls in {n} deliveries, "
                f"allowing only {curr_runs} runs. Batters under pressure."
            )
            evidence = [
                f"{curr_dots}/{n} deliveries were dot balls",
                f"Only {curr_runs} runs conceded in the spell",
                f"Run rate suppressed from {round(prev_runs*6/n,1)} to {round(curr_runs*6/n,1)}",
            ]
            return "bowling_stranglehold", title, desc, evidence, self.bowling_team

        # ── Milestone momentum ───────────────────────────────
        if anchor.is_milestone and shift >= THRESHOLD:
            milestone = anchor.milestone_description or "milestone"
            title = f"Milestone lifts momentum: {milestone}"
            desc  = (
                f"The milestone at {anchor.over_ball} injected fresh momentum. "
                f"Scoring intensity changed significantly around this delivery."
            )
            evidence = [
                milestone,
                f"Momentum shift of {round(shift, 1)} detected",
                f"Run scoring: {prev_runs} (before) vs {curr_runs} (after)",
            ]
            return "milestone", title, desc, evidence, self.batting_team

        # ── Generic momentum swing ───────────────────────────
        if shift >= THRESHOLD * 1.5:
            direction = "batting" if momentum_after_gt_before(anchor) else "bowling"
            title = f"Momentum swing at {anchor.over_ball}"
            desc  = (
                f"A significant shift in match dynamics detected at over {anchor.over_ball}. "
                f"The {direction} side gained a notable advantage."
            )
            evidence = [
                f"Momentum shift magnitude: {round(shift, 1)}",
                f"Runs: {prev_runs} → {curr_runs} | Wickets: {prev_wkts} → {curr_wkts}",
                f"Boundaries: {prev_bounds} → {curr_bounds}",
            ]
            return "momentum_swing", title, desc, evidence, None

        return None, None, None, None, None


def momentum_after_gt_before(ball: ParsedBall) -> bool:
    return ball.event_type in ("BOUNDARY_4", "BOUNDARY_6", "NORMAL_RUN", "MILESTONE")
