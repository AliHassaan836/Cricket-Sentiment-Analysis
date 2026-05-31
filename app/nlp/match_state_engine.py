"""
Match State Engine
==================
Maintains authoritative match state across an innings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from copy import deepcopy

from app.nlp.commentary_parser import ParsedBall, get_phase  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class LiveBatterState:
    name: str
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = False
    dismissal_text: str = ""

    @property
    def strike_rate(self) -> float:
        return round((self.runs / self.balls * 100), 2) if self.balls > 0 else 0.0


@dataclass
class LiveBowlerState:
    name: str
    balls: int = 0
    runs: int = 0
    wickets: int = 0
    wides: int = 0
    no_balls: int = 0
    dot_balls: int = 0

    @property
    def overs(self) -> str:
        return f"{self.balls // 6}.{self.balls % 6}"

    @property
    def economy(self) -> float:
        overs_f = self.balls / 6
        return round(self.runs / overs_f, 2) if overs_f > 0 else 0.0


@dataclass
class LivePartnership:
    batsman1: str
    batsman2: str
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    start_over: str = ""
    wicket_number: int = 0

    @property
    def run_rate(self) -> float:
        overs = self.balls / 6
        return round(self.runs / overs, 2) if overs > 0 else 0.0


@dataclass
class OverRecord:
    over_number: int
    bowler: str
    balls: List[str] = field(default_factory=list)
    runs: int = 0
    wickets: int = 0
    phase: str = "Powerplay"

    @property
    def run_rate(self) -> float:
        return round(self.runs * 6 / max(len(self.balls), 1), 2)


class MatchStateEngine:

    def __init__(
        self,
        batting_team: str,
        bowling_team: str,
        innings_number: int = 1,
        target: Optional[int] = None,
    ):
        self.batting_team = batting_team
        self.bowling_team = bowling_team
        self.innings_number = innings_number
        self.target = target

        self.total_runs: int = 0
        self.wickets: int = 0
        self.total_balls: int = 0
        self.extras: Dict[str, int] = {
            "wides": 0, "no_balls": 0, "byes": 0, "leg_byes": 0, "penalty": 0
        }

        self.striker: Optional[str] = None
        self.non_striker: Optional[str] = None
        self.current_bowler: Optional[str] = None
        self.batters: Dict[str, LiveBatterState] = {}
        self.bowlers: Dict[str, LiveBowlerState] = {}

        self.partnerships: List[LivePartnership] = []
        self.current_partnership: Optional[LivePartnership] = None

        self.over_history: List[OverRecord] = []
        self._current_over: Optional[OverRecord] = None

        self.phase_scores: Dict[str, int] = {"Powerplay": 0, "Middle": 0, "Death": 0}
        self.phase_wickets: Dict[str, int] = {"Powerplay": 0, "Middle": 0, "Death": 0}

        self.milestones: List[Dict] = []

    def process_ball(self, ball: ParsedBall) -> None:
        self._ensure_over(ball.over, ball.bowler)
        phase = get_phase(ball.over)

        # ── Register new batsman ──────────────────────────────
        if ball.batsman:
            self._get_batter(ball.batsman)
            # First ball of innings
            if self.striker is None:
                self.striker = ball.batsman
            # New batsman coming in (not yet seen)
            elif ball.batsman not in (self.striker, self.non_striker):
                # They are the incoming batsman — set as striker
                self.striker = ball.batsman
            # Set non-striker if still unknown
            if self.non_striker is None and ball.batsman != self.striker:
                self.non_striker = ball.batsman

        # ── Score update ──────────────────────────────────────
        if ball.is_extra:
            extra_runs = ball.total_extras or 1
            self.total_runs += extra_runs
            self.phase_scores[phase] += extra_runs
            et = ball.extra_type or "wide"
            key = {"wide": "wides", "no_ball": "no_balls",
                   "bye": "byes", "leg_bye": "leg_byes"}.get(et, "wides")
            self.extras[key] += extra_runs
            bname = ball.bowler or self.current_bowler
            if bname:
                bw = self._get_bowler(bname)
                bw.runs += extra_runs
                if et == "wide":
                    bw.wides += 1
                elif et == "no_ball":
                    bw.no_balls += 1
        else:
            self.total_runs += ball.runs
            self.phase_scores[phase] += ball.runs
            self.total_balls += 1

            if ball.batsman:
                bt = self._get_batter(ball.batsman)
                bt.runs += ball.runs
                bt.balls += 1
                if ball.boundary_type == "4":
                    bt.fours += 1
                elif ball.boundary_type == "6":
                    bt.sixes += 1

            if ball.bowler:
                bw = self._get_bowler(ball.bowler)
                bw.balls += 1
                bw.runs += ball.runs
                if ball.runs == 0 and not ball.is_wicket:
                    bw.dot_balls += 1

        # ── Partnership update (before wicket processing) ─────
        if not ball.is_extra and not ball.is_wicket:
            self._update_partnership(ball)
            # Rotate strike on odd runs
            if ball.runs % 2 == 1:
                self.striker, self.non_striker = self.non_striker, self.striker

        # ── Wicket ────────────────────────────────────────────
        if ball.is_wicket:
            self.wickets += 1
            self.phase_wickets[phase] = self.phase_wickets.get(phase, 0) + 1

            if ball.bowler:
                self._get_bowler(ball.bowler).wickets += 1

            dismissed = ball.dismissed_batsman or ball.batsman
            if dismissed:
                bt = self._get_batter(dismissed)
                bt.is_out = True
                bt.dismissal_text = ball.raw_commentary[:80]

            # Close current partnership
            if self.current_partnership:
                self.current_partnership.wicket_number = self.wickets
                self.partnerships.append(deepcopy(self.current_partnership))
                self.current_partnership = None

            # Clear the dismissed batsman from striker/non-striker
            if self.striker == dismissed:
                self.striker = None
            elif self.non_striker == dismissed:
                self.non_striker = None

        # ── Over record ───────────────────────────────────────
        if self._current_over is not None:
            ball_char = (
                "W" if ball.is_wicket else
                str(ball.boundary_type) if ball.is_boundary else
                "Wd" if ball.extra_type == "wide" else
                "Nb" if ball.extra_type == "no_ball" else
                str(ball.runs)
            )
            self._current_over.balls.append(ball_char)
            if not ball.is_extra:
                self._current_over.runs += ball.runs
            else:
                self._current_over.runs += ball.total_extras or 1
            if ball.is_wicket:
                self._current_over.wickets += 1

        # ── Milestones ────────────────────────────────────────
        if ball.is_milestone and ball.milestone_description:
            self.milestones.append({
                "over": ball.over_ball,
                "description": ball.milestone_description,
                "player": ball.batsman or ball.bowler,
            })

        # ── Sync bowler ───────────────────────────────────────
        if ball.bowler:
            self.current_bowler = ball.bowler

    def snapshot(self) -> dict:
        # Include current (incomplete) over in history
        all_overs = list(self.over_history)
        if self._current_over is not None:
            all_overs.append(self._current_over)

        return {
            "innings_number": self.innings_number,
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "total_runs": self.total_runs,
            "wickets": self.wickets,
            "overs": self.total_balls // 6,
            "balls_in_over": self.total_balls % 6,
            "total_balls": self.total_balls,
            "run_rate": self._run_rate(),
            "required_run_rate": self._rrr(),
            "target": self.target,
            "striker": self.striker,
            "non_striker": self.non_striker,
            "current_bowler": self.current_bowler,
            "extras": self.extras,
            "phase_scores": self.phase_scores,
            "phase_wickets": self.phase_wickets,
            "milestones": self.milestones,
            "batting_stats": {
                k: {
                    "player": v.name, "runs": v.runs, "balls_faced": v.balls,
                    "fours": v.fours, "sixes": v.sixes,
                    "strike_rate": v.strike_rate, "is_out": v.is_out,
                    "dismissal": v.dismissal_text or None,
                }
                for k, v in self.batters.items()
            },
            "bowling_stats": {
                k: {
                    "player": v.name, "overs_bowled": v.overs,
                    "balls_bowled": v.balls, "runs_conceded": v.runs,
                    "wickets": v.wickets, "economy": v.economy,
                    "dot_balls": v.dot_balls, "wides": v.wides, "no_balls": v.no_balls,
                }
                for k, v in self.bowlers.items()
            },
            "partnerships": [
                {
                    "wicket_number": p.wicket_number,
                    "batsman1": p.batsman1, "batsman2": p.batsman2,
                    "runs": p.runs, "balls": p.balls,
                    "run_rate": p.run_rate, "fours": p.fours, "sixes": p.sixes,
                }
                for p in self.partnerships
            ],
            "current_partnership": (
                {
                    "batsman1": self.current_partnership.batsman1,
                    "batsman2": self.current_partnership.batsman2,
                    "runs": self.current_partnership.runs,
                    "balls": self.current_partnership.balls,
                    "run_rate": self.current_partnership.run_rate,
                }
                if self.current_partnership else None
            ),
            "over_history": [
                {
                    "over_number": o.over_number,
                    "bowler": o.bowler,
                    "balls": o.balls,
                    "runs": o.runs,
                    "wickets": o.wickets,
                    "phase": o.phase,
                    "run_rate": o.run_rate,
                }
                for o in all_overs
            ],
        }

    def _run_rate(self) -> float:
        overs = self.total_balls / 6
        return round(self.total_runs / overs, 2) if overs > 0 else 0.0

    def _rrr(self) -> Optional[float]:
        if self.target is None:
            return None
        overs_left = 20 - (self.total_balls / 6)
        runs_needed = self.target - self.total_runs
        return round(runs_needed / overs_left, 2) if overs_left > 0 else None

    def _get_batter(self, name: str) -> LiveBatterState:
        if name not in self.batters:
            self.batters[name] = LiveBatterState(name=name)
        return self.batters[name]

    def _get_bowler(self, name: str) -> LiveBowlerState:
        if name not in self.bowlers:
            self.bowlers[name] = LiveBowlerState(name=name)
        return self.bowlers[name]

    def _ensure_over(self, over: int, bowler: Optional[str]) -> None:
        if self._current_over is None or self._current_over.over_number != over:
            if self._current_over is not None:
                self.over_history.append(deepcopy(self._current_over))
            self._current_over = OverRecord(
                over_number=over,
                bowler=bowler or self.current_bowler or "Unknown",
                phase=get_phase(over),
            )
            # Rotate strike at start of new over
            if self.striker and self.non_striker:
                self.striker, self.non_striker = self.non_striker, self.striker

    def _update_partnership(self, ball: ParsedBall) -> None:
        # Start a new partnership if none exists and we have two batsmen
        if self.current_partnership is None:
            b1 = self.striker or ball.batsman or "Unknown"
            b2 = self.non_striker or "Unknown"
            # Don't create a partnership with placeholder names if we only know one batsman
            if b1 == "Unknown" and b2 == "Unknown":
                return
            self.current_partnership = LivePartnership(
                batsman1=b1,
                batsman2=b2,
                start_over=ball.over_ball,
                wicket_number=self.wickets + 1,
            )

        self.current_partnership.runs += ball.runs
        self.current_partnership.balls += 1
        if ball.boundary_type == "4":
            self.current_partnership.fours += 1
        elif ball.boundary_type == "6":
            self.current_partnership.sixes += 1


def build_match_state(
    events: List[ParsedBall],
    batting_team: str,
    bowling_team: str,
    innings_number: int = 1,
    target: Optional[int] = None,
) -> MatchStateEngine:
    engine = MatchStateEngine(
        batting_team=batting_team,
        bowling_team=bowling_team,
        innings_number=innings_number,
        target=target,
    )
    for ball in events:
        try:
            engine.process_ball(ball)
        except Exception as e:
            logger.warning(f"Error processing ball {ball.over_ball}: {e}")
    return engine
