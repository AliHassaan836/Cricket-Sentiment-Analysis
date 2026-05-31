"""
Pydantic schemas for Cricket Match Intelligence System.
All request/response models and internal data structures.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class EventType(str, Enum):
    NORMAL_RUN    = "NORMAL_RUN"
    DOT_BALL      = "DOT_BALL"
    BOUNDARY_4    = "BOUNDARY_4"
    BOUNDARY_6    = "BOUNDARY_6"
    WICKET        = "WICKET"
    EXTRA         = "EXTRA"
    MILESTONE     = "MILESTONE"
    BOWLING_CHANGE = "BOWLING_CHANGE"

class WicketType(str, Enum):
    BOWLED      = "Bowled"
    CAUGHT      = "Caught"
    LBW         = "LBW"
    RUN_OUT     = "Run Out"
    STUMPED     = "Stumped"
    HIT_WICKET  = "Hit Wicket"
    OBSTRUCTING = "Obstructing the Field"
    UNKNOWN     = "Unknown"

class ExtraType(str, Enum):
    WIDE     = "wide"
    NO_BALL  = "no_ball"
    BYE      = "bye"
    LEG_BYE  = "leg_bye"
    PENALTY  = "penalty"

class InningsPhase(str, Enum):
    POWERPLAY   = "Powerplay"    # 0–6
    MIDDLE      = "Middle"       # 7–15
    DEATH       = "Death"        # 16–20

class ImpactLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    CRITICAL = "critical"


# ──────────────────────────────────────────────
# Core Event Model
# ──────────────────────────────────────────────

class BallEvent(BaseModel):
    """Structured representation of a single delivery."""
    over: int
    ball: int
    over_ball: str                        # e.g. "3.4"
    raw_commentary: str

    event_type: EventType = EventType.NORMAL_RUN
    runs: int = 0
    total_extras: int = 0

    batsman: Optional[str] = None
    bowler: Optional[str] = None
    fielder: Optional[str] = None

    # Wicket details
    is_wicket: bool = False
    wicket_type: Optional[WicketType] = None
    dismissed_batsman: Optional[str] = None

    # Boundary details
    is_boundary: bool = False
    boundary_type: Optional[str] = None   # "4" or "6"

    # Extra details
    is_extra: bool = False
    extra_type: Optional[ExtraType] = None

    # Milestone
    is_milestone: bool = False
    milestone_description: Optional[str] = None

    # Analytics
    phase: Optional[InningsPhase] = None
    cumulative_score: int = 0
    run_rate_at_ball: float = 0.0

    # NLP confidence
    parse_confidence: float = 1.0
    entities_extracted: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Player Statistics
# ──────────────────────────────────────────────

class BattingStats(BaseModel):
    player: str
    runs: int = 0
    balls_faced: int = 0
    fours: int = 0
    sixes: int = 0
    strike_rate: float = 0.0
    is_out: bool = False
    dismissal: Optional[str] = None
    phase_runs: Dict[str, int] = Field(default_factory=lambda: {"Powerplay": 0, "Middle": 0, "Death": 0})

    @field_validator("strike_rate", mode="before")
    @classmethod
    def compute_sr(cls, v, info):
        data = info.data
        balls = data.get("balls_faced", 0)
        runs = data.get("runs", 0)
        if balls > 0:
            return round((runs / balls) * 100, 2)
        return 0.0

class BowlingStats(BaseModel):
    player: str
    overs_bowled: float = 0.0
    balls_bowled: int = 0
    runs_conceded: int = 0
    wickets: int = 0
    economy: float = 0.0
    dot_balls: int = 0
    wides: int = 0
    no_balls: int = 0
    phase_economy: Dict[str, float] = Field(default_factory=dict)

class Partnership(BaseModel):
    wicket_number: int
    batsman1: str
    batsman2: str
    runs: int = 0
    balls: int = 0
    run_rate: float = 0.0
    fours: int = 0
    sixes: int = 0
    start_over: str = ""
    end_over: str = ""
    broken_by: Optional[str] = None     # bowler who ended partnership


# ──────────────────────────────────────────────
# Match State
# ──────────────────────────────────────────────

class OverSummary(BaseModel):
    over_number: int
    runs: int
    wickets: int
    balls: List[str]         # compact ball-by-ball: ["4","W","1","0","6","1"]
    bowler: str
    run_rate: float
    phase: InningsPhase

class InningsState(BaseModel):
    innings_number: int = 1
    batting_team: str
    bowling_team: str

    total_runs: int = 0
    wickets: int = 0
    overs: int = 0
    balls_in_over: int = 0
    total_balls: int = 0

    run_rate: float = 0.0
    required_run_rate: Optional[float] = None
    target: Optional[int] = None

    striker: Optional[str] = None
    non_striker: Optional[str] = None
    current_bowler: Optional[str] = None

    batting_stats: Dict[str, BattingStats] = Field(default_factory=dict)
    bowling_stats: Dict[str, BowlingStats] = Field(default_factory=dict)
    partnerships: List[Partnership] = Field(default_factory=list)
    current_partnership: Optional[Partnership] = None
    over_history: List[OverSummary] = Field(default_factory=list)

    extras: Dict[str, int] = Field(default_factory=lambda: {
        "wides": 0, "no_balls": 0, "byes": 0, "leg_byes": 0, "penalty": 0
    })
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    phase_scores: Dict[str, int] = Field(default_factory=lambda: {
        "Powerplay": 0, "Middle": 0, "Death": 0
    })


# ──────────────────────────────────────────────
# Turning Points
# ──────────────────────────────────────────────

class TurningPoint(BaseModel):
    over_ball: str
    title: str
    description: str
    evidence: List[str]
    impact_score: float = Field(ge=0, le=100)
    impact_level: ImpactLevel
    momentum_before: float
    momentum_after: float
    momentum_shift: float
    event_type: str
    affected_team: Optional[str] = None

class MomentumPoint(BaseModel):
    over_ball: str
    momentum: float
    cumulative_runs: int
    wickets: int
    phase: str


# ──────────────────────────────────────────────
# Analytics
# ──────────────────────────────────────────────

class InningsAnalytics(BaseModel):
    innings_number: int
    batting_team: str
    final_score: str          # "187/4"
    top_scorer: Optional[str] = None
    top_scorer_runs: int = 0
    best_bowler: Optional[str] = None
    best_bowling_figures: str = ""
    total_boundaries: int = 0
    total_sixes: int = 0
    dot_ball_percentage: float = 0.0
    boundary_percentage: float = 0.0
    powerplay_score: str = ""
    middle_overs_score: str = ""
    death_overs_score: str = ""
    highest_partnership: int = 0
    turning_points: List[TurningPoint] = Field(default_factory=list)
    momentum_data: List[MomentumPoint] = Field(default_factory=list)

class MatchAnalytics(BaseModel):
    match_id: Optional[str] = None
    team1: str
    team2: str
    venue: Optional[str] = None
    innings: List[InningsAnalytics] = Field(default_factory=list)
    result: Optional[str] = None
    player_of_match: Optional[str] = None


# ──────────────────────────────────────────────
# API Request / Response Models
# ──────────────────────────────────────────────

class CommentaryInput(BaseModel):
    commentary: str = Field(..., min_length=10, description="Raw ball-by-ball commentary text")
    team1: str = Field(default="Team A")
    team2: str = Field(default="Team B")
    innings: int = Field(default=1, ge=1, le=4)
    target: Optional[int] = None
    match_id: Optional[str] = None

class ParsedCommentaryResponse(BaseModel):
    success: bool
    match_id: str
    innings: int
    events_parsed: int
    match_state: InningsState
    events: List[BallEvent]
    validation_report: "ValidationReport"
    parse_time_ms: float

class ValidationReport(BaseModel):
    is_valid: bool
    score_consistent: bool
    wickets_consistent: bool
    overs_consistent: bool
    player_consistency: bool
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

class AnalyticsResponse(BaseModel):
    match_id: str
    analytics: MatchAnalytics
    turning_points: List[TurningPoint]
    momentum_timeline: List[MomentumPoint]

class SummaryRequest(BaseModel):
    match_id: str
    summary_type: str = Field(default="detailed", pattern="^(short|detailed|headline)$")
    include_turning_points: bool = True
    include_player_highlights: bool = True

class SummaryResponse(BaseModel):
    match_id: str
    summary_type: str
    headline: str
    short_summary: str
    detailed_report: Optional[str] = None
    key_moments: List[str] = Field(default_factory=list)
    validated: bool = True

class QARequest(BaseModel):
    match_id: str
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class QAResponse(BaseModel):
    match_id: str
    question: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float
    grounded: bool = True
    retrieved_context: List[str] = Field(default_factory=list)

# Resolve forward reference
ParsedCommentaryResponse.model_rebuild()
