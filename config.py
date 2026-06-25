"""
config.py
=========
Central configuration for the AI Cricket Intelligence System.

Every threshold, weight, and tunable parameter used anywhere in the analytics
lives here. This is deliberate: the system promises *explainable* output, so we
never want a "magic number" buried in a function. If a turning point is flagged
because an over went for 15+ runs, the reader can come here and see exactly what
"high-scoring over" means and change it.

Nothing in this file invents cricket data. These are only the rules that decide
how *already-parsed* facts are interpreted (e.g. what counts as a "collapse").
"""

from dataclasses import dataclass, field
from typing import Dict


# --------------------------------------------------------------------------- #
# Match format detection
# --------------------------------------------------------------------------- #
# The format is inferred from the highest over number seen in the commentary.
# We never assume a format the data does not support.
FORMAT_OVERS = {
    "T20": 20,
    "ODI": 50,
    "TEST": None,  # unbounded; phases are not applied
}


@dataclass(frozen=True)
class PhaseConfig:
    """Over ranges (0-indexed over numbers, inclusive) for each scoring phase."""
    powerplay: tuple
    middle: tuple
    death: tuple


# Phase definitions per format. Over numbers are 0-indexed to match the parser
# (over "0.1" => over_num 0). T20 powerplay = overs 0-5 (the first 6 overs).
PHASE_DEFINITIONS: Dict[str, PhaseConfig] = {
    "T20": PhaseConfig(powerplay=(0, 5), middle=(6, 14), death=(15, 19)),
    "ODI": PhaseConfig(powerplay=(0, 9), middle=(10, 39), death=(40, 49)),
}


# --------------------------------------------------------------------------- #
# Event / turning-point detection thresholds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EventThresholds:
    high_scoring_over_runs: int = 15      # an over conceding >= this is "expensive"
    big_partnership_runs: int = 40        # partnership >= this is "significant"
    collapse_wickets: int = 3             # this many wickets...
    collapse_within_balls: int = 18       # ...inside this many legal balls = collapse
    milestone_steps: tuple = (50, 100, 150, 200)  # individual batting milestones
    run_rate_swing_delta: float = 1.5     # change in cumulative RR considered a "swing"


EVENT_THRESHOLDS = EventThresholds()


# --------------------------------------------------------------------------- #
# Impact-player scoring weights  (Module 8)
# --------------------------------------------------------------------------- #
# The impact score is a transparent weighted sum, normalised to 0-100 across the
# players in the match. The full formula is documented in src/analytics/impact.py
# and surfaced verbatim in the UI. These weights are the ONLY tunable inputs.
@dataclass(frozen=True)
class ImpactWeights:
    # Batting components
    run_value: float = 1.0            # points per run scored
    boundary_bonus: float = 1.5       # extra points per boundary (four or six)
    six_extra_bonus: float = 1.0      # additional bonus per six on top of boundary
    strike_rate_pivot: float = 130.0  # SR above this is rewarded, below is penalised
    strike_rate_value: float = 0.20   # points per (SR - pivot) point, * balls/100 scale

    # Bowling components
    wicket_value: float = 20.0        # points per wicket taken (bowler-credited)
    economy_pivot: float = 8.0        # economy below this is rewarded
    economy_value: float = 4.0        # points per (pivot - economy) run, * overs scale
    maiden_bonus: float = 5.0         # points per maiden over

    # Fielding components
    catch_value: float = 8.0          # points per catch
    runout_value: float = 10.0        # points per run-out involvement
    stumping_value: float = 9.0       # points per stumping

    min_balls_for_sr: int = 6         # ignore strike-rate term below this many balls
    min_overs_for_econ: float = 1.0   # ignore economy term below this many overs


IMPACT_WEIGHTS = ImpactWeights()


# --------------------------------------------------------------------------- #
# Sentiment / momentum (Module 10)
# --------------------------------------------------------------------------- #
# Rule-based event sentiment from the *batting side's* perspective.
# Positive = good for the batting team. These are used for the momentum curve
# and as a deterministic fallback when transformer sentiment is unavailable.
EVENT_SENTIMENT = {
    "SIX": 1.0,
    "FOUR": 0.7,
    "WICKET": -1.0,
    "DOT": -0.15,
    "SINGLE": 0.1,
    "DOUBLE": 0.25,
    "TRIPLE": 0.4,
    "WIDE": 0.2,
    "NO_BALL": 0.3,
    "BYES": 0.1,
    "LEG_BYES": 0.1,
    "NEUTRAL": 0.0,
}

# Window (in legal balls) for the rolling momentum calculation.
MOMENTUM_WINDOW_BALLS = 12


# --------------------------------------------------------------------------- #
# NLP model configuration (all optional; system degrades gracefully)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NLPConfig:
    spacy_model: str = "en_core_web_sm"
    summarizer_model: str = "sshleifer/distilbart-cnn-12-6"  # lightweight BART
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    summarizer_max_len: int = 220
    summarizer_min_len: int = 60
    enable_transformers: bool = True   # if libs/models missing -> auto fallback


NLP_CONFIG = NLPConfig()


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
DATABASE_PATH = "cricket_intelligence.db"

# The exact string shown whenever a value cannot be derived from commentary.
UNKNOWN = "Cannot determine from provided commentary."
