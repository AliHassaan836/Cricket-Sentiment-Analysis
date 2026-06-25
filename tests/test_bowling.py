"""
tests/test_bowling.py
=====================
Unit tests for bowling analytics (Module 4).

The fixture deliberately exercises the tricky conventions:
  * a maiden over (six legal dot balls, no runs);
  * a caught wicket, which IS credited to the bowler;
  * a run-out, which is NOT credited to the bowler;
  * a wide and a no-ball, which add to the bowler's runs but are not legal balls.

All expected figures are hand-computable.

Run with:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.commentary_parser import parse_commentary
from src.analytics.bowling import bowling_scorecard, over_by_over


FIXTURE = """0.1 Bumrah to Warner, no run
0.2 Bumrah to Warner, no run
0.3 Bumrah to Warner, no run
0.4 Bumrah to Warner, no run
0.5 Bumrah to Warner, no run
0.6 Bumrah to Warner, no run
1.1 Shami to Smith, OUT, caught by Kohli
1.2 Shami to Warner, wide
1.2 Shami to Warner, 1 run
1.3 Shami to Warner, OUT, run out
1.4 Shami to Maxwell, no ball
1.4 Shami to Maxwell, SIX"""


def _card():
    df = parse_commentary(FIXTURE)
    return bowling_scorecard(df).set_index("Bowler")


def test_maiden_over():
    b = _card().loc["Bumrah"]
    assert b["Balls"] == 6
    assert b["Runs"] == 0
    assert b["Maidens"] == 1
    assert b["Economy"] == 0.0
    assert b["Dot %"] == 100.0


def test_runout_not_credited_to_bowler():
    s = _card().loc["Shami"]
    # Only the caught dismissal counts; the run-out does not.
    assert s["Wickets"] == 1


def test_bowler_runs_include_wide_and_no_ball():
    s = _card().loc["Shami"]
    # 0 (caught) + 1 (wide) + 1 (single) + 0 (run out) + 1 (no ball) + 6 (six)
    assert s["Runs"] == 9


def test_legal_balls_exclude_wide_and_no_ball():
    s = _card().loc["Shami"]
    # Legal deliveries: 1.1, the single, the run-out ball, the six = 4.
    assert s["Balls"] == 4


def test_economy_calculation():
    s = _card().loc["Shami"]
    # 9 runs over 4 legal balls = 9 / (4/6) = 13.5 runs per over.
    assert round(s["Economy"], 2) == 13.5


def test_over_by_over_runs_sum():
    df = parse_commentary(FIXTURE)
    obo = over_by_over(df)
    # Total runs across overs equals the bowlers' conceded runs (9).
    assert int(obo["runs"].sum()) == 9


def test_empty_input_is_safe():
    import pandas as pd
    card = bowling_scorecard(parse_commentary(""))
    assert isinstance(card, pd.DataFrame)
    assert card.empty
