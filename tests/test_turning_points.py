"""
tests/test_turning_points.py
============================
Unit tests for turning-point detection (Module 7).

The fixture contains three clearly distinct phases:
  * a steady opening over,
  * an explosive 30-run over (positive momentum swing),
  * an over with a key wicket followed by a run of dot balls (negative swing).

Rather than asserting on cosmetic labels, the tests pin down the contract the
UI and report layers rely on: turning points carry numeric run-rate context,
non-empty evidence drawn from the commentary, and are ordered by importance.

Run with:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.commentary_parser import parse_commentary
from src.analytics.turning_points import detect_turning_points


FIXTURE = """0.1 Starc to Kohli, 1 run
0.2 Starc to Rohit, 1 run
0.3 Starc to Kohli, FOUR
0.4 Starc to Kohli, 1 run
0.5 Starc to Rohit, 1 run
0.6 Starc to Rohit, no run
1.1 Cummins to Kohli, SIX
1.2 Cummins to Kohli, FOUR
1.3 Cummins to Kohli, FOUR
1.4 Cummins to Kohli, SIX
1.5 Cummins to Kohli, FOUR
1.6 Cummins to Kohli, SIX
2.1 Starc to Kohli, OUT, caught by Smith
2.2 Starc to Rohit, no run
2.3 Starc to Rohit, no run
2.4 Starc to Rohit, no run
2.5 Starc to Rohit, no run
2.6 Starc to Rohit, no run"""


def _tps(top_n=5):
    return detect_turning_points(parse_commentary(FIXTURE), top_n=top_n)


def test_detects_turning_points():
    tps = _tps()
    assert len(tps) >= 2


def test_respects_top_n():
    assert len(_tps(top_n=1)) <= 1


def test_each_has_required_fields():
    for t in _tps():
        for key in ("over", "title", "reason", "importance",
                    "rr_before", "rr_after", "swing", "evidence"):
            assert key in t
        # Run-rate context must be numeric, not fabricated text.
        assert isinstance(t["rr_before"], (int, float))
        assert isinstance(t["rr_after"], (int, float))


def test_evidence_is_grounded():
    # Every turning point must cite at least one piece of commentary evidence.
    for t in _tps():
        assert isinstance(t["evidence"], list)
        assert len(t["evidence"]) >= 1


def test_ordered_by_importance():
    imps = [t["importance"] for t in _tps()]
    assert imps == sorted(imps, reverse=True)


def test_big_over_is_most_important():
    # The 30-run over should rank as the single most important turning point.
    top = _tps()[0]
    assert top["swing"] > 0
    assert "30 runs" in top["title"]


def test_wicket_over_present_with_negative_swing():
    # The collapse over (wicket + dots) should appear with a downward swing.
    swings = [t["swing"] for t in _tps()]
    assert any(s < 0 for s in swings)


def test_empty_input_is_safe():
    assert detect_turning_points(parse_commentary(""), top_n=5) == []
