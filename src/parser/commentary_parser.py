"""
commentary_parser.py  (Module 1: Commentary Parsing)
====================================================
Convert unstructured ball-by-ball commentary into structured, verifiable records.

DESIGN PRINCIPLE — ACCURACY FIRST
---------------------------------
This module is the single source of truth for every number in the application.
It is rule-based and fully deterministic: the same input always produces the same
output, and every field is traceable to a substring of the original line. No
machine-learning model is allowed to *create* a statistic; models may only
describe what this parser has already extracted.

Expected line grammar
----------------------
    <over>.<ball> <bowler> to <batter>, <description>

Examples
    0.1 Shaheen to Rohit, no run
    0.2 Shaheen to Rohit, FOUR
    12.3 Starc to Kohli, FOUR
    0.4 Shaheen to Gill, OUT, caught behind
    5.2 Bumrah to Warner, 1 wide
    7.5 Rashid to Maxwell, 2 leg byes

Lines that do not match the grammar are collected in `unparsed_lines` and are
never silently dropped — the validation layer reports them to the user.

Scoring conventions (documented so analytics are reproducible)
--------------------------------------------------------------
* batter_runs  : runs credited to the striker (off the bat).
* extras       : wides + no-balls + byes + leg-byes.
* total_runs   : batter_runs + extras (what the team scores off the delivery).
* legal ball   : any delivery that is NOT a wide and NOT a no-ball. Only legal
                 balls advance the over count and the bowler's quota.
* ball_faced   : True for everything except wides (the striker did not face a
                 wide). Byes/leg-byes/no-balls are treated as faced.
* bowler runs  : batter_runs + wides + no_balls. Byes and leg-byes are NOT
                 charged to the bowler (standard cricket scoring).
* run-outs are NOT credited to the bowler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Regular expressions
# --------------------------------------------------------------------------- #
# Leading "over.ball  bowler  to  batter ,  rest"
LINE_RE = re.compile(
    r"^\s*(?P<over>\d+)\.(?P<ball>\d+):?\s+"
    r"(?P<bowler>.+?)\s+to\s+(?P<batter>.+?)\s*[,\-]\s*(?P<desc>.+?)\s*$"
)

# Structural / header lines to skip silently (not counted as parse errors).
_SKIP_RE = re.compile(
    r"^\s*(?:"
    r"over\s+\d+"                 # "Over 5: ..."
    r"|.*innings"                 # "INDIA INNINGS", "NEW ZEALAND INNINGS (Target: ...)"
    r"|toss\s*:"                  # "Toss: ..."
    r"|.*\bvs\.?\s"              # "India vs New Zealand"
    r"|target\s*:"               # "Target: ..."
    r"|.*walks\s+in\b"           # "Ishan Kishan walks in at number 3"
    r"|end\s+of\s+\w+"           # "End of Powerplay: ..."
    r"|-{2,}"                    # "---" separator lines
    r"|={2,}"                    # "===" separator lines
    r"|.*\bscore\s*:"            # "Score: ..."
    r")",
    re.IGNORECASE,
)

# Numeric runs like "1 run", "3 runs"
RUNS_RE = re.compile(r"(?<!\w)(\d+)\s+runs?\b", re.IGNORECASE)
# Extras with optional leading count: "2 wides", "wide", "1 no ball", "4 byes"
WIDE_RE = re.compile(r"(?:(\d+)\s+)?wides?\b", re.IGNORECASE)
NOBALL_RE = re.compile(r"(?:(\d+)\s+)?no[\s-]?balls?\b", re.IGNORECASE)
LEGBYE_RE = re.compile(r"(?:(\d+)\s+)?leg[\s-]?byes?\b", re.IGNORECASE)
BYE_RE = re.compile(r"(?:(\d+)\s+)?byes?\b", re.IGNORECASE)

DISMISSAL_KEYWORDS = [
    "run out", "runout", "caught and bowled", "caught behind", "caught",
    "bowled", "lbw", "stumped", "hit wicket", "retired hurt", "obstructing",
]


@dataclass
class Delivery:
    """One parsed delivery. Mirrors exactly one commentary line."""
    seq: int
    over_num: int
    ball_num: int
    over_str: str
    bowler: str
    batter: str
    batter_runs: int = 0
    extras_total: int = 0
    wides: int = 0
    no_balls: int = 0
    byes: int = 0
    leg_byes: int = 0
    total_runs: int = 0
    is_wicket: bool = False
    dismissal_type: Optional[str] = None
    dismissed_batter: Optional[str] = None
    fielders: str = ""
    is_four: bool = False
    is_six: bool = False
    is_boundary: bool = False
    is_legal: bool = True       # counts toward over / bowler quota
    ball_faced: bool = True     # striker faced the delivery
    is_dot: bool = False        # no runs at all off the delivery
    bowler_runs: int = 0        # runs charged to the bowler
    event_label: str = "NEUTRAL"
    raw_text: str = ""
    innings: int = 1


class CommentaryParser:
    """Parses a commentary file/text into a list of :class:`Delivery` records."""

    def __init__(self) -> None:
        self.deliveries: List[Delivery] = []
        self.unparsed_lines: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse_text(self, text: str) -> List[Delivery]:
        """Parse a multi-line commentary string."""
        self.deliveries = []
        self.unparsed_lines = []
        seq = 0
        innings = 1
        for line_no, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Detect innings boundary (e.g. "INDIA INNINGS", "NEW ZEALAND INNINGS")
            if re.search(r"\binnings\b", line, re.IGNORECASE):
                # First time we see "innings" it's still innings 1; subsequent
                # occurrences bump the counter.
                if seq > 0:
                    innings += 1
                continue
            # Skip structural/header lines silently
            if _SKIP_RE.match(line):
                continue
            delivery = self._parse_line(line, seq)
            if delivery is None:
                self.unparsed_lines.append({"line_no": line_no, "text": raw})
            else:
                delivery.innings = innings
                self.deliveries.append(delivery)
                seq += 1
        return self.deliveries

    def parse_file(self, path: str) -> List[Delivery]:
        with open(path, "r", encoding="utf-8") as fh:
            return self.parse_text(fh.read())

    def to_dataframe(self) -> pd.DataFrame:
        """Return parsed deliveries as a tidy DataFrame (empty-safe)."""
        if not self.deliveries:
            return pd.DataFrame(columns=[f.name for f in Delivery.__dataclass_fields__.values()])
        return pd.DataFrame([asdict(d) for d in self.deliveries])

    # ------------------------------------------------------------------ #
    # Line-level parsing
    # ------------------------------------------------------------------ #
    def _parse_line(self, line: str, seq: int) -> Optional[Delivery]:
        m = LINE_RE.match(line)
        if not m:
            return None

        over_num = int(m.group("over"))
        ball_num = int(m.group("ball"))
        bowler = self._clean_name(m.group("bowler"))
        batter = self._clean_name(m.group("batter"))
        desc = m.group("desc").strip()

        d = Delivery(
            seq=seq,
            over_num=over_num,
            ball_num=ball_num,
            over_str=f"{over_num}.{ball_num}",
            bowler=bowler,
            batter=batter,
            raw_text=line,
        )

        self._extract_extras(desc, d)
        self._extract_wicket(desc, d, batter)
        self._extract_runs(desc, d)
        self._finalise(d)
        return d

    # ------------------------------------------------------------------ #
    # Field extractors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_name(name: str) -> str:
        return re.sub(r"\s+", " ", name).strip().strip(".,")

    def _extract_extras(self, desc: str, d: Delivery) -> None:
        low = desc.lower().strip()

        # A genuine wide delivery starts with "wide" / "1 wide" / "2 wides" etc.
        # Phrases like "wide outside off" (describing delivery line) must NOT
        # trigger a wide extra.
        if re.match(r"(?:\d+\s+)?wides?\b", low):
            mm = WIDE_RE.search(low)
            n = int(mm.group(1)) if mm and mm.group(1) else 1
            d.wides = n
            d.is_legal = False
            d.ball_faced = False

        if re.search(r"no[\s-]?ball", low):
            mm = NOBALL_RE.search(low)
            n = int(mm.group(1)) if mm and mm.group(1) else 1
            d.no_balls = n
            d.is_legal = False  # no-ball does not count toward the over

        if re.search(r"leg[\s-]?bye", low):
            mm = LEGBYE_RE.search(low)
            n = int(mm.group(1)) if mm and mm.group(1) else 1
            d.leg_byes = n
        elif re.search(r"\bbye", low):
            mm = BYE_RE.search(low)
            n = int(mm.group(1)) if mm and mm.group(1) else 1
            d.byes = n

        d.extras_total = d.wides + d.no_balls + d.byes + d.leg_byes

    def _extract_wicket(self, desc: str, d: Delivery, batter: str) -> None:
        low = desc.lower().strip()
        # A delivery is a wicket only if the description begins with "OUT"
        # (followed by !, comma, or whitespace). This avoids false positives
        # from common commentary phrases like "dug out", "bowled slowly",
        # "appeal for LBW", "picked out the fielder", etc.
        is_out = bool(re.match(r"out[\s!,.\-]", low)) or low == "out"
        if not is_out:
            return

        d.is_wicket = True
        d.dismissed_batter = batter

        dtype = None
        for kw in DISMISSAL_KEYWORDS:
            if kw in low:
                dtype = "run out" if kw == "runout" else kw
                break
        d.dismissal_type = dtype if dtype else "out"

        # Fielder extraction: "caught by Smith", "Caught by Smith", "c Smith", "stumped Patel"
        fielder_match = re.search(r"(?:caught by|c |stumped by|stumped|run out by)\s+([A-Za-z][A-Za-z .'-]+)", desc, re.IGNORECASE)
        if fielder_match:
            d.fielders = self._clean_name(fielder_match.group(1))

    def _extract_runs(self, desc: str, d: Delivery) -> None:
        low = desc.lower()

        # Boundaries are flagged ONLY by explicit words ("FOUR", "SIX",
        # "boundary"). A numeric like "4 runs" is NOT auto-classified as a
        # boundary, because all-run 4s and 6s exist — we never infer a boundary
        # that the commentary did not explicitly state.
        if re.search(r"\bsix\b|sixes\b", low):
            d.is_six = True
            d.is_boundary = True
            d.batter_runs = max(d.batter_runs, 6)
        if re.search(r"\bfour\b|\bboundary\b", low):
            d.is_four = True
            d.is_boundary = True
            d.batter_runs = max(d.batter_runs, 4)

        # Explicit numeric "N run(s)" — but only credit to the batter if those
        # runs are NOT already accounted for as extras. If the delivery is a
        # wide/bye/leg-bye, the numeric belongs to the extra, not the batter.
        if not d.is_boundary:
            mm = RUNS_RE.search(low)
            if mm:
                runs = int(mm.group(1))
                if d.wides or d.byes or d.leg_byes:
                    # numeric already represents the extra count; do nothing
                    pass
                elif d.no_balls:
                    # "1 no ball" -> the 1 is the no-ball penalty (already counted).
                    # "no ball, 2 runs" style would credit the batter; handled by
                    # checking whether the run token is separate from 'no ball'.
                    if not re.search(r"\d+\s+no[\s-]?ball", low):
                        d.batter_runs = runs
                else:
                    d.batter_runs = runs

        # "no run" / "no runs" / "dot" => explicit zero off the bat
        if re.search(r"\bno runs?\b|\bdot\b", low) and not d.is_boundary:
            d.batter_runs = 0

    def _finalise(self, d: Delivery) -> None:
        d.total_runs = d.batter_runs + d.extras_total
        # Bowler is charged batter runs + wides + no-balls (not byes/leg-byes).
        d.bowler_runs = d.batter_runs + d.wides + d.no_balls
        d.is_dot = (d.total_runs == 0)

        # Event label (used for sentiment / momentum). Wicket dominates.
        if d.is_wicket:
            d.event_label = "WICKET"
        elif d.is_six:
            d.event_label = "SIX"
        elif d.is_four:
            d.event_label = "FOUR"
        elif d.wides:
            d.event_label = "WIDE"
        elif d.no_balls:
            d.event_label = "NO_BALL"
        elif d.leg_byes:
            d.event_label = "LEG_BYES"
        elif d.byes:
            d.event_label = "BYES"
        elif d.batter_runs == 1:
            d.event_label = "SINGLE"
        elif d.batter_runs == 2:
            d.event_label = "DOUBLE"
        elif d.batter_runs == 3:
            d.event_label = "TRIPLE"
        elif d.is_dot:
            d.event_label = "DOT"
        else:
            d.event_label = "NEUTRAL"


def parse_commentary(text: str) -> pd.DataFrame:
    """Convenience function: text -> tidy DataFrame of deliveries."""
    p = CommentaryParser()
    p.parse_text(text)
    return p.to_dataframe()
