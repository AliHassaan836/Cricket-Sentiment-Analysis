"""
Commentary Parsing Engine — NLP Core
=====================================
Uses spaCy for NER, regex patterns for structured cricket event extraction,
and rule-based classification for event typing.

NLP Pipeline:
  1. Tokenisation & sentence segmentation (spaCy)
  2. Named Entity Recognition for player/team names
  3. Pattern matching for cricket-specific events
  4. Structured event extraction with confidence scoring
  5. Coreference handling for player mentions
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Regex Pattern Library
# ──────────────────────────────────────────────────────────

# Over.Ball header  e.g. "14.3:" or "14.3 -"
OVER_BALL_RE   = re.compile(r'^(\d{1,2})\.(\d)\s*[:\-–]?\s*')

# Player name extraction  "Bumrah to Smith" / "to Root"
BOWLER_BATSMAN = re.compile(
    r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+to\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
    re.IGNORECASE
)
BATSMAN_ONLY   = re.compile(r'(?:to\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', re.IGNORECASE)

# Runs pattern  "2 runs", "3 run"
RUNS_RE        = re.compile(r'\b(\d)\s+runs?\b', re.IGNORECASE)

# Wicket patterns
WICKET_PHRASES = re.compile(
    r'\b(out|dismissed|taken|falls?|departs?|gone|wicket)\b', re.IGNORECASE
)
WICKET_TYPES = {
    'bowled':    re.compile(r'\b(bowled|stumps shattered|castled|clean bowled)\b', re.IGNORECASE),
    'caught':    re.compile(r'\b(caught|taken at|snaffled|pouched|edged.*?slip|c &? b)\b', re.IGNORECASE),
    'lbw':       re.compile(r'\b(lbw|leg before|plumb)\b', re.IGNORECASE),
    'run_out':   re.compile(r'\b(run.?out|direct hit|short of crease)\b', re.IGNORECASE),
    'stumped':   re.compile(r'\b(stumped|st )\b', re.IGNORECASE),
    'hit_wicket':re.compile(r'\b(hit wicket)\b', re.IGNORECASE),
}

# Boundary patterns
BOUNDARY_6 = re.compile(
    r'\b(six|sixer|maximum|six!|over the boundary|into the stands|out of the ground|'
    r'launched|smashed.*over|hit.*over|cleared.*rope)\b', re.IGNORECASE
)
BOUNDARY_4 = re.compile(
    r'\b(four|boundary|races to the fence|beats the fielder|'
    r'driven.*cover|cut.*point|pulled.*fine|swept.*boundary)\b', re.IGNORECASE
)

# Extra patterns
WIDE_RE    = re.compile(r'\b(wide|down the leg|outside off.*wide)\b', re.IGNORECASE)
NOBALL_RE  = re.compile(r'\b(no.?ball|overstepped|front foot)\b', re.IGNORECASE)
BYE_RE     = re.compile(r'\b(bye|byes)\b', re.IGNORECASE)
LEGBYE_RE  = re.compile(r'\b(leg.?bye|off the pad|off the thigh)\b', re.IGNORECASE)

# Milestone patterns
MILESTONE_RE = re.compile(
    r'\b(fifty|half.?century|50\s*off|100\s*off|century|hundred|'
    r'five.?wicket|five.?for|hat.?trick|\d+\s*not\s*out)\b', re.IGNORECASE
)
FIFTY_RE     = re.compile(r'\b(fifty|half.?century|50\s*off)\b', re.IGNORECASE)
CENTURY_RE   = re.compile(r'\b(century|hundred|100\s*off)\b', re.IGNORECASE)
FIFER_RE     = re.compile(r'\b(five.?wicket|five.?for|5\s*wickets?)\b', re.IGNORECASE)
HATTRICK_RE  = re.compile(r'\b(hat.?trick)\b', re.IGNORECASE)

# Bowling change
BOWLING_CHANGE_RE = re.compile(
    r'\b(comes? on to bowl|new spell|bowling change|replaces?|takes? over)\b',
    re.IGNORECASE
)

# Fielder extraction  "caught by Warner" / "taken by Rohit at slip"
FIELDER_RE = re.compile(
    r'(?:caught|taken|held|pouched)\s+(?:by\s+)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
    re.IGNORECASE
)

# Score for batsman in dismissal  "Root goes for 54"
SCORE_AT_DISMISSAL = re.compile(
    r'([A-Z][a-z]+)\s+(?:goes?|departs?|dismissed)\s+(?:for\s+)?(\d+)',
    re.IGNORECASE
)

# Innings phase calculator
def get_phase(over: int) -> str:
    if over <= 5:   return "Powerplay"
    if over <= 14:  return "Middle"
    return "Death"


# ──────────────────────────────────────────────────────────
# SpaCy NER wrapper  (lazy-loaded so tests run without model)
# ──────────────────────────────────────────────────────────

class NLPEngine:
    _nlp = None

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            try:
                import spacy
                cls._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded: en_core_web_sm")
            except Exception as e:
                logger.warning(f"spaCy unavailable, falling back to regex-only: {e}")
                cls._nlp = None
        return cls._nlp

    @classmethod
    def extract_persons(cls, text: str) -> List[str]:
        """Extract PERSON entities using spaCy NER."""
        nlp = cls.get_nlp()
        if nlp is None:
            return []
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]


# ──────────────────────────────────────────────────────────
# Parsed Ball Dataclass
# ──────────────────────────────────────────────────────────

@dataclass
class ParsedBall:
    over: int
    ball: int
    over_ball: str
    raw_commentary: str

    event_type: str = "NORMAL_RUN"
    runs: int = 0
    total_extras: int = 0

    batsman: Optional[str] = None
    bowler: Optional[str] = None
    fielder: Optional[str] = None

    is_wicket: bool = False
    wicket_type: Optional[str] = None
    dismissed_batsman: Optional[str] = None

    is_boundary: bool = False
    boundary_type: Optional[str] = None

    is_extra: bool = False
    extra_type: Optional[str] = None

    is_milestone: bool = False
    milestone_description: Optional[str] = None

    phase: str = "Powerplay"
    cumulative_score: int = 0
    run_rate_at_ball: float = 0.0

    parse_confidence: float = 1.0
    entities_extracted: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
# Main Parser Class
# ──────────────────────────────────────────────────────────

class CommentaryParser:
    """
    Parses ball-by-ball cricket commentary into structured BallEvent objects.

    NLP methods used:
    - Rule-based regex pattern matching (primary)
    - spaCy NER for player name validation (secondary)
    - Confidence scoring based on pattern coverage
    - Contextual inference for ambiguous deliveries
    """

    def __init__(self):
        self._nlp_engine = NLPEngine()
        self._known_players: set[str] = set()

    # ----------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------

    def parse(self, commentary_text: str) -> List[ParsedBall]:
        """Parse full commentary block. Returns list of ParsedBall objects."""
        start = time.perf_counter()
        lines = [l.strip() for l in commentary_text.strip().split('\n') if l.strip()]
        events: List[ParsedBall] = []
        cumulative = 0

        for line in lines:
            ball = self._parse_line(line)
            if ball is None:
                continue

            # Update player registry for cross-ball consistency
            for p in [ball.batsman, ball.bowler, ball.fielder]:
                if p:
                    self._known_players.add(p)

            # Cumulative score tracking
            if not ball.is_extra:
                cumulative += ball.runs
            else:
                cumulative += ball.total_extras if ball.total_extras else 1
            ball.cumulative_score = cumulative

            # Run rate
            total_balls = ball.over * 6 + ball.ball
            if total_balls > 0:
                ball.run_rate_at_ball = round((cumulative / total_balls) * 6, 2)

            events.append(ball)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Parsed {len(events)} events in {elapsed_ms:.1f}ms")
        return events

    # ----------------------------------------------------------
    # Per-line parsing
    # ----------------------------------------------------------

    def _parse_line(self, line: str) -> Optional[ParsedBall]:
        ob_match = OVER_BALL_RE.match(line)
        if not ob_match:
            return None

        over = int(ob_match.group(1))
        ball = int(ob_match.group(2))
        commentary = line[ob_match.end():].strip()

        pb = ParsedBall(
            over=over,
            ball=ball,
            over_ball=f"{over}.{ball}",
            raw_commentary=commentary,
            phase=get_phase(over),
        )

        self._extract_players(pb, commentary)
        self._classify_event(pb, commentary)
        self._extract_milestone(pb, commentary)
        pb.parse_confidence = self._compute_confidence(pb)
        pb.entities_extracted = [p for p in [pb.batsman, pb.bowler, pb.fielder] if p]

        return pb

    # ----------------------------------------------------------
    # Player extraction
    # ----------------------------------------------------------

    def _extract_players(self, pb: ParsedBall, text: str) -> None:
        # Primary: "Bumrah to Smith"
        m = BOWLER_BATSMAN.search(text)
        if m:
            pb.bowler = m.group(1).strip()
            pb.batsman = m.group(2).strip()
        else:
            # Fallback: "to Smith"
            m2 = BATSMAN_ONLY.search(text)
            if m2:
                pb.batsman = m2.group(1).strip()
            # Bowler from start of line before "to"
            parts = text.split(" to ", 1)
            if len(parts) == 2:
                pb.bowler = parts[0].strip().split(",")[0].strip()

        # Fielder for caught dismissals
        fm = FIELDER_RE.search(text)
        if fm:
            pb.fielder = fm.group(1).strip()

        # Cross-validate with spaCy (soft check)
        try:
            spacy_persons = self._nlp_engine.extract_persons(text)
            for person in spacy_persons:
                if pb.batsman and person.lower() in pb.batsman.lower():
                    pass  # confirmed
                elif pb.bowler and person.lower() in pb.bowler.lower():
                    pass  # confirmed
                # If spaCy finds a new person not caught by regex
                elif person not in (pb.batsman or "", pb.bowler or "", pb.fielder or ""):
                    if pb.fielder is None:
                        pb.fielder = person
        except Exception:
            pass

    # ----------------------------------------------------------
    # Event classification
    # ----------------------------------------------------------

    def _classify_event(self, pb: ParsedBall, text: str) -> None:
        upper = text.upper()

        # ── Wicket ──────────────────────────────────────────
        is_wicket = bool(WICKET_PHRASES.search(text))
        if is_wicket:
            pb.is_wicket = True
            pb.event_type = "WICKET"
            pb.wicket_type = self._classify_wicket_type(text)
            pb.dismissed_batsman = pb.batsman
            # Runs before wicket (e.g. "2 runs and then OUT" rare but possible)
            rm = RUNS_RE.search(text)
            pb.runs = int(rm.group(1)) if rm else 0
            return

        # ── Extra ────────────────────────────────────────────
        if WIDE_RE.search(text):
            pb.is_extra = True
            pb.extra_type = "wide"
            pb.event_type = "EXTRA"
            pb.total_extras = 1
            return

        if NOBALL_RE.search(text):
            pb.is_extra = True
            pb.extra_type = "no_ball"
            pb.event_type = "EXTRA"
            pb.total_extras = 1
            rm = RUNS_RE.search(text)
            pb.runs = int(rm.group(1)) if rm else 0
            return

        if LEGBYE_RE.search(text):
            pb.is_extra = True
            pb.extra_type = "leg_bye"
            pb.event_type = "EXTRA"
            rm = RUNS_RE.search(text)
            pb.total_extras = int(rm.group(1)) if rm else 1
            return

        if BYE_RE.search(text) and "bye-bye" not in text.lower():
            pb.is_extra = True
            pb.extra_type = "bye"
            pb.event_type = "EXTRA"
            rm = RUNS_RE.search(text)
            pb.total_extras = int(rm.group(1)) if rm else 1
            return

        # ── Six ──────────────────────────────────────────────
        if BOUNDARY_6.search(text):
            pb.is_boundary = True
            pb.boundary_type = "6"
            pb.event_type = "BOUNDARY_6"
            pb.runs = 6
            return

        # ── Four ─────────────────────────────────────────────
        if BOUNDARY_4.search(text):
            pb.is_boundary = True
            pb.boundary_type = "4"
            pb.event_type = "BOUNDARY_4"
            pb.runs = 4
            return

        # ── Normal runs ──────────────────────────────────────
        rm = RUNS_RE.search(text)
        if rm:
            n = int(rm.group(1))
            pb.runs = n
            pb.event_type = "NORMAL_RUN" if n > 0 else "DOT_BALL"
            return

        # ── Dot ball / no run ────────────────────────────────
        if re.search(r'\b(no run|dot|played and missed|beaten|defended)\b', text, re.IGNORECASE):
            pb.runs = 0
            pb.event_type = "DOT_BALL"
            return

        # Default: assume dot ball
        pb.runs = 0
        pb.event_type = "DOT_BALL"

    # ----------------------------------------------------------
    # Wicket type classification
    # ----------------------------------------------------------

    def _classify_wicket_type(self, text: str) -> str:
        for wtype, pattern in WICKET_TYPES.items():
            if pattern.search(text):
                return wtype.replace("_", " ").title()
        return "Unknown"

    # ----------------------------------------------------------
    # Milestone extraction
    # ----------------------------------------------------------

    def _extract_milestone(self, pb: ParsedBall, text: str) -> None:
        if not MILESTONE_RE.search(text):
            return
        pb.is_milestone = True
        pb.event_type = "MILESTONE"

        if CENTURY_RE.search(text):
            pb.milestone_description = f"{pb.batsman or 'Batsman'} reaches century"
        elif FIFTY_RE.search(text):
            pb.milestone_description = f"{pb.batsman or 'Batsman'} reaches fifty"
        elif FIFER_RE.search(text):
            pb.milestone_description = f"{pb.bowler or 'Bowler'} takes five-wicket haul"
        elif HATTRICK_RE.search(text):
            pb.milestone_description = f"{pb.bowler or 'Bowler'} completes hat-trick"
        else:
            m = MILESTONE_RE.search(text)
            pb.milestone_description = m.group(0) if m else "Milestone"

    # ----------------------------------------------------------
    # Confidence scoring
    # ----------------------------------------------------------

    def _compute_confidence(self, pb: ParsedBall) -> float:
        score = 0.5  # base
        if pb.batsman:    score += 0.15
        if pb.bowler:     score += 0.15
        if pb.event_type != "NORMAL_RUN":  score += 0.1
        if pb.is_wicket and pb.wicket_type != "Unknown":  score += 0.1
        return min(round(score, 2), 1.0)
