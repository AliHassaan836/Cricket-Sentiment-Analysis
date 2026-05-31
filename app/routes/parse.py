"""POST /parse — parse commentary and build full match state."""
from __future__ import annotations

import io
import csv
import time
import uuid
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

from app.nlp.commentary_parser import CommentaryParser
from app.nlp.match_state_engine import build_match_state
from app.nlp.turning_point_detector import TurningPointDetector
from app.nlp.rag_pipeline import (
    chunk_commentary, chunk_structured_facts, _vector_store
)
from app.store import save_match, get_match, get_match_meta

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Text/JSON parse ─────────────────────────────────────────────────────────

from pydantic import BaseModel
from typing import Optional as Opt

class CommentaryInput(BaseModel):
    commentary: str
    team1: str = "Team A"
    team2: str = "Team B"
    innings: int = 1
    target: Opt[int] = None
    match_id: Opt[str] = None


@router.post("/parse")
async def parse_commentary(body: CommentaryInput):
    return await _run_parse(
        commentary=body.commentary,
        team1=body.team1,
        team2=body.team2,
        innings=body.innings,
        target=body.target,
        match_id=body.match_id,
    )


# ── File upload parse ────────────────────────────────────────────────────────

@router.post("/parse/upload")
async def parse_file(
    file: UploadFile = File(...),
    team1: str = Form(default="Team A"),
    team2: str = Form(default="Team B"),
    innings: int = Form(default=1),
    target: Optional[int] = Form(default=None),
):
    raw = await file.read()
    filename = file.filename or ""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    if filename.endswith(".csv"):
        commentary = _extract_from_csv(text)
    else:
        commentary = _extract_from_txt(text)

    if not commentary.strip():
        raise HTTPException(status_code=422, detail="No commentary lines found in file.")

    return await _run_parse(
        commentary=commentary,
        team1=team1,
        team2=team2,
        innings=innings,
        target=target,
        match_id=None,
    )


def _extract_from_txt(text: str) -> str:
    """Return lines that look like ball commentary (contain over.ball pattern)."""
    import re
    lines = text.splitlines()
    ball_lines = [l.strip() for l in lines if re.match(r'^\d{1,2}\.\d', l.strip())]
    if ball_lines:
        return "\n".join(ball_lines)
    # Fallback: return all non-empty lines (let parser filter)
    return "\n".join(l.strip() for l in lines if l.strip())


def _extract_from_csv(text: str) -> str:
    """
    Flexible CSV extraction. Looks for columns named:
    over, ball, commentary / description / text / narrative
    OR a single column that contains over.ball formatted lines.
    """
    import re
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        return _extract_from_txt(text)

    headers = [h.lower().strip() for h in (reader.fieldnames or [])]

    # Find over/ball columns
    over_col = next((h for h in headers if h in ("over", "overs", "over_number")), None)
    ball_col = next((h for h in headers if h in ("ball", "ball_number", "delivery")), None)
    text_col = next((h for h in headers if h in (
        "commentary", "description", "text", "narrative",
        "comment", "ball_commentary", "details", "event"
    )), None)

    lines = []
    for row in rows:
        # Normalise keys
        row_lower = {k.lower().strip(): v for k, v in row.items()}

        if over_col and ball_col and text_col:
            ov = row_lower.get(over_col, "").strip()
            bl = row_lower.get(ball_col, "").strip()
            tx = row_lower.get(text_col, "").strip()
            if ov and bl and tx:
                lines.append(f"{ov}.{bl}: {tx}")
        elif text_col:
            tx = row_lower.get(text_col, "").strip()
            if re.match(r'^\d{1,2}\.\d', tx):
                lines.append(tx)
        else:
            # Try every cell for over.ball pattern
            for val in row_lower.values():
                val = str(val).strip()
                if re.match(r'^\d{1,2}\.\d', val):
                    lines.append(val)
                    break

    return "\n".join(lines) if lines else _extract_from_txt(text)


# ── Core parse logic ─────────────────────────────────────────────────────────

async def _run_parse(commentary, team1, team2, innings, target, match_id):
    t0 = time.perf_counter()
    match_id = match_id or str(uuid.uuid4())[:8]

    parser = CommentaryParser()
    try:
        parsed_balls = parser.parse(commentary)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {e}")

    if not parsed_balls:
        raise HTTPException(
            status_code=422,
            detail="No ball events found. Each line must start with over.ball e.g. '1.1: Bumrah to Smith, 1 run'"
        )

    engine = build_match_state(parsed_balls, batting_team=team1, bowling_team=team2,
                                innings_number=innings, target=target)
    state = engine.snapshot()

    detector = TurningPointDetector(batting_team=team1, bowling_team=team2)
    turning_points, momentum_timeline = detector.detect(parsed_balls)

    events_dicts = [
        {"over_ball": b.over_ball, "raw_commentary": b.raw_commentary,
         "is_wicket": b.is_wicket, "is_boundary": b.is_boundary, "runs": b.runs}
        for b in parsed_balls
    ]

    try:
        all_chunks = chunk_commentary(events_dicts) + chunk_structured_facts(state)
        _vector_store.index_match(match_id, all_chunks)
    except Exception as e:
        logger.warning(f"RAG indexing failed (non-fatal): {e}")

    validation = _validate(parsed_balls, state)

    save_match(match_id, innings, {
        "state": state,
        "events": events_dicts,
        "turning_points": [_tp_to_dict(tp) for tp in turning_points],
        "momentum_timeline": [
            {"over_ball": mp.over_ball, "momentum": mp.momentum,
             "cumulative_runs": mp.cumulative_runs, "wickets": mp.wickets, "phase": mp.phase}
            for mp in momentum_timeline
        ],
        "team1": team1, "team2": team2, "innings": innings,
    })

    elapsed_ms = (time.perf_counter() - t0) * 1000

    meta = get_match_meta(match_id)
    return {
        "success": True,
        "match_id": match_id,
        "innings": innings,
        "innings_available": meta["innings_available"] if meta else [innings],
        "events_parsed": len(parsed_balls),
        "match_state": state,
        "events": [_ball_to_dict(b) for b in parsed_balls],
        "validation_report": validation,
        "parse_time_ms": round(elapsed_ms, 1),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _validate(parsed_balls, state) -> dict:
    issues, warnings = [], []
    computed = sum(b.runs for b in parsed_balls if not b.is_extra)
    extras   = sum(b.total_extras or 1 for b in parsed_balls if b.is_extra)
    score_ok = abs((computed + extras) - state["total_runs"]) <= 2
    wkt_count = sum(1 for b in parsed_balls if b.is_wicket)
    wkt_ok = wkt_count == state["wickets"]
    if not wkt_ok:
        issues.append(f"Wicket mismatch: parsed {wkt_count}, state has {state['wickets']}")
    players = {b.batsman for b in parsed_balls if b.batsman} | {b.bowler for b in parsed_balls if b.bowler}
    player_ok = len(players) >= 2
    if not player_ok:
        warnings.append("Very few players detected — check commentary format")
    return {
        "is_valid": score_ok and wkt_ok,
        "score_consistent": score_ok,
        "wickets_consistent": wkt_ok,
        "overs_consistent": state["total_balls"] > 0,
        "player_consistency": player_ok,
        "issues": issues,
        "warnings": warnings,
        "confidence": 0.9 if (score_ok and wkt_ok and player_ok) else 0.6,
    }


def _tp_to_dict(tp) -> dict:
    return {
        "over_ball": tp.over_ball, "title": tp.title, "description": tp.description,
        "evidence": tp.evidence, "impact_score": tp.impact_score,
        "impact_level": tp.impact_level.value if hasattr(tp.impact_level, "value") else tp.impact_level,
        "momentum_before": tp.momentum_before, "momentum_after": tp.momentum_after,
        "momentum_shift": tp.momentum_shift, "event_type": tp.event_type,
        "affected_team": tp.affected_team,
    }


def _ball_to_dict(b) -> dict:
    return {
        "over": b.over, "ball": b.ball, "over_ball": b.over_ball,
        "raw_commentary": b.raw_commentary, "event_type": b.event_type,
        "runs": b.runs, "total_extras": b.total_extras,
        "batsman": b.batsman, "bowler": b.bowler, "fielder": b.fielder,
        "is_wicket": b.is_wicket, "wicket_type": b.wicket_type,
        "dismissed_batsman": b.dismissed_batsman, "is_boundary": b.is_boundary,
        "boundary_type": b.boundary_type, "is_extra": b.is_extra,
        "extra_type": b.extra_type, "is_milestone": b.is_milestone,
        "milestone_description": b.milestone_description, "phase": b.phase,
        "cumulative_score": b.cumulative_score, "run_rate_at_ball": b.run_rate_at_ball,
        "parse_confidence": b.parse_confidence, "entities_extracted": b.entities_extracted,
    }
