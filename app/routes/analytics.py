"""Analytics, match state, and innings routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from app.store import get_match, get_match_meta, list_matches

router = APIRouter()


@router.get("/analytics/{match_id}")
async def get_analytics(match_id: str, innings: int = Query(default=1)):
    match = get_match(match_id, innings)
    if not match:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' innings {innings} not found.")
    return {
        "match_id": match_id,
        "innings": innings,
        "turning_points": match.get("turning_points", []),
        "momentum_timeline": match.get("momentum_timeline", []),
        "phase_scores": match["state"].get("phase_scores", {}),
        "phase_wickets": match["state"].get("phase_wickets", {}),
        "over_history": match["state"].get("over_history", []),
    }


@router.get("/match/{match_id}")
async def get_match_state(match_id: str, innings: int = Query(default=1)):
    match = get_match(match_id, innings)
    if not match:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' innings {innings} not found.")
    return match["state"]


@router.get("/match/{match_id}/meta")
async def get_match_info(match_id: str):
    meta = get_match_meta(match_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found.")
    return meta


@router.get("/matches")
async def list_all_matches():
    return {"match_ids": list_matches()}
