"""POST /summary — AI-generated match summary."""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.nlp.rag_pipeline import generate_match_summary
from app.store import get_match

router = APIRouter()
logger = logging.getLogger(__name__)


class SummaryRequest(BaseModel):
    match_id: str
    summary_type: str = Field(default="detailed", pattern="^(short|detailed|headline)$")
    innings: int = Field(default=1)


@router.post("/summary")
async def get_summary(body: SummaryRequest):
    match = get_match(body.match_id, body.innings)
    if not match:
        match = get_match(body.match_id, 1)
    if not match:
        raise HTTPException(status_code=404, detail=f"Match '{body.match_id}' not found.")

    result = await generate_match_summary(
        match_state=match["state"],
        turning_points=match.get("turning_points", []),
        summary_type=body.summary_type,
    )
    return {
        "match_id": body.match_id,
        "summary_type": body.summary_type,
        "text": result["text"],
        "validated": result.get("validated", True),
    }
