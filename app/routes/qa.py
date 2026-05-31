"""POST /qa — RAG-powered question answering."""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.nlp.rag_pipeline import RAGQAEngine, _vector_store
from app.store import get_match, get_match_meta

router = APIRouter()
logger = logging.getLogger(__name__)
_qa_engine = RAGQAEngine(_vector_store)


class QARequest(BaseModel):
    match_id: str
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    innings: int = Field(default=1)


@router.post("/qa")
async def answer_question(body: QARequest):
    match = get_match(body.match_id, body.innings)
    if not match:
        # Try innings 1 as fallback
        match = get_match(body.match_id, 1)
    if not match:
        raise HTTPException(status_code=404, detail=f"Match '{body.match_id}' not found.")

    result = await _qa_engine.answer(
        match_id=body.match_id,
        question=body.question,
        top_k=body.top_k,
        match_state=match["state"],
    )
    return {"match_id": body.match_id, "question": body.question, **result}
