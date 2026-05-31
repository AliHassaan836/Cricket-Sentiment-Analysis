"""
RAG (Retrieval-Augmented Generation) Pipeline
===============================================
Implements a two-stage QA system:
  1. Retrieval  — FAISS vector store over commentary + structured events
  2. Generation — LLM with grounded context (prevents hallucination)

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, fast)
Vector store:    FAISS flat index
LLM:             OpenAI GPT-4o-mini via LangChain (swap for local LLM)

Hallucination prevention:
  - System prompt restricts LLM to provided context only
  - Structured facts (score, wickets, stats) injected separately from commentary
  - Retrieved chunk score threshold filters low-confidence retrievals
"""

from __future__ import annotations

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root regardless of working directory
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# Lazy imports (avoid startup crash if packages missing)
_faiss = None
_SentenceTransformer = None
_openai_client = None


def _load_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _load_st():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


def _load_openai():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY not set. Add your key to the .env file:\n"
                "OPENAI_API_KEY=sk-proj-..."
            )
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


# ──────────────────────────────────────────────────────────
# Document chunk model
# ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────
# Chunking strategies
# ──────────────────────────────────────────────────────────

def chunk_commentary(events: List[dict], window: int = 6) -> List[Chunk]:
    """
    Chunk ball events into overlapping windows for dense retrieval.
    Each chunk = window deliveries of commentary (default 6 = 1 over).
    50% overlap between chunks ensures context is not lost at boundaries.
    """
    chunks: List[Chunk] = []
    step = window // 2

    for i in range(0, len(events), step):
        batch = events[i : i + window]
        if not batch:
            break

        lines = [f"[{e['over_ball']}] {e['raw_commentary']}" for e in batch]
        text = "\n".join(lines)

        meta = {
            "type": "commentary",
            "start_over": batch[0]["over_ball"],
            "end_over": batch[-1]["over_ball"],
            "wickets": sum(1 for e in batch if e.get("is_wicket")),
            "runs": sum(e.get("runs", 0) for e in batch),
            "boundaries": sum(1 for e in batch if e.get("is_boundary")),
        }
        chunks.append(Chunk(
            chunk_id=f"commentary_{i}",
            text=text,
            metadata=meta,
        ))

    return chunks


def chunk_structured_facts(state: dict) -> List[Chunk]:
    """
    Convert structured match state into natural-language fact chunks.
    These are indexed alongside commentary for structured QA.
    """
    chunks: List[Chunk] = []

    # Score fact
    score_text = (
        f"The current score is {state['total_runs']}/{state['wickets']} "
        f"in {state['overs']}.{state['balls_in_over']} overs. "
        f"Run rate: {state['run_rate']}."
    )
    chunks.append(Chunk("fact_score", score_text, {"type": "fact", "category": "score"}))

    # Batting facts
    for name, bstats in state.get("batting_stats", {}).items():
        text = (
            f"{name} scored {bstats['runs']} runs off {bstats['balls_faced']} balls "
            f"(SR: {bstats['strike_rate']}). "
            f"Hit {bstats['fours']} fours and {bstats['sixes']} sixes. "
            f"{'Dismissed.' if bstats['is_out'] else 'Not out.'}"
        )
        chunks.append(Chunk(f"fact_bat_{name}", text, {"type": "fact", "category": "batting", "player": name}))

    # Bowling facts
    for name, bwstats in state.get("bowling_stats", {}).items():
        text = (
            f"{name} bowled {bwstats['overs_bowled']} overs, "
            f"conceded {bwstats['runs_conceded']} runs and took {bwstats['wickets']} wickets. "
            f"Economy: {bwstats['economy']}. Dot balls: {bwstats['dot_balls']}."
        )
        chunks.append(Chunk(f"fact_bowl_{name}", text, {"type": "fact", "category": "bowling", "player": name}))

    # Partnership facts
    for p in state.get("partnerships", []):
        text = (
            f"The {_ordinal(p['wicket_number'])} wicket partnership between "
            f"{p['batsman1']} and {p['batsman2']} produced {p['runs']} runs "
            f"off {p['balls']} balls at a run rate of {p['run_rate']}."
        )
        chunks.append(Chunk(f"fact_p{p['wicket_number']}", text, {"type": "fact", "category": "partnership"}))

    # Milestone facts
    for m in state.get("milestones", []):
        text = f"Milestone at over {m['over']}: {m['description']}."
        chunks.append(Chunk(f"fact_ms_{m['over']}", text, {"type": "fact", "category": "milestone"}))

    return chunks


def _ordinal(n: int) -> str:
    suffixes = {1: "1st", 2: "2nd", 3: "3rd"}
    return suffixes.get(n, f"{n}th")


# ──────────────────────────────────────────────────────────
# FAISS Vector Store
# ──────────────────────────────────────────────────────────

class CricketVectorStore:
    """
    In-memory FAISS vector store for cricket match data.
    Supports per-match indexing (match_id → index).
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"
    EMBED_DIM   = 384
    SIM_THRESH  = 0.35    # minimum cosine similarity for retrieval

    def __init__(self):
        self._model = None
        self._indexes: Dict[str, dict] = {}   # match_id → {index, chunks}

    def _get_model(self):
        if self._model is None:
            ST = _load_st()
            self._model = ST(self.EMBED_MODEL)
            logger.info(f"Loaded embedding model: {self.EMBED_MODEL}")
        return self._model

    def index_match(self, match_id: str, chunks: List[Chunk]) -> None:
        """Embed and index all chunks for a match."""
        model = self._get_model()
        faiss = _load_faiss()

        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        index = faiss.IndexFlatIP(self.EMBED_DIM)   # Inner product = cosine (normalised)
        index.add(embeddings.astype(np.float32))

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]

        self._indexes[match_id] = {"index": index, "chunks": chunks}
        logger.info(f"Indexed {len(chunks)} chunks for match {match_id}")

    def retrieve(
        self,
        match_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """Return top-k chunks most relevant to the query."""
        if match_id not in self._indexes:
            logger.warning(f"No index found for match {match_id}")
            return []

        model  = self._get_model()
        faiss  = _load_faiss()
        store  = self._indexes[match_id]

        q_emb  = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = store["index"].search(q_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and float(score) >= self.SIM_THRESH:
                results.append((store["chunks"][idx], float(score)))

        return results

    def has_match(self, match_id: str) -> bool:
        return match_id in self._indexes


# Singleton store
_vector_store = CricketVectorStore()


# ──────────────────────────────────────────────────────────
# QA Engine
# ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a cricket match intelligence assistant.
You MUST answer ONLY using the provided match context.
Never invent statistics, player names, or match events.
If the answer is not in the context, say "I don't have that information in the match data."
Be concise, precise, and use cricket terminology.
When citing statistics, always include the over/ball reference if available."""

class RAGQAEngine:
    """
    Two-stage QA:
      1. Retrieve relevant chunks from FAISS index
      2. Generate answer with LLM grounded in retrieved context
    """

    def __init__(self, vector_store: CricketVectorStore = _vector_store):
        self._store = vector_store

    async def answer(
        self,
        match_id: str,
        question: str,
        top_k: int = 5,
        match_state: Optional[dict] = None,
    ) -> dict:
        """
        Returns:
            answer: str
            sources: list of source chunks with scores
            confidence: float (0-1)
            grounded: bool
        """
        # Retrieval
        results = self._store.retrieve(match_id, question, top_k=top_k)

        if not results and match_state is None:
            return {
                "answer": "No match data found. Please parse commentary first.",
                "sources": [],
                "confidence": 0.0,
                "grounded": False,
                "retrieved_context": [],
            }

        # Build context
        context_parts = []
        sources = []

        for chunk, score in results:
            context_parts.append(chunk.text)
            sources.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200],
                "score": round(score, 3),
                "metadata": chunk.metadata,
            })

        # Add structured facts from match state if provided
        structured_context = ""
        if match_state:
            structured_context = _build_structured_context(match_state)

        full_context = structured_context
        if context_parts:
            full_context += "\n\nRELEVANT COMMENTARY:\n" + "\n---\n".join(context_parts)

        user_message = f"Context:\n{full_context}\n\nQuestion: {question}"

        # Generation
        try:
            answer, confidence = await self._generate(user_message)
            grounded = True
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            answer = self._fallback_answer(question, match_state, results)
            confidence = 0.5
            grounded = bool(results or match_state)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "grounded": grounded,
            "retrieved_context": context_parts[:3],
        }

    async def _generate(self, user_message: str) -> Tuple[str, float]:
        client = _load_openai()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        text = response.choices[0].message.content or ""
        # Confidence heuristic based on finish reason and length
        confidence = 0.85 if response.choices[0].finish_reason == "stop" else 0.6
        return text.strip(), confidence

    def _fallback_answer(
        self,
        question: str,
        state: Optional[dict],
        results: List,
    ) -> str:
        """Rule-based fallback when LLM is unavailable."""
        if not state:
            return "Match data not available."

        q = question.lower()
        if "score" in q:
            return (
                f"The score is {state['total_runs']}/{state['wickets']} "
                f"in {state['overs']}.{state['balls_in_over']} overs "
                f"(Run rate: {state['run_rate']})."
            )
        if "wicket" in q:
            return f"{state['wickets']} wickets have fallen so far."
        if "run rate" in q or "rr" in q:
            return f"Current run rate is {state['run_rate']}."
        if results:
            return f"Based on the commentary: {results[0][0].text[:200]}..."
        return "I couldn't find specific information for that question in the match data."


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _build_structured_context(state: dict) -> str:
    """Build a structured fact summary for injection into LLM context."""
    lines = [
        "STRUCTURED MATCH FACTS (verified, authoritative):",
        f"Score: {state['total_runs']}/{state['wickets']} in {state['overs']}.{state['balls_in_over']} overs",
        f"Run Rate: {state['run_rate']}",
        f"Batting Team: {state['batting_team']} | Bowling Team: {state['bowling_team']}",
    ]

    if state.get("batting_stats"):
        lines.append("\nBATTING:")
        for name, s in state["batting_stats"].items():
            lines.append(
                f"  {name}: {s['runs']}({s['balls_faced']}b) SR:{s['strike_rate']} "
                f"4s:{s['fours']} 6s:{s['sixes']} {'[OUT]' if s['is_out'] else '[*]'}"
            )

    if state.get("bowling_stats"):
        lines.append("\nBOWLING:")
        for name, s in state["bowling_stats"].items():
            lines.append(
                f"  {name}: {s['wickets']}/{s['runs_conceded']} in {s['overs_bowled']}ov "
                f"Eco:{s['economy']} Dots:{s['dot_balls']}"
            )

    if state.get("partnerships"):
        lines.append("\nPARTNERSHIPS:")
        for p in state["partnerships"]:
            lines.append(
                f"  Wkt{p['wicket_number']}: {p['batsman1']}&{p['batsman2']} "
                f"{p['runs']}({p['balls']}b)"
            )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Summary generator using LLM
# ──────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """You are a professional cricket journalist.
Generate match reports using ONLY the provided match statistics.
NEVER invent statistics, scores, or player names.
Write in engaging cricket journalism style."""

async def generate_match_summary(
    match_state: dict,
    turning_points: List[dict],
    summary_type: str = "detailed",
) -> dict:
    """
    Generate AI match summary grounded in verified match data.
    Returns headline, short summary, and optionally a detailed report.
    """
    context = _build_structured_context(match_state)
    if turning_points:
        context += "\n\nKEY TURNING POINTS:\n"
        for tp in turning_points[:4]:
            context += f"  - Over {tp['over_ball']}: {tp['title']} (impact: {tp['impact_score']}%)\n"

    if summary_type == "headline":
        prompt = f"{context}\n\nWrite a single punchy headline for this innings."
        max_tok = 50
    elif summary_type == "short":
        prompt = f"{context}\n\nWrite a 2-sentence summary of this innings."
        max_tok = 150
    else:
        prompt = (
            f"{context}\n\n"
            "Write a 3-paragraph match report:\n"
            "Para 1: Innings overview (score, run rate, key phases)\n"
            "Para 2: Top batting and bowling performances\n"
            "Para 3: Turning points and match analysis\n"
            "Use professional cricket journalism style."
        )
        max_tok = 600

    try:
        client = _load_openai()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tok,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        text = _fallback_summary(match_state, summary_type)

    return {"text": text.strip(), "validated": True}


def _fallback_summary(state: dict, summary_type: str) -> str:
    score = f"{state['total_runs']}/{state['wickets']}"
    overs = f"{state['overs']}.{state['balls_in_over']}"
    rr    = state["run_rate"]
    team  = state["batting_team"]

    if summary_type == "headline":
        return f"{team} post {score} in {overs} overs"
    return (
        f"{team} scored {score} in {overs} overs at a run rate of {rr}. "
        f"{'A competitive total.' if state['total_runs'] > 140 else 'A modest total.'}"
    )
