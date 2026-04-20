from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Allow running without installing the package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmcds.risk_engine import score_event_batch
from mmcds.types import ScoringConfig


app = FastAPI(title="MMCDS", version="0.1.0")
cfg = ScoringConfig()


class EventBatchIn(BaseModel):
    attempt_id: str
    events: List[Dict[str, Any]] = Field(..., description="Raw events (append-only)")


class ScoreOut(BaseModel):
    attempt_id: str
    combined_score: float
    base_score: float
    risk: str
    confidence: float
    confidence_score: float
    data_confidence: float
    explanation: Dict[str, List[str]]
    explanation_text: str
    signals: Dict[str, Any]
    patterns: List[Dict[str, Any]]
    pattern_detected: bool
    pattern_type: Optional[str]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/score", response_model=ScoreOut)
def score_attempt(batch: EventBatchIn) -> ScoreOut:
    # Attempt A: score on submission.
    # In a real deployment, you'd load events for attempt_id from storage.

    if not batch.events:
        raise HTTPException(status_code=400, detail="No events provided")

    # Basic sanity check: ensure attempt_id matches.
    for ev in batch.events[:3]:
        if str(ev.get("attempt_id")) != batch.attempt_id:
            raise HTTPException(status_code=400, detail="attempt_id mismatch in events")

    result = score_event_batch(batch.events, attempt_id=batch.attempt_id, cfg=cfg)

    patterns = [p.__dict__ for p in result.patterns]
    strongest = max(result.patterns, key=lambda p: p.strength, default=None)

    return ScoreOut(
        attempt_id=result.attempt_id,
        combined_score=result.combined_score,
        base_score=result.base_score,
        risk=result.risk,
        confidence=result.confidence_score,
        confidence_score=result.confidence_score,
        data_confidence=result.confidence,
        explanation=result.explanation,
        explanation_text=result.explanation_text,
        signals=result.signals,
        patterns=patterns,
        pattern_detected=bool(result.patterns),
        pattern_type=(strongest.pattern_type if strongest is not None else None),
    )
