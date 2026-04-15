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

from mmcds.explanations import generate_explanations
from mmcds.features import compute_features
from mmcds.reasoning import combine_signals
from mmcds.risk import assign_risk, compute_confidence
from mmcds.signals import score_signals
from mmcds.types import Event, ScoringConfig


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
    explanation: Dict[str, List[str]]
    signals: Dict[str, Any]


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

    features = compute_features(batch.events, cfg.feature)
    signals = score_signals(features, cfg.signal)
    combined = combine_signals(signals, cfg.reasoning)

    risk = assign_risk(combined["combined_score"], combined, cfg.risk)
    confidence = compute_confidence(features, signals, combined, cfg.confidence)
    explanation = generate_explanations(features, signals, risk)

    return ScoreOut(
        attempt_id=batch.attempt_id,
        combined_score=combined["combined_score"],
        base_score=combined["base_score"],
        risk=risk,
        confidence=confidence,
        explanation=explanation,
        signals=signals,
    )
