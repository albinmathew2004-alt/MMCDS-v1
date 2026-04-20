"""Small demo: score a single synthetic-ish event batch and print output.

Run:
  python scripts/demo_score_api_payload.py

This is meant as an example input/output for the production-style scorer.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmcds.risk_engine import score_event_batch


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    attempt_id = "attempt_demo_001"
    now = datetime.now(timezone.utc)

    # Minimal but realistic event sequence (includes a suspicious pattern)
    events = []
    seq = 0

    def ev(event_type: str, seconds: float, payload: dict) -> dict:
        nonlocal seq
        seq += 1
        t = now + timedelta(seconds=seconds)
        return {
            "event_id": f"e{seq}",
            "attempt_id": attempt_id,
            "user_id": "user_demo",
            "assessment_id": "asm_demo",
            "occurred_at": utc_iso(t),
            "received_at": utc_iso(t + timedelta(milliseconds=50)),
            "client_seq": seq,
            "event_type": event_type,
            "client_meta": {"platform": "windows", "browser": "edge"},
            "payload": payload,
        }

    events.append(ev("attempt_state", 0, {"action": "start"}))
    events.append(ev("question_view", 5, {"action": "enter", "question_id": "q1"}))

    # Idle spike then tab hidden then paste then quick leave
    events.append(ev("idle_state", 30, {"state": "idle"}))
    events.append(ev("idle_state", 65, {"state": "active"}))
    events.append(ev("visibility_change", 70, {"state": "hidden"}))
    events.append(ev("clipboard", 80, {"action": "paste", "question_id": "q1"}))
    events.append(ev("clipboard", 81, {"action": "paste", "question_id": "q1"}))
    events.append(ev("visibility_change", 82, {"state": "visible"}))
    events.append(ev("question_view", 88, {"action": "leave", "question_id": "q1"}))

    events.append(ev("attempt_state", 90, {"action": "submit"}))

    result = score_event_batch(events, attempt_id=attempt_id)

    out = {
        "risk": result.risk,
        "confidence": result.confidence_score,
        "confidence_score": result.confidence_score,
        "data_confidence": result.confidence,
        "signals": result.signals,
        "patterns": [p.__dict__ for p in result.patterns],
        "explanation": result.explanation_text,
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
