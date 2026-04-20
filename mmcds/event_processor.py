from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .types import Event


def _parse_ts(ts: str) -> datetime:
    # Accept the Z suffix written by the synthetic generator.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


@dataclass(frozen=True)
class ProcessedEvents:
    attempt_id: str
    events: List[Event]
    occurred_at: List[datetime]


class EventProcessingError(ValueError):
    pass


def normalize_events(raw_events: Iterable[Dict[str, Any]], *, attempt_id: Optional[str] = None) -> ProcessedEvents:
    """Normalize and validate an event batch.

    - Parses timestamps
    - Sorts by occurred_at then client_seq
    - Filters obviously malformed rows (missing occurred_at/event_type)
    - Ensures attempt_id consistency when provided

    This function is intentionally strict: production systems should reject
    ill-formed telemetry rather than silently produce misleading scores.
    """

    events: List[Event] = []
    occurred: List[datetime] = []

    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        if "occurred_at" not in ev or "event_type" not in ev:
            continue
        try:
            t = _parse_ts(str(ev["occurred_at"]))
        except Exception:
            continue

        if attempt_id is not None:
            if str(ev.get("attempt_id", "")) != str(attempt_id):
                raise EventProcessingError("attempt_id mismatch in events")

        events.append(ev)  # type: ignore[arg-type]
        occurred.append(t)

    if not events:
        raise EventProcessingError("No valid events provided")

    inferred_attempt_id = str(attempt_id or events[0].get("attempt_id", ""))
    if not inferred_attempt_id:
        raise EventProcessingError("Missing attempt_id")

    # stable sort by occurred time then client_seq
    order = sorted(range(len(events)), key=lambda i: (occurred[i], int(events[i].get("client_seq", 0) or 0)))
    events_sorted = [events[i] for i in order]
    occurred_sorted = [occurred[i] for i in order]

    return ProcessedEvents(attempt_id=inferred_attempt_id, events=events_sorted, occurred_at=occurred_sorted)
