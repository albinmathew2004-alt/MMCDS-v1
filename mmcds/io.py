from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

from .types import Event


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def group_events_by_attempt(events: Iterable[Event]) -> Dict[str, List[Event]]:
    grouped: Dict[str, List[Event]] = defaultdict(list)
    for ev in events:
        grouped[str(ev["attempt_id"])].append(ev)

    # Sort per attempt in a stable way.
    for attempt_id, evs in grouped.items():
        evs.sort(key=lambda e: (e.get("client_seq", 0), e.get("occurred_at", "")))
    return dict(grouped)


def load_grouped_events(events_jsonl_path: str) -> Dict[str, List[Event]]:
    return group_events_by_attempt(read_jsonl(events_jsonl_path))


def load_attempt_manifest(attempts_jsonl_path: str) -> Dict[str, dict]:
    attempts: Dict[str, dict] = {}
    for row in read_jsonl(attempts_jsonl_path):
        attempts[str(row["attempt_id"])] = row
    return attempts
