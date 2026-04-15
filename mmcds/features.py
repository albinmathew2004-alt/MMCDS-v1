from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .types import Event, FeatureConfig, Features


def _parse_ts(ts: str) -> datetime:
    # Accept the Z suffix written by the synthetic generator.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def compute_features(events: List[Event], cfg: FeatureConfig) -> Features:
    """Convert raw events into attempt-level features.

    This is deterministic and explainable: every feature is directly traceable to events.
    """

    # Basic bookkeeping
    event_count_total = len(events)

    # Client seq gaps
    seqs = [int(ev.get("client_seq", 0)) for ev in events if isinstance(ev.get("client_seq", 0), int)]
    seqs_sorted = sorted(set(seqs))
    client_seq_gaps = 0
    if seqs_sorted:
        expected = set(range(seqs_sorted[0], seqs_sorted[-1] + 1))
        client_seq_gaps = len(expected - set(seqs_sorted))

    attempt_id = str(events[0].get("attempt_id", "")) if events else ""
    user_id = str(events[0].get("user_id", "")) if events else ""
    assessment_id = str(events[0].get("assessment_id", "")) if events else ""

    # Identify attempt boundaries
    start_ts: Optional[datetime] = None
    submit_ts: Optional[datetime] = None
    for ev in events:
        if ev.get("event_type") == "attempt_state":
            action = (ev.get("payload") or {}).get("action")
            t = _parse_ts(ev["occurred_at"])
            if action == "start" and start_ts is None:
                start_ts = t
            if action == "submit":
                submit_ts = t

    if start_ts is None and events:
        start_ts = _parse_ts(events[0]["occurred_at"])
    if submit_ts is None and events:
        submit_ts = _parse_ts(events[-1]["occurred_at"])

    has_submit_event = any(
        ev.get("event_type") == "attempt_state" and (ev.get("payload") or {}).get("action") == "submit" for ev in events
    )

    attempt_duration_s = max(0.0, (_parse_ts(events[-1]["occurred_at"]) - start_ts).total_seconds()) if events and start_ts else 0.0
    if submit_ts and start_ts:
        attempt_duration_s = max(0.0, (submit_ts - start_ts).total_seconds())

    # Question times: pair enter/leave.
    enter_ts_by_q: Dict[str, datetime] = {}
    question_times: List[float] = []
    for ev in events:
        if ev.get("event_type") != "question_view":
            continue
        payload = ev.get("payload") or {}
        qid = str(payload.get("question_id", ""))
        action = payload.get("action")
        t = _parse_ts(ev["occurred_at"])
        if action == "enter":
            enter_ts_by_q[qid] = t
        elif action == "leave":
            if qid in enter_ts_by_q:
                dt = (t - enter_ts_by_q[qid]).total_seconds()
                if dt >= 0:
                    question_times.append(dt)
                enter_ts_by_q.pop(qid, None)

    questions_seen = len(question_times)

    if question_times:
        mean_q = statistics.mean(question_times)
        median_q = statistics.median(question_times)
        std_q = statistics.pstdev(question_times) if len(question_times) > 1 else 0.0
    else:
        mean_q = median_q = std_q = 0.0

    time_per_question_cv = _safe_div(std_q, mean_q)

    p10 = _quantile(question_times, 0.10)
    p50 = _quantile(question_times, 0.50)
    p90 = _quantile(question_times, 0.90)

    fast_question_ratio = _safe_div(sum(1 for t in question_times if t <= cfg.fast_question_s), max(1, questions_seen))

    # Tab hidden segments from visibility_change events.
    hidden_start: Optional[datetime] = None
    tab_hidden_count = 0
    tab_hidden_total_s = 0.0
    tab_hidden_longest_s = 0.0

    for ev in events:
        if ev.get("event_type") != "visibility_change":
            continue
        state = (ev.get("payload") or {}).get("state")
        t = _parse_ts(ev["occurred_at"])
        if state == "hidden" and hidden_start is None:
            hidden_start = t
            tab_hidden_count += 1
        elif state == "visible" and hidden_start is not None:
            dt = max(0.0, (t - hidden_start).total_seconds())
            tab_hidden_total_s += dt
            tab_hidden_longest_s = max(tab_hidden_longest_s, dt)
            hidden_start = None

    tab_hidden_ratio = _safe_div(tab_hidden_total_s, attempt_duration_s)
    tab_hidden_mean_s = _safe_div(tab_hidden_total_s, tab_hidden_count)

    # Clipboard counts
    copy_count = 0
    paste_count = 0
    paste_by_q: Dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.get("event_type") != "clipboard":
            continue
        payload = ev.get("payload") or {}
        action = payload.get("action")
        if action == "copy":
            copy_count += 1
        elif action == "paste":
            paste_count += 1
            qid = str(payload.get("question_id", ""))
            if qid:
                paste_by_q[qid] += 1

    paste_questions_affected = sum(1 for v in paste_by_q.values() if v > 0)
    paste_per_question = _safe_div(paste_count, max(1, questions_seen))
    if paste_count > 0 and paste_by_q:
        paste_concentration = max(paste_by_q.values()) / paste_count
    else:
        paste_concentration = 0.0

    # Typing bursts
    typing_bursts = 0
    keystrokes_total = 0
    typing_time_total_s = 0.0
    for ev in events:
        if ev.get("event_type") != "typing_burst":
            continue
        payload = ev.get("payload") or {}
        n_keys = int(payload.get("n_keystrokes", 0) or 0)
        dur_ms = int(payload.get("duration_ms", 0) or 0)
        if n_keys <= 0 or dur_ms <= 0:
            continue
        typing_bursts += 1
        keystrokes_total += n_keys
        typing_time_total_s += dur_ms / 1000.0

    keystrokes_per_s_mean = _safe_div(keystrokes_total, typing_time_total_s)
    typing_burst_rate_per_min = _safe_div(typing_bursts, attempt_duration_s / 60.0)

    # Idle segments
    idle_start: Optional[datetime] = None
    idle_count = 0
    idle_total_s = 0.0
    idle_longest_s = 0.0
    idle_spike_count = 0

    for ev in events:
        if ev.get("event_type") != "idle_state":
            continue
        state = (ev.get("payload") or {}).get("state")
        t = _parse_ts(ev["occurred_at"])
        if state == "idle" and idle_start is None:
            idle_start = t
            idle_count += 1
        elif state == "active" and idle_start is not None:
            dt = max(0.0, (t - idle_start).total_seconds())
            idle_total_s += dt
            idle_longest_s = max(idle_longest_s, dt)
            if dt >= cfg.idle_spike_s:
                idle_spike_count += 1
            idle_start = None

    idle_ratio = _safe_div(idle_total_s, attempt_duration_s)

    # Answer changes
    answer_change_count = 0
    changed_questions = set()
    answer_change_last_minute_count = 0

    if submit_ts is None and events:
        submit_ts = _parse_ts(events[-1]["occurred_at"])

    last_min_start = submit_ts - timedelta(seconds=60) if submit_ts else None

    for ev in events:
        if ev.get("event_type") != "answer_change":
            continue
        payload = ev.get("payload") or {}
        qid = str(payload.get("question_id", ""))
        if qid:
            changed_questions.add(qid)
        answer_change_count += 1
        if last_min_start is not None:
            t = _parse_ts(ev["occurred_at"])
            if t >= last_min_start:
                answer_change_last_minute_count += 1

    answer_change_per_question = _safe_div(answer_change_count, max(1, questions_seen))

    features: Features = {
        "attempt_id": attempt_id,
        "user_id": user_id,
        "assessment_id": assessment_id,
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_version": 1,
        # timing
        "questions_seen": questions_seen,
        "attempt_duration_s": attempt_duration_s,
        "time_per_question_mean_s": mean_q,
        "time_per_question_median_s": median_q,
        "time_per_question_std_s": std_q,
        "time_per_question_cv": time_per_question_cv,
        "time_per_question_p10_s": p10,
        "time_per_question_p50_s": p50,
        "time_per_question_p90_s": p90,
        "fast_question_ratio": fast_question_ratio,
        # tab
        "tab_hidden_count": tab_hidden_count,
        "tab_hidden_total_s": tab_hidden_total_s,
        "tab_hidden_ratio": tab_hidden_ratio,
        "tab_hidden_mean_s": tab_hidden_mean_s,
        "tab_hidden_longest_s": tab_hidden_longest_s,
        # clipboard
        "paste_count": paste_count,
        "copy_count": copy_count,
        "paste_per_question": paste_per_question,
        "paste_questions_affected": paste_questions_affected,
        "paste_concentration": paste_concentration,
        # typing
        "typing_bursts": typing_bursts,
        "keystrokes_total": keystrokes_total,
        "typing_time_total_s": typing_time_total_s,
        "keystrokes_per_s_mean": keystrokes_per_s_mean,
        "typing_burst_rate_per_min": typing_burst_rate_per_min,
        # idle
        "idle_count": idle_count,
        "idle_total_s": idle_total_s,
        "idle_ratio": idle_ratio,
        "idle_longest_s": idle_longest_s,
        "idle_spike_count": idle_spike_count,
        # answer changes
        "answer_change_count": answer_change_count,
        "answer_change_per_question": answer_change_per_question,
        "answer_change_last_minute_count": answer_change_last_minute_count,
        "changed_questions_count": len(changed_questions),
        # data quality
        "event_count_total": event_count_total,
        "client_seq_gaps": client_seq_gaps,
        "has_submit_event": has_submit_event,
    }

    return features
