from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from .types import Event, Features, Signals


@dataclass(frozen=True)
class PatternMatch:
    pattern_type: str
    strength: float  # 0..1
    evidence: Dict[str, Any]


@dataclass(frozen=True)
class PatternConfig:
    # Windows
    tab_to_paste_window_s: float = 30.0
    idle_to_tab_window_s: float = 45.0
    paste_to_fast_answer_window_s: float = 30.0

    # MEDIUM-only patterns (promotion rules live in risk_engine)
    medium_tab_switch_min: int = 3
    medium_tab_switch_max: int = 6
    # NOTE: This threshold is paired with a strict out-of-tab ratio guardrail.
    # It is intentionally not "very low" by itself.
    medium_low_avg_time_s: float = 80.0
    medium_tab_low_time_min_out_of_tab_ratio: float = 0.04
    medium_min_questions: int = 10
    idle_to_fast_leave_window_s: float = 35.0
    medium_idle_fast_min_tab_hidden: int = 3
    medium_idle_fast_min_out_of_tab_ratio: float = 0.01

    # Thresholds
    min_idle_spike_s: float = 30.0
    fast_answer_s: float = 10.0

    # Guardrails to reduce false positives
    require_paste_count: int = 2
    require_hidden_events: int = 1


def _parse_ts(ts: str) -> datetime:
    # Keep local parsing here to avoid import cycles.
    # Accept the Z suffix written by the synthetic generator.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _ts(ev: Event) -> Optional[datetime]:
    try:
        return _parse_ts(str(ev.get("occurred_at", "")))
    except Exception:
        return None


def _is_tab_hidden(ev: Event) -> bool:
    return ev.get("event_type") == "visibility_change" and (ev.get("payload") or {}).get("state") == "hidden"


def _is_paste(ev: Event) -> bool:
    return ev.get("event_type") == "clipboard" and (ev.get("payload") or {}).get("action") == "paste"


def _is_idle_end(ev: Event) -> bool:
    return ev.get("event_type") == "idle_state" and (ev.get("payload") or {}).get("state") == "active"


def _is_question_leave(ev: Event) -> bool:
    return ev.get("event_type") == "question_view" and (ev.get("payload") or {}).get("action") == "leave"


def _count_tab_hidden(events: Sequence[Event]) -> int:
    return sum(1 for ev in events if _is_tab_hidden(ev))


def detect_patterns(
    events: List[Event],
    *,
    features: Features,
    signals: Signals,
    cfg: PatternConfig,
) -> List[PatternMatch]:
    """Detect higher-order cheating patterns using temporal rules.

    Patterns are designed to be high precision (low FP): they require multiple
    different kinds of events within tight time windows.
    """

    patterns: List[PatternMatch] = []

    paste_count = int(features.get("paste_count", 0) or 0)
    if paste_count < cfg.require_paste_count:
        # Most strong patterns require paste activity.
        paste_gate = False
    else:
        paste_gate = True

    # PATTERN A: idle spike -> tab hidden -> paste -> fast answer
    # We approximate "idle spike" from features (idle_spike_count) and then
    # look for the sequence in raw events.
    if paste_gate and int(features.get("idle_spike_count", 0) or 0) >= 1 and _count_tab_hidden(events) >= cfg.require_hidden_events:
        # Build a quick index of candidate event times
        times: List[tuple[int, datetime]] = []
        for i, ev in enumerate(events):
            t = _ts(ev)
            if t is None:
                continue
            times.append((i, t))

        # Find paste events, then look backwards/forwards in windows
        for idx, t_paste in times:
            if not _is_paste(events[idx]):
                continue

            # Tab hidden within tab_to_paste_window before paste
            tab_ok = False
            t_tab: Optional[datetime] = None
            for j, t in reversed(times[: times.index((idx, t_paste))]):
                if (t_paste - t).total_seconds() > cfg.tab_to_paste_window_s:
                    break
                if _is_tab_hidden(events[j]):
                    tab_ok = True
                    t_tab = t
                    break

            if not tab_ok:
                continue

            # Idle end within idle_to_tab_window before the tab hidden
            idle_ok = False
            t_idle_end: Optional[datetime] = None
            if t_tab is not None:
                for j, t in reversed(times[: times.index((idx, t_paste))]):
                    if (t_tab - t).total_seconds() > cfg.idle_to_tab_window_s:
                        break
                    if _is_idle_end(events[j]):
                        idle_ok = True
                        t_idle_end = t
                        break

            if not idle_ok:
                continue

            # Fast answer within paste_to_fast_answer_window after paste
            fast_ok = False
            t_leave: Optional[datetime] = None
            for j, t in times[times.index((idx, t_paste)) + 1 :]:
                if (t - t_paste).total_seconds() > cfg.paste_to_fast_answer_window_s:
                    break
                if _is_question_leave(events[j]):
                    fast_ok = True
                    t_leave = t
                    break

            if not fast_ok:
                continue

            # Strength: incorporate signal strength for timing+clipboard
            timing = float((signals.get("timing") or {}).get("score", 0.0) or 0.0)
            clipboard = float((signals.get("clipboard") or {}).get("score", 0.0) or 0.0)
            tab = float((signals.get("tab") or {}).get("score", 0.0) or 0.0)
            idle = float((signals.get("idle") or {}).get("score", 0.0) or 0.0)
            strength = min(1.0, 0.30 + 0.25 * clipboard + 0.20 * timing + 0.15 * tab + 0.10 * idle)

            patterns.append(
                PatternMatch(
                    pattern_type="idle_tab_paste_fast",
                    strength=float(strength),
                    evidence={
                        "idle_end_at": t_idle_end.isoformat() if t_idle_end else None,
                        "tab_hidden_at": t_tab.isoformat() if t_tab else None,
                        "paste_at": t_paste.isoformat(),
                        "question_leave_at": t_leave.isoformat() if t_leave else None,
                        "paste_count": paste_count,
                    },
                )
            )
            break

    # PATTERN B: high tab switching + low time (non-temporal, high-level)
    # Require both signals to be elevated to avoid false positives.
    tab_s = float((signals.get("tab") or {}).get("score", 0.0) or 0.0)
    timing_s = float((signals.get("timing") or {}).get("score", 0.0) or 0.0)
    if tab_s >= 0.70 and timing_s >= 0.70:
        patterns.append(
            PatternMatch(
                pattern_type="high_tab_low_time",
                strength=float(min(1.0, 0.40 + 0.35 * tab_s + 0.25 * timing_s)),
                evidence={
                    "tab_score": tab_s,
                    "timing_score": timing_s,
                    "tab_hidden_count": int(features.get("tab_hidden_count", 0) or 0),
                    "p10_time_s": float(features.get("time_per_question_p10_s", 0.0) or 0.0),
                },
            )
        )

    # PATTERN C (MEDIUM-only): moderate tab switching + low average time + no paste
    # This is intended to be a borderline indicator (e.g., quick switching + fast pace)
    # but is NOT strong enough for HIGH by itself.
    paste_count0 = int(features.get("paste_count", 0) or 0)
    q_seen = int(features.get("questions_seen", 0) or 0)
    avg_time = float(features.get("time_per_question_mean_s", 0.0) or 0.0)
    tab_count = int(features.get("tab_hidden_count", 0) or 0)
    out_of_tab_ratio = float(features.get("tab_hidden_ratio", 0.0) or 0.0)
    if (
        paste_count0 == 0
        and q_seen >= cfg.medium_min_questions
        and cfg.medium_tab_switch_min <= tab_count <= cfg.medium_tab_switch_max
        and out_of_tab_ratio >= cfg.medium_tab_low_time_min_out_of_tab_ratio
        and avg_time > 0.0
        and avg_time <= cfg.medium_low_avg_time_s
    ):
        # Strength increases as time decreases and as tab_count sits near the middle of the band.
        band_center = (cfg.medium_tab_switch_min + cfg.medium_tab_switch_max) / 2.0
        tab_term = 1.0 - min(1.0, abs(tab_count - band_center) / max(1.0, band_center))
        time_term = min(1.0, (cfg.medium_low_avg_time_s - avg_time) / max(1.0, cfg.medium_low_avg_time_s))
        out_term = min(1.0, out_of_tab_ratio / max(1e-6, cfg.medium_tab_low_time_min_out_of_tab_ratio))
        strength = min(0.85, 0.40 + 0.20 * tab_term + 0.25 * time_term + 0.15 * out_term)
        patterns.append(
            PatternMatch(
                pattern_type="medium_tab_low_time_no_paste",
                strength=float(strength),
                evidence={
                    "tab_hidden_count": tab_count,
                    "tab_hidden_ratio": out_of_tab_ratio,
                    "avg_time_per_question_s": avg_time,
                    "questions_seen": q_seen,
                    "paste_count": paste_count0,
                },
            )
        )

    # PATTERN D (MEDIUM-only): idle spike -> fast answer (no paste)
    # Detect a long idle segment ending, followed soon by a fast question leave.
    idle_spike_count = int(features.get("idle_spike_count", 0) or 0)
    if (
        paste_count0 == 0
        and idle_spike_count >= 1
        and q_seen >= cfg.medium_min_questions
        and (
            tab_count >= cfg.medium_idle_fast_min_tab_hidden
            or out_of_tab_ratio >= cfg.medium_idle_fast_min_out_of_tab_ratio
        )
    ):
        last_enter_by_q: Dict[str, datetime] = {}
        idle_end_times: List[datetime] = []

        for ev in events:
            t = _ts(ev)
            if t is None:
                continue
            et = ev.get("event_type")
            payload = ev.get("payload") or {}

            if et == "idle_state" and payload.get("state") == "active":
                idle_end_times.append(t)
                continue

            if et == "question_view":
                qid = str(payload.get("question_id", ""))
                action = payload.get("action")
                if not qid:
                    continue
                if action == "enter":
                    last_enter_by_q[qid] = t
                elif action == "leave":
                    enter_t = last_enter_by_q.get(qid)
                    if enter_t is None:
                        continue
                    q_dt = max(0.0, (t - enter_t).total_seconds())
                    if q_dt <= cfg.fast_answer_s:
                        # Is this leave close to any recent idle end?
                        for t_idle_end in reversed(idle_end_times[-5:]):
                            if 0.0 <= (t - t_idle_end).total_seconds() <= cfg.idle_to_fast_leave_window_s:
                                timing_s2 = float((signals.get("timing") or {}).get("score", 0.0) or 0.0)
                                time_term2 = max(0.0, 1.0 - (q_dt / max(1e-6, cfg.fast_answer_s)))
                                idle_term2 = min(1.0, idle_spike_count / 2.0)
                                strength2 = min(0.85, 0.55 + 0.15 * idle_term2 + 0.15 * time_term2 + 0.05 * timing_s2)
                                patterns.append(
                                    PatternMatch(
                                        pattern_type="medium_idle_fast_no_paste",
                                        strength=float(strength2),
                                        evidence={
                                            "idle_end_at": t_idle_end.isoformat(),
                                            "fast_leave_at": t.isoformat(),
                                            "question_id": qid,
                                            "question_time_s": q_dt,
                                            "idle_spike_count": idle_spike_count,
                                            "paste_count": paste_count0,
                                        },
                                    )
                                )
                                break
                    # keep last_enter_by_q; multiple leaves possible but harmless
        

    return patterns
