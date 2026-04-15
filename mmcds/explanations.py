from __future__ import annotations

from typing import Dict, List, Tuple

from .types import Features, RiskLevel, Signals


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def _top_signals(signals: Signals, k: int = 3) -> List[Tuple[str, float]]:
    items = [(name, float(data.get("score", 0.0) or 0.0)) for name, data in signals.items()]
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:k]


def generate_explanations(features: Features, signals: Signals, risk: RiskLevel) -> Dict[str, List[str]]:
    """Generate human-readable explanations.

    Output structure is intentionally simple:
    - summary: 1-2 lines
    - reasons: 2-4 lines, each tied to a specific signal + concrete values
    """

    summary: List[str] = []
    reasons: List[str] = []

    questions = int(features.get("questions_seen", 0) or 0)
    duration_s = float(features.get("attempt_duration_s", 0.0) or 0.0)

    summary.append(f"Risk={risk}. Attempt had {questions} questions over ~{duration_s/60.0:.1f} minutes.")

    # Only explain what is materially elevated.
    for name, score in _top_signals(signals, k=4):
        if score < 0.50:
            continue

        if name == "tab":
            r = float(signals[name]["components"].get("tab_hidden_ratio", 0.0) or 0.0)
            per_min = float(signals[name]["components"].get("tab_hidden_per_min", 0.0) or 0.0)
            reasons.append(f"Tab/focus: out-of-tab time {_fmt_pct(r)} with ~{per_min:.2f} hide events/min.")
        elif name == "clipboard":
            paste_count = int(signals[name]["components"].get("paste_count", 0) or 0)
            ppq = float(signals[name]["components"].get("paste_per_question", 0.0) or 0.0)
            affected = int(signals[name]["components"].get("paste_questions_affected", 0) or 0)
            reasons.append(f"Clipboard: {paste_count} paste events across {affected} questions (paste/question={ppq:.2f}).")
        elif name == "timing":
            fast_ratio = float(signals[name]["components"].get("fast_ratio", 0.0) or 0.0)
            p10 = float(signals[name]["components"].get("p10_time_s", 0.0) or 0.0)
            cv = float(signals[name]["components"].get("cv", 0.0) or 0.0)
            reasons.append(f"Timing: fast-question ratio {_fmt_pct(fast_ratio)}; p10 time={p10:.1f}s; variability CV={cv:.2f}.")
        elif name == "idle":
            idle_ratio = float(signals[name]["components"].get("idle_ratio", 0.0) or 0.0)
            spikes = float(signals[name]["components"].get("idle_spike_count", 0.0) or 0.0)
            reasons.append(f"Idle: idle time {_fmt_pct(idle_ratio)} with {int(spikes)} long idle spikes.")
        elif name == "answer_changes":
            cpm = float(signals[name]["components"].get("answer_change_per_question", 0.0) or 0.0)
            last = float(signals[name]["components"].get("answer_change_last_minute_count", 0.0) or 0.0)
            reasons.append(f"Answer changes: {cpm:.2f} changes/question; {int(last)} changes in last minute.")
        elif name == "typing":
            kps = float(signals[name]["components"].get("keystrokes_per_s_mean", 0.0) or 0.0)
            bpm = float(signals[name]["components"].get("typing_bursts_per_min", 0.0) or 0.0)
            reasons.append(f"Typing timing: {kps:.2f} keystrokes/s with {bpm:.2f} typing bursts/min.")
        elif name == "anomaly":
            reasons.append("Pattern anomaly: feature combination is unusual versus typical attempts for this assessment.")

    if not reasons:
        reasons.append("No strong suspicious patterns; behavior stayed within expected ranges.")

    return {"summary": summary, "reasons": reasons}
