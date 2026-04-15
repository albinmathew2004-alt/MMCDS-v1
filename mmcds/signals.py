from __future__ import annotations

from typing import Any, Dict, Tuple

from .types import Features, SignalConfig, Signals


def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def ramp(x: float, low: float, high: float) -> float:
    """Linear ramp from low->high.

    - x<=low => 0
    - x>=high => 1
    """
    if high == low:
        return 1.0 if x >= high else 0.0
    return clamp01((x - low) / (high - low))


def ramp_inverted(x: float, low: float, high: float) -> float:
    """Inverted ramp where smaller x is more suspicious.

    - x>=low => 0
    - x<=high => 1

    Here, low > high typically.
    """
    if low == high:
        return 1.0 if x <= high else 0.0
    return clamp01((low - x) / (low - high))


def score_signals(features: Features, cfg: SignalConfig) -> Signals:
    """Normalize features into interpretable 0..1 signals.

    Signals are grouped by behavior category to support weighting and explanations.
    """

    duration_s = float(features.get("attempt_duration_s", 0.0) or 0.0)
    duration_min = duration_s / 60.0 if duration_s else 0.0

    # Timing
    fast_ratio = float(features.get("fast_question_ratio", 0.0) or 0.0)
    p10_time = float(features.get("time_per_question_p10_s", 0.0) or 0.0)
    cv = float(features.get("time_per_question_cv", 0.0) or 0.0)

    timing_fast = max(
        ramp(fast_ratio, cfg.fast_ratio_low, cfg.fast_ratio_high),
        ramp_inverted(p10_time, cfg.p10_time_low_s, cfg.p10_time_high_s),
    )
    timing_variance = ramp(cv, cfg.cv_low, cfg.cv_high)
    timing = max(timing_fast * 0.7 + timing_variance * 0.3, timing_variance * 0.85)

    # Tab
    tab_hidden_ratio = float(features.get("tab_hidden_ratio", 0.0) or 0.0)
    tab_hidden_count = float(features.get("tab_hidden_count", 0.0) or 0.0)
    tab_hidden_per_min = (tab_hidden_count / duration_min) if duration_min else 0.0

    tab = max(
        ramp(tab_hidden_ratio, cfg.tab_hidden_ratio_low, cfg.tab_hidden_ratio_high),
        ramp(tab_hidden_per_min, cfg.tab_hidden_per_min_low, cfg.tab_hidden_per_min_high),
    )

    # Clipboard
    paste_per_q = float(features.get("paste_per_question", 0.0) or 0.0)
    paste_q_affected = float(features.get("paste_questions_affected", 0.0) or 0.0)
    clipboard = max(
        ramp(paste_per_q, cfg.paste_per_q_low, cfg.paste_per_q_high),
        ramp(paste_q_affected, cfg.paste_q_affected_low, cfg.paste_q_affected_high),
    )

    # Typing (weak signal; only timing)
    kps = float(features.get("keystrokes_per_s_mean", 0.0) or 0.0)
    bursts_per_min = float(features.get("typing_burst_rate_per_min", 0.0) or 0.0)
    typing = max(
        ramp_inverted(kps, cfg.kps_low, cfg.kps_high),
        ramp_inverted(bursts_per_min, cfg.typing_bursts_per_min_low, cfg.typing_bursts_per_min_high),
    )

    # Idle
    idle_ratio = float(features.get("idle_ratio", 0.0) or 0.0)
    idle_spikes = float(features.get("idle_spike_count", 0.0) or 0.0)
    idle = max(
        ramp(idle_ratio, cfg.idle_ratio_low, cfg.idle_ratio_high),
        ramp(idle_spikes, cfg.idle_spike_count_low, cfg.idle_spike_count_high),
    )

    # Answer changes
    changes_per_q = float(features.get("answer_change_per_question", 0.0) or 0.0)
    last_min_changes = float(features.get("answer_change_last_minute_count", 0.0) or 0.0)
    answer_changes = max(
        ramp(changes_per_q, cfg.changes_per_q_low, cfg.changes_per_q_high),
        ramp(last_min_changes, cfg.last_min_changes_low, cfg.last_min_changes_high),
    )

    # Bundle signals and keep details for explanations.
    signals: Signals = {
        "timing": {
            "score": float(clamp01(timing)),
            "components": {
                "fast_ratio": fast_ratio,
                "p10_time_s": p10_time,
                "cv": cv,
            },
        },
        "tab": {
            "score": float(clamp01(tab)),
            "components": {
                "tab_hidden_ratio": tab_hidden_ratio,
                "tab_hidden_per_min": tab_hidden_per_min,
            },
        },
        "clipboard": {
            "score": float(clamp01(clipboard)),
            "components": {
                "paste_per_question": paste_per_q,
                "paste_questions_affected": paste_q_affected,
                "paste_count": int(features.get("paste_count", 0) or 0),
            },
        },
        "typing": {
            "score": float(clamp01(typing)),
            "components": {
                "keystrokes_per_s_mean": kps,
                "typing_bursts_per_min": bursts_per_min,
            },
        },
        "idle": {
            "score": float(clamp01(idle)),
            "components": {
                "idle_ratio": idle_ratio,
                "idle_spike_count": idle_spikes,
            },
        },
        "answer_changes": {
            "score": float(clamp01(answer_changes)),
            "components": {
                "answer_change_per_question": changes_per_q,
                "answer_change_last_minute_count": last_min_changes,
                "answer_change_count": int(features.get("answer_change_count", 0) or 0),
            },
        },
    }

    return signals
