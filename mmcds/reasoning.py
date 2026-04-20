from __future__ import annotations

from typing import Dict, List, Tuple

from .types import ReasoningConfig, Signals


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def combine_signals(signals: Signals, cfg: ReasoningConfig) -> Dict[str, float]:
    """Combine signals with weights + explicit gating.

    Key properties:
    - No single signal can push risk to HIGH alone.
    - Multiple independent signals are required for HIGH.
    """

    scores: Dict[str, float] = {k: float(v.get("score", 0.0) or 0.0) for k, v in signals.items()}

    weights = dict(cfg.weights)

    # Weighted average (normalized).
    total_w = sum(weights.get(k, 0.0) for k in scores.keys())
    if total_w <= 0:
        base = 0.0
    else:
        base = sum(weights.get(k, 0.0) * scores.get(k, 0.0) for k in scores.keys()) / total_w

    elevated = [k for k, s in scores.items() if s >= cfg.elevated_threshold]
    strong = [k for k, s in scores.items() if s >= cfg.strong_threshold]
    support = [k for k, s in scores.items() if s >= cfg.support_threshold]

    # Multi-signal reinforcement (explainable):
    # If at least two independent signals are moderately elevated, slightly boost the score
    # based on the average of the top-2 signals. This helps capture "borderline" patterns
    # where no single signal is extreme but multiple are consistently suspicious.
    bonus = 0.0
    if len(support) >= 2:
        top2 = sorted(scores.values(), reverse=True)[:2]
        bonus = 0.20 * ((top2[0] + top2[1]) / 2.0)
        base = min(1.0, base + bonus)

    # Gating: if fewer than N elevated signals, cap the combined score.
    gated = base
    if len(elevated) < cfg.require_elevated_signals:
        # You can still be MEDIUM based on weak, broad evidence.
        gated = min(gated, 0.66)
    if len(elevated) == 0:
        gated = min(gated, 0.45)

    # If multiple signals are strong, allow the score to rise slightly.
    if len(strong) >= (cfg.require_elevated_signals + 1):
        gated = min(1.0, gated + 0.05)

    return {
        "base_score": float(_clamp01(base)),
        "combined_score": float(_clamp01(gated)),
        "elevated_count": float(len(elevated)),
        "strong_count": float(len(strong)),
        "support_count": float(len(support)),
        "bonus": float(bonus),
    }
