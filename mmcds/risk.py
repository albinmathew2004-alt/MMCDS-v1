from __future__ import annotations

from typing import Dict

from .types import ConfidenceConfig, Features, RiskConfig, RiskLevel, Signals


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def assign_risk(score: float, combined: Dict[str, float], cfg: RiskConfig) -> RiskLevel:
    elevated_count = int(combined.get("elevated_count", 0.0) or 0)
    support_count = int(combined.get("support_count", 0.0) or 0)

    # HIGH: require strong multi-signal evidence.
    if score >= cfg.high_threshold and elevated_count >= 2:
        return "HIGH"

    # MEDIUM: require at least two moderately elevated signals.
    if score >= cfg.medium_threshold and support_count >= 2:
        return "MEDIUM"

    return "LOW"


def compute_confidence(features: Features, signals: Signals, combined: Dict[str, float], cfg: ConfidenceConfig) -> float:
    """Confidence reflects evidence strength + data sufficiency.

    It is not "probability of cheating".
    """

    questions = int(features.get("questions_seen", 0) or 0)
    duration_s = float(features.get("attempt_duration_s", 0.0) or 0.0)

    # Data sufficiency: more questions and more time => higher confidence.
    q_term = min(1.0, questions / max(1, cfg.min_questions_for_full_conf))
    d_term = min(1.0, duration_s / max(1.0, cfg.min_duration_s_for_full_conf))
    data_conf = 0.55 * q_term + 0.45 * d_term

    # Evidence: number of elevated signals.
    elevated_count = int(combined.get("elevated_count", 0.0) or 0)
    strong_count = int(combined.get("strong_count", 0.0) or 0)

    evidence_conf = min(1.0, 0.35 * elevated_count + 0.25 * strong_count)

    conf = 0.55 * data_conf + 0.45 * evidence_conf

    # Penalties for missing / low-quality data.
    seq_gaps = int(features.get("client_seq_gaps", 0) or 0)
    conf -= cfg.seq_gap_penalty_per_gap * seq_gaps

    if not bool(features.get("has_submit_event", False)):
        conf -= cfg.missing_submit_penalty

    return float(_clamp01(conf))
