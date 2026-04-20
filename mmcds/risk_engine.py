from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .event_processor import normalize_events
from .explanations import generate_explanations
from .feature_engineering import build_features
from .pattern_engine import PatternConfig, PatternMatch, detect_patterns
from .reasoning import combine_signals
from .risk import assign_risk, compute_confidence
from .signal_detection import WeightedSignals, detect_signals
from .types import Event, Features, RiskLevel, ScoringConfig, Signals


@dataclass(frozen=True)
class RiskResult:
    attempt_id: str
    risk: RiskLevel
    confidence: float
    confidence_score: float
    promoted_by_pattern: bool
    base_score: float
    combined_score: float
    signals: Signals
    patterns: List[PatternMatch]
    explanation: Dict[str, List[str]]
    explanation_text: str


@dataclass(frozen=True)
class ConfidenceScoreConfig:
    """Evidence-based confidence score.

    This is separate from `mmcds.risk.compute_confidence`, which is primarily
    a data-sufficiency confidence.

    Evidence confidence is based on:
    - how many signals are meaningfully active
    - how strong the top signals are
    - whether strong temporal patterns were detected
    """

    support_threshold: float = 0.20
    elevated_threshold: float = 0.60
    strong_threshold: float = 0.85

    w_count: float = 0.35
    w_strength: float = 0.45
    w_patterns: float = 0.20


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def compute_confidence_score(
    *,
    signal_scores: Mapping[str, float],
    patterns: List[PatternMatch],
    cfg: ConfidenceScoreConfig,
) -> float:
    scores = [float(v) for v in signal_scores.values()]
    scores.sort(reverse=True)

    support_count = sum(1 for s in scores if s >= cfg.support_threshold)
    elevated_count = sum(1 for s in scores if s >= cfg.elevated_threshold)
    strong_count = sum(1 for s in scores if s >= cfg.strong_threshold)

    # Count term: more independent signals => higher confidence
    count_term = min(1.0, (0.20 * support_count + 0.35 * elevated_count + 0.45 * strong_count) / 2.0)

    # Strength term: average of top-3 (or fewer)
    if scores:
        topk = scores[:3]
        strength_term = sum(topk) / len(topk)
    else:
        strength_term = 0.0

    # Pattern term: strongest pattern strength
    pattern_term = max((p.strength for p in patterns), default=0.0)

    conf = cfg.w_count * count_term + cfg.w_strength * strength_term + cfg.w_patterns * pattern_term
    return float(_clamp01(conf))


def _pattern_summary(patterns: List[PatternMatch]) -> str:
    if not patterns:
        return "No cheating patterns detected."
    top = sorted(patterns, key=lambda p: p.strength, reverse=True)[:2]
    parts = [f"{p.pattern_type} (strength={p.strength:.2f})" for p in top]
    return "Patterns: " + "; ".join(parts)


def score_event_batch(
    raw_events: List[Dict[str, Any]],
    *,
    attempt_id: Optional[str] = None,
    cfg: Optional[ScoringConfig] = None,
    pattern_cfg: Optional[PatternConfig] = None,
    evidence_conf_cfg: Optional[ConfidenceScoreConfig] = None,
) -> RiskResult:
    """Production-style entrypoint: events -> features -> signals -> patterns -> risk."""

    scoring_cfg = cfg or ScoringConfig()
    patt_cfg = pattern_cfg or PatternConfig(
        min_idle_spike_s=scoring_cfg.feature.idle_spike_s,
        fast_answer_s=scoring_cfg.feature.fast_question_s,
    )
    ev_conf_cfg = evidence_conf_cfg or ConfidenceScoreConfig(
        support_threshold=scoring_cfg.reasoning.support_threshold,
        elevated_threshold=scoring_cfg.reasoning.elevated_threshold,
        strong_threshold=scoring_cfg.reasoning.strong_threshold,
    )

    processed = normalize_events(raw_events, attempt_id=attempt_id)
    features: Features = build_features(processed.events, scoring_cfg.feature)

    ws: WeightedSignals = detect_signals(features, scoring_cfg.signal)
    combined = combine_signals(ws.signals, scoring_cfg.reasoning)

    patterns = detect_patterns(processed.events, features=features, signals=ws.signals, cfg=patt_cfg)

    # Decision logic
    # - Keep existing HIGH logic unchanged (no pattern can trigger HIGH)
    # - Allow a strict MEDIUM-only promotion layer (LOW -> MEDIUM) driven by high-precision patterns
    risk = assign_risk(combined["combined_score"], combined, scoring_cfg.risk)

    promoted_by_pattern = False
    if risk == "LOW" and patterns:
        medium_only = {"medium_tab_low_time_no_paste", "medium_idle_fast_no_paste"}
        if any((p.pattern_type in medium_only) and (float(p.strength) >= 0.60) for p in patterns):
            risk = "MEDIUM"
            promoted_by_pattern = True

    confidence = compute_confidence(features, ws.signals, combined, scoring_cfg.confidence)
    confidence_score = compute_confidence_score(signal_scores=ws.scores, patterns=patterns, cfg=ev_conf_cfg)

    # Safeguard: MEDIUM promotion should not look as confident as a strong multi-signal HIGH.
    if promoted_by_pattern and risk == "MEDIUM":
        elevated_count = int(combined.get("elevated_count", 0.0) or 0)
        strong_count = int(combined.get("strong_count", 0.0) or 0)
        if elevated_count == 0 and strong_count == 0:
            confidence_score = min(float(confidence_score), 0.75)

    explanation = generate_explanations(features, ws.signals, risk)
    explanation_text = " ".join(explanation.get("summary", [])) + " " + _pattern_summary(patterns)

    return RiskResult(
        attempt_id=processed.attempt_id,
        risk=risk,
        confidence=float(confidence),
        confidence_score=float(confidence_score),
        promoted_by_pattern=bool(promoted_by_pattern),
        base_score=float(combined["base_score"]),
        combined_score=float(combined["combined_score"]),
        signals=ws.signals,
        patterns=patterns,
        explanation=explanation,
        explanation_text=explanation_text.strip(),
    )
