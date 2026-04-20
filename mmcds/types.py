from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class FeatureConfig:
    # Feature-level thresholds used only for derived feature helpers.
    fast_question_s: float = 10.0
    idle_spike_s: float = 30.0


@dataclass(frozen=True)
class SignalConfig:
    # Each signal is produced via a simple ramp: score = clamp((x - low)/(high-low), 0, 1)
    # Thresholds are intentionally conservative to reduce false positives.

    # Timing
    fast_ratio_low: float = 0.10
    fast_ratio_high: float = 0.35
    p10_time_low_s: float = 9.0
    p10_time_high_s: float = 4.0  # inverted: smaller is more suspicious
    cv_low: float = 0.60
    cv_high: float = 1.20

    # Tab/focus
    tab_hidden_ratio_low: float = 0.08
    tab_hidden_ratio_high: float = 0.25
    tab_hidden_per_min_low: float = 0.25
    tab_hidden_per_min_high: float = 1.20

    # Clipboard
    paste_per_q_low: float = 0.05
    paste_per_q_high: float = 0.35
    paste_q_affected_low: float = 1.0
    paste_q_affected_high: float = 5.0

    # Typing
    kps_low: float = 2.2  # low typing speed can be suspicious, but weakly
    kps_high: float = 0.8
    typing_bursts_per_min_low: float = 0.5
    typing_bursts_per_min_high: float = 0.1

    # Idle
    idle_ratio_low: float = 0.10
    idle_ratio_high: float = 0.28
    idle_spike_count_low: float = 1.0
    idle_spike_count_high: float = 4.0

    # Answer changes
    changes_per_q_low: float = 0.45
    changes_per_q_high: float = 1.20
    last_min_changes_low: float = 2.0
    last_min_changes_high: float = 8.0


@dataclass(frozen=True)
class ReasoningConfig:
    weights: Mapping[str, float] = None  # type: ignore[assignment]

    # gating rules
    elevated_threshold: float = 0.60
    strong_threshold: float = 0.85
    support_threshold: float = 0.20
    require_elevated_signals: int = 2

    # caps to ensure no single signal dominates
    max_single_signal_contribution: float = 0.55

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(
                self,
                "weights",
                {
                    "timing": 0.22,
                    "tab": 0.17,
                    "clipboard": 0.22,
                    "idle": 0.12,
                    "answer_changes": 0.16,
                    "typing": 0.11,
                },
            )


@dataclass(frozen=True)
class RiskConfig:
    # Risk thresholds operate on final combined score after gating/caps.
    medium_threshold: float = 0.18
    high_threshold: float = 0.52


@dataclass(frozen=True)
class ConfidenceConfig:
    # Data sufficiency thresholds
    min_questions_for_full_conf: int = 20
    min_duration_s_for_full_conf: float = 20 * 60

    # Penalties
    seq_gap_penalty_per_gap: float = 0.08
    missing_submit_penalty: float = 0.35


@dataclass(frozen=True)
class ScoringConfig:
    feature: FeatureConfig = FeatureConfig()
    signal: SignalConfig = SignalConfig()
    reasoning: ReasoningConfig = ReasoningConfig()
    risk: RiskConfig = RiskConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()


def json_safe(obj: Any) -> Any:
    # Helper for dumping dataclasses/sets/etc. Keep minimal.
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, set):
        return sorted(list(obj))
    return str(obj)


Event = Dict[str, Any]
Features = Dict[str, Any]
Signals = Dict[str, Any]
