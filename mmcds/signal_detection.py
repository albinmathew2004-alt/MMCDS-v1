from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .signals import score_signals
from .types import Features, ReasoningConfig, SignalConfig, Signals


@dataclass(frozen=True)
class WeightedSignals:
    signals: Signals
    # Convenience view: just scores per signal
    scores: Dict[str, float]


# Optional alias map to support “product language” names.
# Your internal categories remain: timing, tab, clipboard, idle, answer_changes, typing
SIGNAL_ALIASES: Mapping[str, str] = {
    "copy_paste": "clipboard",
    "tab_switch": "tab",
    "low_time": "timing",
    "idle_burst": "idle",
}


def detect_signals(features: Features, cfg: SignalConfig) -> WeightedSignals:
    signals = score_signals(features, cfg)
    scores = {k: float(v.get("score", 0.0) or 0.0) for k, v in signals.items()}
    return WeightedSignals(signals=signals, scores=scores)


def build_weight_config_from_example() -> Dict[str, float]:
    """Return the example weights requested in the prompt, mapped to internal categories."""

    # Example weights in prompt:
    # copy_paste → 0.4, idle_burst → 0.3, tab_switch → 0.2, low_time → 0.1
    return {
        "clipboard": 0.40,
        "idle": 0.30,
        "tab": 0.20,
        "timing": 0.10,
        # keep other categories present but downweighted
        "answer_changes": 0.10,
        "typing": 0.05,
    }


def apply_signal_weights(reasoning_cfg: ReasoningConfig, weights: Mapping[str, float]) -> ReasoningConfig:
    """Return a new ReasoningConfig with weights replaced (dataclass is frozen)."""

    return ReasoningConfig(
        weights=dict(weights),
        elevated_threshold=reasoning_cfg.elevated_threshold,
        strong_threshold=reasoning_cfg.strong_threshold,
        support_threshold=reasoning_cfg.support_threshold,
        require_elevated_signals=reasoning_cfg.require_elevated_signals,
        max_single_signal_contribution=reasoning_cfg.max_single_signal_contribution,
    )
