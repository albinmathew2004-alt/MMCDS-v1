from __future__ import annotations

from typing import List

from .features import compute_features
from .types import Event, FeatureConfig, Features


def build_features(events: List[Event], cfg: FeatureConfig) -> Features:
    """Thin wrapper to keep the public pipeline names stable."""

    return compute_features(events, cfg)
