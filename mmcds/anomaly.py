from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Tuple

from .types import Features


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mad(xs: List[float], med: float) -> float:
    """Median absolute deviation (MAD)."""
    if not xs:
        return 0.0
    dev = [abs(x - med) for x in xs]
    return float(statistics.median(dev))


def build_baseline_by_assessment(feature_rows: Iterable[Features], feature_names: List[str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Build per-assessment robust baselines using (median, MAD) per feature."""

    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for f in feature_rows:
        assessment_id = str(f.get("assessment_id", ""))
        if not assessment_id:
            continue
        for name in feature_names:
            try:
                buckets[assessment_id][name].append(float(f.get(name, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue

    baseline: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for asm, feats in buckets.items():
        baseline[asm] = {}
        for name, values in feats.items():
            med = _median(values)
            mad = _mad(values, med)
            baseline[asm][name] = (med, mad)
    return baseline


def robust_z(x: float, med: float, mad: float) -> float:
    """Robust z-score based on MAD.

    Uses 1.4826*MAD as a robust estimator of stddev (for normal distributions).
    """
    denom = 1.4826 * mad
    if denom <= 1e-9:
        return 0.0
    return (x - med) / denom


def anomaly_score(features: Features, baseline: Mapping[str, Tuple[float, float]], feature_names: List[str]) -> Tuple[float, Dict[str, float]]:
    """Compute an explainable anomaly score in [0,1] from robust z-scores.

    This is not a "cheating probability". It's a measure of how unusual the pattern is
    relative to typical attempts for the same assessment.

    Scoring:
    - Compute abs(z) for each feature
    - Take mean(abs(z))
    - Map to [0,1] via a gentle ramp: mean_abs_z <= 1 => ~0, >= 4 => ~1
    """

    zs: Dict[str, float] = {}
    abs_zs: List[float] = []

    for name in feature_names:
        med, mad = baseline.get(name, (0.0, 0.0))
        x = float(features.get(name, 0.0) or 0.0)
        z = robust_z(x, med, mad)
        zs[name] = float(z)
        abs_zs.append(abs(z))

    mean_abs_z = sum(abs_zs) / len(abs_zs) if abs_zs else 0.0

    # Ramp: 1 -> 0, 4 -> 1
    score = (mean_abs_z - 1.0) / 3.0
    score = 0.0 if score <= 0.0 else (1.0 if score >= 1.0 else score)

    return float(score), zs
