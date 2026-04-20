"""Constrained tuning for MMCDS risk thresholds and reasoning knobs.

This script does NOT retrain an ML model. It grid-searches a small,
interpretable parameter space (thresholds/weights/bonus) using the
already-computed per-attempt signal scores in a scored JSONL file.

Goal: choose parameters that maximize suspicious (HIGH) recall under a
cap on normal false positive rate (normal flagged MEDIUM/HIGH).

Run:
  python scripts/tune_constrained.py --scored data/tune/scored.jsonl --max_fpr 0.01

Output:
  - Best configuration found (and a few runners-up)
  - Metrics summary for the best configuration

Notes:
  - Uses only stdlib.
  - Expects each row to have: label, signals.{category}.score
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
Label = Literal["normal", "borderline", "suspicious"]


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def label_to_truth(label: str) -> Optional[RiskLevel]:
    lab = (label or "").strip().lower()
    if lab == "normal":
        return "LOW"
    if lab == "borderline":
        return "MEDIUM"
    if lab == "suspicious":
        return "HIGH"
    return None


@dataclass(frozen=True)
class Candidate:
    # Reasoning
    weights: Dict[str, float]
    support_threshold: float
    elevated_threshold: float
    strong_threshold: float
    require_elevated_signals: int
    bonus_coef: float

    # Risk thresholds
    medium_threshold: float
    high_threshold: float

    # Risk logic knobs
    medium_support_min: int
    high_support_min: int
    allow_elevated_and_strong: bool


def combine_signals(scores: Dict[str, float], c: Candidate) -> Tuple[float, int, int, int]:
    """Return (combined_score, support_count, elevated_count, strong_count)."""

    total_w = 0.0
    weighted = 0.0
    for k, s in scores.items():
        w = float(c.weights.get(k, 0.0) or 0.0)
        if w <= 0.0:
            continue
        total_w += w
        weighted += w * float(s)

    base = weighted / total_w if total_w > 0.0 else 0.0

    support = [k for k, s in scores.items() if float(s) >= c.support_threshold]
    elevated = [k for k, s in scores.items() if float(s) >= c.elevated_threshold]
    strong = [k for k, s in scores.items() if float(s) >= c.strong_threshold]

    # Reinforcement bonus (top-2 mean)
    if len(support) >= 2:
        top2 = sorted((float(v) for v in scores.values()), reverse=True)[:2]
        bonus = c.bonus_coef * ((top2[0] + top2[1]) / 2.0)
        base = min(1.0, base + bonus)

    gated = base
    # Gating caps (kept consistent with mmcds.reasoning.combine_signals)
    if len(elevated) < c.require_elevated_signals:
        gated = min(gated, 0.66)
    if len(elevated) == 0:
        gated = min(gated, 0.45)
    if len(strong) >= (c.require_elevated_signals + 1):
        gated = min(1.0, gated + 0.05)

    return float(_clamp01(gated)), len(support), len(elevated), len(strong)


def assign_risk(score: float, support_count: int, elevated_count: int, strong_count: int, c: Candidate) -> RiskLevel:
    if score >= c.high_threshold and (
        elevated_count >= c.require_elevated_signals
        or (c.allow_elevated_and_strong and elevated_count >= 1 and strong_count >= 1)
        or support_count >= c.high_support_min
    ):
        return "HIGH"

    if score >= c.medium_threshold and (
        support_count >= c.medium_support_min or elevated_count >= 1 or strong_count >= 1
    ):
        return "MEDIUM"

    return "LOW"


@dataclass
class Metrics:
    accuracy: float
    fpr_normal_flagged: float
    recall_high: float
    precision_high: float
    recall_medium: float


def evaluate(rows: List[Tuple[RiskLevel, Dict[str, float]]], c: Candidate) -> Metrics:
    n = len(rows)
    if n == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0)

    correct = 0

    # For FPR
    normal_total = 0
    normal_flagged = 0

    # For HIGH precision/recall
    true_high = 0
    pred_high = 0
    true_high_pred_high = 0

    # For MEDIUM recall
    true_medium = 0
    true_medium_pred_medium = 0

    for truth, scores in rows:
        combined, support_count, elevated_count, strong_count = combine_signals(scores, c)
        pred = assign_risk(combined, support_count, elevated_count, strong_count, c)

        if pred == truth:
            correct += 1

        if truth == "LOW":
            normal_total += 1
            if pred != "LOW":
                normal_flagged += 1

        if truth == "HIGH":
            true_high += 1
            if pred == "HIGH":
                true_high_pred_high += 1

        if pred == "HIGH":
            pred_high += 1

        if truth == "MEDIUM":
            true_medium += 1
            if pred == "MEDIUM":
                true_medium_pred_medium += 1

    accuracy = correct / n
    fpr = (normal_flagged / normal_total) if normal_total else 0.0
    recall_high = (true_high_pred_high / true_high) if true_high else 0.0
    precision_high = (true_high_pred_high / pred_high) if pred_high else 0.0
    recall_medium = (true_medium_pred_medium / true_medium) if true_medium else 0.0

    return Metrics(
        accuracy=float(accuracy),
        fpr_normal_flagged=float(fpr),
        recall_high=float(recall_high),
        precision_high=float(precision_high),
        recall_medium=float(recall_medium),
    )


def load_rows(scored_path: str) -> List[Tuple[RiskLevel, Dict[str, float]]]:
    rows: List[Tuple[RiskLevel, Dict[str, float]]] = []
    with open(scored_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            truth = label_to_truth(str(obj.get("label", "")))
            if truth is None:
                continue
            signals = obj.get("signals") or {}
            scores: Dict[str, float] = {}
            for k, v in signals.items():
                try:
                    scores[str(k)] = float((v or {}).get("score", 0.0) or 0.0)
                except (TypeError, ValueError, AttributeError):
                    scores[str(k)] = 0.0
            rows.append((truth, scores))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Constrained tuner for MMCDS thresholds.")
    p.add_argument("--scored", required=True, help="Path to scored JSONL with signals + label")
    p.add_argument("--max_fpr", type=float, default=0.01, help="Max normal FPR allowed (default: 0.01)")
    p.add_argument(
        "--topk",
        type=int,
        default=8,
        help="How many top candidates to print (default: 8)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = load_rows(args.scored)
    if not rows:
        raise SystemExit("No usable rows found (need label + signals.*.score).")

    # Weight presets (keep small + interpretable)
    weight_presets: List[Dict[str, float]] = [
        # Balanced
        {
            "timing": 0.20,
            "tab": 0.18,
            "clipboard": 0.18,
            "idle": 0.14,
            "answer_changes": 0.15,
            "typing": 0.15,
        },
        # Clipboard+timing emphasis
        {
            "timing": 0.22,
            "tab": 0.17,
            "clipboard": 0.22,
            "idle": 0.12,
            "answer_changes": 0.16,
            "typing": 0.11,
        },
        # Slightly more answer-churn sensitive
        {
            "timing": 0.21,
            "tab": 0.16,
            "clipboard": 0.21,
            "idle": 0.12,
            "answer_changes": 0.20,
            "typing": 0.10,
        },
    ]

    grid = {
        "support_threshold": [0.20, 0.25, 0.30],
        "elevated_threshold": [0.60, 0.65, 0.70],
        "strong_threshold": [0.85],
        "require_elevated_signals": [2],
        "bonus_coef": [0.12, 0.16, 0.20],
        "medium_threshold": [0.12, 0.15, 0.18, 0.20, 0.22],
        "high_threshold": [0.52, 0.55, 0.58, 0.60, 0.62],
        "medium_support_min": [1, 2],
        "high_support_min": [3, 4],
        "allow_elevated_and_strong": [True],
    }

    candidates: List[Tuple[Candidate, Metrics]] = []

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    for weights in weight_presets:
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            # Basic sanity constraints
            if params["medium_threshold"] >= params["high_threshold"]:
                continue

            cand = Candidate(weights=weights, **params)  # type: ignore[arg-type]
            m = evaluate(rows, cand)

            if m.fpr_normal_flagged <= float(args.max_fpr):
                candidates.append((cand, m))

    if not candidates:
        raise SystemExit(f"No candidates satisfied max_fpr={args.max_fpr}.")

    # Sort by: maximize HIGH recall, then accuracy, then MEDIUM recall, then minimize FPR
    def sort_key(item: Tuple[Candidate, Metrics]) -> Tuple[float, float, float, float]:
        _, m = item
        return (
            m.recall_high,
            m.accuracy,
            m.recall_medium,
            -m.fpr_normal_flagged,
        )

    candidates.sort(key=sort_key, reverse=True)

    best_c, best_m = candidates[0]

    print("Constrained tuning results")
    print(f"Rows evaluated: {len(rows)}")
    print(f"Constraint: normal_FPR <= {args.max_fpr:.4f}")
    print()

    def fmt(c: Candidate, m: Metrics) -> str:
        return (
            f"HIGH_recall={m.recall_high:.3f}  acc={m.accuracy:.3f}  MED_recall={m.recall_medium:.3f}  "
            f"normal_FPR={m.fpr_normal_flagged:.4f}  "
            f"med_th={c.medium_threshold:.2f} high_th={c.high_threshold:.2f} "
            f"support_th={c.support_threshold:.2f} elev_th={c.elevated_threshold:.2f} bonus={c.bonus_coef:.2f} "
            f"med_support_min={c.medium_support_min} high_support_min={c.high_support_min} "
            f"weights={c.weights}"
        )

    print("Best:")
    print(fmt(best_c, best_m))
    print()

    print(f"Top {min(args.topk, len(candidates))} candidates:")
    for cand, m in candidates[: int(args.topk)]:
        print(fmt(cand, m))


if __name__ == "__main__":
    main()
