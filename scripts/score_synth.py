"""Score synthetic dataset end-to-end: events -> features -> signals -> reasoning -> risk/confidence -> explanations.

Run:
  python scripts/score_synth.py --events data/synth/events.jsonl --attempts data/synth/attempts.jsonl --out data/synth/scored.jsonl

Output JSONL rows with:
- attempt_id, label (from manifest), combined_score, risk, confidence
- signals (per category)
- explanation (summary + reasons)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List


# Allow running as a plain script without installing the package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmcds.anomaly import anomaly_score, build_baseline_by_assessment
from mmcds.features import compute_features
from mmcds.io import load_attempt_manifest, load_grouped_events
from mmcds.risk_engine import score_event_batch
from mmcds.types import ScoringConfig, json_safe


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score attempts from events JSONL.")
    p.add_argument("--events", type=str, required=True, help="events.jsonl path")
    p.add_argument("--attempts", type=str, required=True, help="attempts.jsonl path")
    p.add_argument("--out", type=str, required=True, help="output scored.jsonl path")
    p.add_argument("--use_anomaly", action="store_true", help="Add optional assessment-calibrated anomaly signal")
    p.add_argument(
        "--legacy", action="store_true", help="Use legacy pipeline (no patterns/confidence_score)."
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = ScoringConfig()

    grouped = load_grouped_events(args.events)
    manifest = load_attempt_manifest(args.attempts)

    anomaly_feature_names = [
        "fast_question_ratio",
        "time_per_question_cv",
        "tab_hidden_ratio",
        "paste_per_question",
        "idle_ratio",
        "answer_change_per_question",
        "typing_burst_rate_per_min",
    ]

    baseline_by_assessment = {}
    if args.use_anomaly:
        all_features = [compute_features(evs, cfg.feature) for evs in grouped.values()]
        baseline_by_assessment = build_baseline_by_assessment(all_features, anomaly_feature_names)
        # Keep anomaly as a small, secondary input.
        cfg.reasoning.weights["anomaly"] = 0.07

    with open(args.out, "w", encoding="utf-8") as f:
        for attempt_id, events in grouped.items():
            if args.legacy:
                features = compute_features(events, cfg.feature)
                from mmcds.signals import score_signals
                from mmcds.reasoning import combine_signals
                from mmcds.risk import assign_risk, compute_confidence
                from mmcds.explanations import generate_explanations

                signals = score_signals(features, cfg.signal)
                if args.use_anomaly:
                    asm = str(features.get("assessment_id", ""))
                    baseline = baseline_by_assessment.get(asm, {})
                    score, zs = anomaly_score(features, baseline, anomaly_feature_names)
                    signals["anomaly"] = {"score": score, "components": {"robust_z": zs}}
                combined = combine_signals(signals, cfg.reasoning)
                risk = assign_risk(combined["combined_score"], combined, cfg.risk)
                confidence = compute_confidence(features, signals, combined, cfg.confidence)
                explanation = generate_explanations(features, signals, risk)
                confidence_score = None
                patterns = None
                explanation_text = None
                data_confidence = None
            else:
                result = score_event_batch(events, attempt_id=attempt_id, cfg=cfg)
                # Keep legacy-expected objects for downstream scripts.
                features = compute_features(events, cfg.feature)
                signals = result.signals
                combined = {"combined_score": result.combined_score, "base_score": result.base_score}
                risk = result.risk
                confidence = result.confidence_score
                data_confidence = result.confidence
                explanation = result.explanation
                confidence_score = result.confidence_score
                patterns = [p.__dict__ for p in result.patterns]
                explanation_text = result.explanation_text
                promoted_by_pattern = bool(getattr(result, "promoted_by_pattern", False))

            label = manifest.get(attempt_id, {}).get("label")
            out_row = {
                "attempt_id": attempt_id,
                "label": label,
                "combined_score": combined["combined_score"],
                "base_score": combined["base_score"],
                "risk": risk,
                "confidence": confidence,
                "data_confidence": (data_confidence if not args.legacy else None),
                "confidence_score": confidence_score,
                "promoted_by_pattern": (promoted_by_pattern if not args.legacy else None),
                "signals": signals,
                "patterns": patterns,
                "explanation": explanation,
                "explanation_text": explanation_text,
                "features": features,
            }
            f.write(json.dumps(out_row, default=json_safe) + "\n")


if __name__ == "__main__":
    main()
