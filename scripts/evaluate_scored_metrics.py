"""Evaluate MMCDS scored outputs against synthetic ground truth.

This script is designed for *interpretability-first* evaluation of MMCDS v1.

It computes:
- Accuracy
- Confusion matrix
- Precision / Recall / F1 (per class; highlights HIGH)
- False Positive Rate: normal users flagged as MEDIUM/HIGH
- False Negative Rate: suspicious users missed (pred != HIGH)
- Confidence diagnostics + simple calibration table (ECE)
- Optional signal/feature pattern diagnostics for FP/FN

Mapping assumed (per spec):
  normal -> LOW
  borderline -> MEDIUM
  suspicious -> HIGH

Run:
  python scripts/evaluate_scored_metrics.py --scored data/synth/scored.jsonl

Notes:
- Input format is JSON Lines. Each row should include ground truth as `label` or
  `user_type`, and prediction as `risk` or `risk_level`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CLASSES: Tuple[str, str, str] = ("LOW", "MEDIUM", "HIGH")


GROUND_TRUTH_TO_RISK: Dict[str, str] = {
    "normal": "LOW",
    "borderline": "MEDIUM",
    "suspicious": "HIGH",
}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm_label(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def _norm_risk(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().upper()


def _get_truth_label(row: Mapping[str, Any]) -> str:
    # Ground truth is called `label` in this repo's scoring script.
    # The user spec also uses `user_type`.
    return _norm_label(row.get("label") or row.get("user_type"))


def _get_pred_risk(row: Mapping[str, Any]) -> str:
    return _norm_risk(row.get("risk") or row.get("risk_level"))


def _get_promoted_by_pattern(row: Mapping[str, Any]) -> bool:
    v = row.get("promoted_by_pattern")
    if isinstance(v, bool):
        return v

    # Back-compat heuristic for older scored JSONL files.
    # We count a row as "promoted" when the final predicted risk is MEDIUM and
    # a MEDIUM-only pattern is present with sufficient strength.
    pred = _get_pred_risk(row)
    if pred != "MEDIUM":
        return False

    medium_only = {"medium_tab_low_time_no_paste", "medium_idle_fast_no_paste"}
    patterns = row.get("patterns")
    if not isinstance(patterns, list):
        return False
    for p in patterns:
        if not isinstance(p, dict):
            continue
        pt = p.get("pattern_type")
        if pt not in medium_only:
            continue
        if _safe_float(p.get("strength"), 0.0) >= 0.60:
            return True
    return False


def _truth_to_risk(truth_label: str) -> str:
    return GROUND_TRUTH_TO_RISK.get(truth_label, "")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _fmt_pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf"
    return f"{x:.4f}"


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


def _per_class_metrics(y_true: Sequence[str], y_pred: Sequence[str], *, classes: Sequence[str]) -> Dict[str, ClassMetrics]:
    out: Dict[str, ClassMetrics] = {}
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[c] = ClassMetrics(precision=precision, recall=recall, f1=f1, support=support)
    return out


def _confusion(y_true: Sequence[str], y_pred: Sequence[str], *, classes: Sequence[str]) -> List[List[int]]:
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0 for _ in classes] for _ in classes]
    for yt, yp in zip(y_true, y_pred):
        if yt not in idx or yp not in idx:
            continue
        m[idx[yt]][idx[yp]] += 1
    return m


def _print_confusion(matrix: List[List[int]], *, classes: Sequence[str]) -> None:
    # Simple aligned text table.
    colw = max(6, max(len(c) for c in classes) + 2)
    header = "".ljust(colw) + "".join(c.ljust(colw) for c in classes)
    print("Confusion Matrix (rows=true, cols=pred)")
    print(header)
    for c, row in zip(classes, matrix):
        print(c.ljust(colw) + "".join(str(v).ljust(colw) for v in row))


def _ece(conf: Sequence[float], correct: Sequence[bool], *, bins: int = 10) -> Tuple[float, List[Tuple[str, int, float, float]]]:
    if not conf:
        return 0.0, []
    # Bin edges: [0,1] split into equal-width bins.
    bucket: List[List[int]] = [[] for _ in range(bins)]
    for i, c in enumerate(conf):
        c = min(max(c, 0.0), 1.0)
        b = min(bins - 1, int(c * bins))
        bucket[b].append(i)

    rows: List[Tuple[str, int, float, float]] = []
    ece = 0.0
    n = len(conf)
    for b, idxs in enumerate(bucket):
        if not idxs:
            continue
        lo = b / bins
        hi = (b + 1) / bins
        avg_conf = sum(conf[i] for i in idxs) / len(idxs)
        acc = sum(1.0 for i in idxs if correct[i]) / len(idxs)
        w = len(idxs) / n
        ece += w * abs(acc - avg_conf)
        rows.append((f"[{lo:.1f},{hi:.1f})", len(idxs), avg_conf, acc))
    return ece, rows


def _extract_numeric_signals(row: Mapping[str, Any]) -> Dict[str, float]:
    """Extract a compact set of interpretable signal values.

    We prefer signal scores + a small set of stable components.
    """
    out: Dict[str, float] = {}
    signals = row.get("signals")
    if isinstance(signals, dict):
        for cat, payload in signals.items():
            if isinstance(payload, dict):
                out[f"signal.{cat}.score"] = _safe_float(payload.get("score"), default=float("nan"))
                comps = payload.get("components")
                if isinstance(comps, dict):
                    # A compact whitelist of common components.
                    for k in (
                        "fast_ratio",
                        "p10_time_s",
                        "cv",
                        "tab_hidden_ratio",
                        "tab_hidden_per_min",
                        "paste_per_question",
                        "paste_count",
                        "paste_questions_affected",
                        "keystrokes_per_s_mean",
                        "typing_bursts_per_min",
                        "idle_ratio",
                        "idle_spike_count",
                        "answer_change_per_question",
                        "answer_change_last_minute_count",
                        "answer_change_count",
                    ):
                        if k in comps:
                            out[f"signal.{cat}.{k}"] = _safe_float(comps.get(k), default=float("nan"))

    # Some stable features are also useful for debugging (duration/questions).
    feats = row.get("features")
    if isinstance(feats, dict):
        for k in ("questions_seen", "attempt_duration_s", "fast_question_ratio", "paste_per_question", "tab_hidden_ratio"):
            if k in feats:
                out[f"feature.{k}"] = _safe_float(feats.get(k), default=float("nan"))

    return out


def _mean(values: Iterable[float]) -> float:
    xs = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _summarize_signal_deltas(rows: Sequence[Mapping[str, Any]], *, y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    """Print a small, practical diagnostic of signal patterns in FP/FN."""
    by_group: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row, yt, yp in zip(rows, y_true, y_pred):
        if yt not in CLASSES or yp not in CLASSES:
            continue
        correct = yt == yp
        by_group["correct" if correct else "incorrect"].append(_extract_numeric_signals(row))
        if yt == "LOW" and yp != "LOW":
            by_group["fp_normal"].append(_extract_numeric_signals(row))
        if yt == "HIGH" and yp != "HIGH":
            by_group["fn_suspicious"].append(_extract_numeric_signals(row))

    def mean_map(dicts: List[Dict[str, float]]) -> Dict[str, float]:
        keys = sorted({k for d in dicts for k in d.keys()})
        out: Dict[str, float] = {}
        for k in keys:
            out[k] = _mean(d.get(k, float("nan")) for d in dicts)
        return out

    if not by_group:
        print("(no signals/features available for diagnostics)")
        return

    means_correct = mean_map(by_group.get("correct", []))
    means_incorrect = mean_map(by_group.get("incorrect", []))
    means_fp = mean_map(by_group.get("fp_normal", []))
    means_fn = mean_map(by_group.get("fn_suspicious", []))

    def top_deltas(a: Dict[str, float], b: Dict[str, float], *, k: int = 8) -> List[Tuple[str, float, float, float]]:
        keys = sorted(set(a.keys()) | set(b.keys()))
        scored: List[Tuple[str, float, float, float]] = []
        for key in keys:
            va, vb = a.get(key, float("nan")), b.get(key, float("nan"))
            if any(math.isnan(v) or math.isinf(v) for v in (va, vb)):
                continue
            scored.append((key, va, vb, va - vb))
        scored.sort(key=lambda t: abs(t[3]), reverse=True)
        return scored[:k]

    print("\nSignal/feature diagnostics (means; higher delta => stronger separation)")
    if by_group.get("incorrect") and by_group.get("correct"):
        print("Top differences: incorrect minus correct")
        for key, va, vb, d in top_deltas(means_incorrect, means_correct):
            print(f"  {key:34s} incorrect={_fmt(va):>8s} correct={_fmt(vb):>8s} delta={_fmt(d):>8s}")
    if by_group.get("fp_normal"):
        print("\nTop differences: fp_normal minus correct")
        for key, va, vb, d in top_deltas(means_fp, means_correct):
            print(f"  {key:34s} fp_normal={_fmt(va):>8s} correct={_fmt(vb):>8s} delta={_fmt(d):>8s}")
    if by_group.get("fn_suspicious"):
        print("\nTop differences: fn_suspicious minus correct")
        for key, va, vb, d in top_deltas(means_fn, means_correct):
            print(f"  {key:34s} fn_suspicious={_fmt(va):>8s} correct={_fmt(vb):>8s} delta={_fmt(d):>8s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="Path to scored.jsonl")
    ap.add_argument("--include_diagnostics", action="store_true", help="Print signal/feature diagnostics")
    ap.add_argument("--bins", type=int, default=10, help="Calibration bins (default: 10)")
    args = ap.parse_args()

    rows = _load_jsonl(args.scored)
    if not rows:
        raise SystemExit("No rows in scored file")

    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []
    eval_rows: List[Mapping[str, Any]] = []

    skipped = 0
    unknown_truth = Counter()
    unknown_pred = Counter()
    for row in rows:
        truth_label = _get_truth_label(row)
        truth_risk = _truth_to_risk(truth_label)
        pred_risk = _get_pred_risk(row)

        if truth_risk not in CLASSES:
            unknown_truth[truth_label or "<missing>"] += 1
        if pred_risk not in CLASSES:
            unknown_pred[pred_risk or "<missing>"] += 1

        if truth_risk not in CLASSES or pred_risk not in CLASSES:
            skipped += 1
            continue

        y_true.append(truth_risk)
        y_pred.append(pred_risk)
        confidences.append(min(max(_safe_float(row.get("confidence"), 0.0), 0.0), 1.0))
        eval_rows.append(row)

    if not y_true:
        raise SystemExit("No usable rows after mapping labels and risks")

    n = len(y_true)
    correct = [yt == yp for yt, yp in zip(y_true, y_pred)]

    acc = sum(1 for c in correct if c) / n
    m = _confusion(y_true, y_pred, classes=CLASSES)
    per = _per_class_metrics(y_true, y_pred, classes=CLASSES)

    # Domain-specific rates.
    n_normal = sum(1 for yt in y_true if yt == "LOW")
    fp_normal = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "LOW" and yp != "LOW")
    fpr_normal = fp_normal / n_normal if n_normal else 0.0

    n_susp = sum(1 for yt in y_true if yt == "HIGH")
    fn_susp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "HIGH" and yp != "HIGH")
    fnr_susp = fn_susp / n_susp if n_susp else 0.0

    print(f"Rows in file: {len(rows)}")
    print(f"Rows evaluated: {n}")
    if skipped:
        print(f"Rows skipped (unknown/missing labels/risks): {skipped}")
        if unknown_truth:
            print(f"  Unknown truth labels: {dict(unknown_truth)}")
        if unknown_pred:
            print(f"  Unknown predicted risk values: {dict(unknown_pred)}")

    print("\nSTEP 1 — Performance")
    print(f"Accuracy: {acc:.4f} ({_fmt_pct(acc)})")
    _print_confusion(m, classes=CLASSES)

    print("\nPer-class metrics")
    for c in CLASSES:
        cm = per[c]
        print(
            f"  {c:6s} support={cm.support:4d}  precision={cm.precision:.4f}  recall={cm.recall:.4f}  f1={cm.f1:.4f}"
        )

    high = per["HIGH"]
    print("\nFocus: HIGH risk (suspicious -> HIGH)")
    print(f"  precision_HIGH={high.precision:.4f}  recall_HIGH={high.recall:.4f}  f1_HIGH={high.f1:.4f}")

    print("\nError rates aligned to your definitions")
    print(f"  False Positive Rate (normal flagged MEDIUM/HIGH): {fpr_normal:.4f} ({_fmt_pct(fpr_normal)})")
    print(f"  False Negative Rate (suspicious missed; pred != HIGH): {fnr_susp:.4f} ({_fmt_pct(fnr_susp)})")

    # MEDIUM-only promotion diagnostics (helps validate the promotion layer without changing HIGH logic)
    promoted_flags = [_get_promoted_by_pattern(r) for r in eval_rows]
    promoted_total = sum(1 for p in promoted_flags if p)
    pred_medium_total = sum(1 for yp in y_pred if yp == "MEDIUM")
    print("\nPromotion diagnostics")
    print(f"  promoted_by_pattern_total={promoted_total} ({_fmt_pct(promoted_total / n)})")
    if pred_medium_total:
        print(f"  promoted_by_pattern_share_of_pred_MEDIUM={promoted_total / pred_medium_total:.4f} ({_fmt_pct(promoted_total / pred_medium_total)})")
    # Breakdown by ground truth (normal/borderline/suspicious)
    by_truth = {c: 0 for c in CLASSES}
    for yt, promoted in zip(y_true, promoted_flags):
        if promoted and yt in by_truth:
            by_truth[yt] += 1
    print(
        "  promoted_by_pattern_by_truth="
        + ", ".join(f"{c}={by_truth[c]}" for c in CLASSES)
    )

    # Confidence diagnostics / calibration.
    print("\nConfidence diagnostics")
    avg_conf = sum(confidences) / n
    avg_conf_correct = sum(c for c, ok in zip(confidences, correct) if ok) / max(1, sum(correct))
    avg_conf_incorrect = sum(c for c, ok in zip(confidences, correct) if not ok) / max(1, n - sum(correct))
    print(f"  mean_confidence_all={avg_conf:.4f}")
    print(f"  mean_confidence_correct={avg_conf_correct:.4f}")
    print(f"  mean_confidence_incorrect={avg_conf_incorrect:.4f}")

    ece, cal_rows = _ece(confidences, correct, bins=max(2, args.bins))
    print(f"  ECE (lower is better)={ece:.4f}")
    if cal_rows:
        print("  Calibration by confidence bin")
        for b, cnt, avg_c, acc_b in cal_rows:
            print(f"    {b:10s} n={cnt:4d}  avg_conf={avg_c:.3f}  empirical_acc={acc_b:.3f}")

    # Mistake breakdown.
    off_diag = Counter((yt, yp) for yt, yp in zip(y_true, y_pred) if yt != yp)
    if off_diag:
        print("\nMost common mistakes")
        for (yt, yp), cnt in off_diag.most_common(8):
            print(f"  true={yt:6s} pred={yp:6s}  count={cnt}")

    if args.include_diagnostics:
        _summarize_signal_deltas(rows, y_true=y_true, y_pred=y_pred)


if __name__ == "__main__":
    main()
