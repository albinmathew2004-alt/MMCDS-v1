"""Evaluate scored attempts (scored.jsonl) against synthetic labels.

This is a lightweight sanity-check tool for tuning thresholds.

Run:
  python scripts/evaluate_scored.py --scored data/synth/scored.jsonl

Outputs:
- Risk distribution overall and by label
- Flag rates (MEDIUM/HIGH) by label
- Simple score percentiles by label
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = p * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.scored, "r", encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit("No rows")

    overall_risk = Counter(r["risk"] for r in rows)
    by_label_risk: Dict[str, Counter] = defaultdict(Counter)
    by_label_scores: Dict[str, List[float]] = defaultdict(list)

    for r in rows:
        lab = r.get("label") or "unknown"
        by_label_risk[lab][r["risk"]] += 1
        by_label_scores[lab].append(float(r.get("combined_score", 0.0) or 0.0))

    print("rows", len(rows))
    print("overall risk", dict(overall_risk))

    for lab in sorted(by_label_risk.keys()):
        c = by_label_risk[lab]
        n = sum(c.values())
        flagged = c.get("MEDIUM", 0) + c.get("HIGH", 0)
        hi = c.get("HIGH", 0)
        scores = by_label_scores[lab]
        print(
            f"label={lab:11s} n={n:4d} risk={dict(c)} flagged={flagged/n:.3f} high={hi/n:.3f} "
            f"score_p50={pct(scores,0.50):.3f} p90={pct(scores,0.90):.3f} p99={pct(scores,0.99):.3f}"
        )


if __name__ == "__main__":
    main()
