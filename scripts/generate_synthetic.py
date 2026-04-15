"""Synthetic data generator for MMCDS (behavioral cheating detection).

Creates raw event logs matching the STEP 2 schema:
- Append-only JSONL events: events.jsonl
- Attempt-level manifest with labels: attempts.jsonl

Design goals:
- Dependency-free (stdlib only) for low cost and portability.
- Variable number of questions per attempt.
- Three cohorts with realistic, overlapping behavior (normal/borderline/suspicious).
- No single signal deterministically defines the suspicious label.

Run:
  python scripts/generate_synthetic.py --out_dir data/synth --n_attempts 600 --seed 42

Notes:
- Timestamps are UTC ISO strings.
- We intentionally include noise and overlap to reduce "toy" separability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple


Label = Literal["normal", "borderline", "suspicious"]


def utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def poisson(lmbda: float, rng: random.Random) -> int:
    """Simple Poisson sampler (Knuth). Good enough for small lambdas."""
    if lmbda <= 0:
        return 0
    L = math.exp(-lmbda)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def lognorm_seconds(median_s: float, sigma: float, rng: random.Random) -> float:
    """Lognormal where median equals median_s."""
    # If X ~ LogNormal(mu, sigma), median = exp(mu)
    mu = math.log(max(0.01, median_s))
    return max(0.2, rng.lognormvariate(mu, sigma))


def choose_question_count(rng: random.Random, mode: str = "variable") -> int:
    """Variable question counts. Keep realistic for online assessments."""
    if mode == "fixed_20":
        return 20

    # Variable: mostly 12-30, sometimes shorter/longer
    r = rng.random()
    if r < 0.10:
        return rng.randint(8, 12)
    if r < 0.85:
        return rng.randint(12, 30)
    return rng.randint(30, 45)


@dataclass
class AttemptContext:
    attempt_id: str
    user_id: str
    assessment_id: str
    label: Label
    archetype: str
    n_questions: int
    start_at: datetime


def base_client_meta(rng: random.Random) -> Dict[str, Any]:
    browsers = ["chrome", "edge", "firefox"]
    platforms = ["windows", "macos", "linux"]
    return {
        "platform": rng.choice(platforms),
        "browser": rng.choice(browsers),
        "timezone_offset_min": rng.choice([-480, -300, -60, 0, 60, 120, 330]),
        "app_version": "1.0",
    }


def make_event(
    *,
    ctx: AttemptContext,
    event_type: str,
    occurred_at: datetime,
    received_at: datetime,
    client_seq: int,
    payload: Optional[Dict[str, Any]] = None,
    client_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "attempt_id": ctx.attempt_id,
        "user_id": ctx.user_id,
        "assessment_id": ctx.assessment_id,
        "occurred_at": utc_iso(occurred_at),
        "received_at": utc_iso(received_at),
        "client_seq": client_seq,
        "event_type": event_type,
        "client_meta": client_meta or {},
        "payload": payload or {},
    }


def network_latency_ms(rng: random.Random) -> int:
    # Typical: 30-250ms, occasional spikes
    if rng.random() < 0.95:
        return int(rng.uniform(30, 250))
    return int(rng.uniform(250, 1500))


def emit_event(
    events: List[Dict[str, Any]],
    *,
    ctx: AttemptContext,
    event_type: str,
    occurred_at: datetime,
    client_seq_ref: List[int],
    payload: Optional[Dict[str, Any]] = None,
    client_meta: Optional[Dict[str, Any]] = None,
    rng: random.Random,
) -> None:
    client_seq_ref[0] += 1
    received_at = occurred_at + timedelta(milliseconds=network_latency_ms(rng))
    events.append(
        make_event(
            ctx=ctx,
            event_type=event_type,
            occurred_at=occurred_at,
            received_at=received_at,
            client_seq=client_seq_ref[0],
            payload=payload,
            client_meta=client_meta,
        )
    )


def pick_label(rng: random.Random, p_normal: float, p_borderline: float) -> Label:
    r = rng.random()
    if r < p_normal:
        return "normal"
    if r < p_normal + p_borderline:
        return "borderline"
    return "suspicious"


def pick_archetype(label: Label, rng: random.Random) -> str:
    # Ensure suspicious is diverse and not a single-pattern label.
    if label == "normal":
        return rng.choice(["baseline", "focused", "methodical"])
    if label == "borderline":
        return rng.choice(["distracted", "copy_paste_light", "pacey"])
    return rng.choice(["paste_heavy", "tab_heavy", "speed_run", "answer_churn"])


@dataclass
class ProfileParams:
    # Timing
    median_q_s: float
    sigma_q: float
    p_fast_mix: float
    fast_median_q_s: float

    # Tab/focus
    p_tab_hide_per_q: float
    tab_hide_median_s: float
    tab_hide_sigma: float

    # Clipboard
    p_paste_per_q: float
    paste_lambda: float
    p_copy_per_q: float

    # Typing bursts
    typing_burst_lambda_per_q: float
    keystrokes_per_s_mean: float
    keystrokes_per_s_sigma: float

    # Idle
    p_idle_per_q: float
    idle_median_s: float
    idle_sigma: float
    p_idle_spike: float

    # Answer changes
    answer_change_lambda_per_q: float


def params_for(label: Label, archetype: str, rng: random.Random) -> ProfileParams:
    """Parameterize behavior by label+archetype with overlap and noise."""

    # Start from label-level baseline.
    if label == "normal":
        base = ProfileParams(
            median_q_s=rng.uniform(55, 90),
            sigma_q=rng.uniform(0.35, 0.55),
            p_fast_mix=rng.uniform(0.02, 0.08),
            fast_median_q_s=rng.uniform(8, 18),
            p_tab_hide_per_q=rng.uniform(0.02, 0.07),
            tab_hide_median_s=rng.uniform(3, 9),
            tab_hide_sigma=rng.uniform(0.45, 0.75),
            p_paste_per_q=rng.uniform(0.00, 0.03),
            paste_lambda=rng.uniform(0.2, 0.6),
            p_copy_per_q=rng.uniform(0.01, 0.05),
            typing_burst_lambda_per_q=rng.uniform(1.5, 4.0),
            keystrokes_per_s_mean=rng.uniform(3.0, 5.0),
            keystrokes_per_s_sigma=rng.uniform(0.25, 0.45),
            p_idle_per_q=rng.uniform(0.03, 0.10),
            idle_median_s=rng.uniform(4, 12),
            idle_sigma=rng.uniform(0.5, 0.9),
            p_idle_spike=rng.uniform(0.05, 0.10),
            answer_change_lambda_per_q=rng.uniform(0.10, 0.35),
        )
    elif label == "borderline":
        base = ProfileParams(
            median_q_s=rng.uniform(40, 75),
            sigma_q=rng.uniform(0.45, 0.75),
            p_fast_mix=rng.uniform(0.05, 0.18),
            fast_median_q_s=rng.uniform(7, 16),
            p_tab_hide_per_q=rng.uniform(0.05, 0.15),
            tab_hide_median_s=rng.uniform(6, 18),
            tab_hide_sigma=rng.uniform(0.55, 0.95),
            p_paste_per_q=rng.uniform(0.01, 0.10),
            paste_lambda=rng.uniform(0.4, 1.2),
            p_copy_per_q=rng.uniform(0.02, 0.08),
            typing_burst_lambda_per_q=rng.uniform(1.0, 3.5),
            keystrokes_per_s_mean=rng.uniform(2.6, 4.6),
            keystrokes_per_s_sigma=rng.uniform(0.28, 0.50),
            p_idle_per_q=rng.uniform(0.08, 0.22),
            idle_median_s=rng.uniform(8, 25),
            idle_sigma=rng.uniform(0.6, 1.0),
            p_idle_spike=rng.uniform(0.10, 0.22),
            answer_change_lambda_per_q=rng.uniform(0.18, 0.55),
        )
    else:
        # suspicious baseline still overlaps: we avoid a caricature.
        base = ProfileParams(
            median_q_s=rng.uniform(28, 65),
            sigma_q=rng.uniform(0.55, 1.05),
            p_fast_mix=rng.uniform(0.10, 0.35),
            fast_median_q_s=rng.uniform(4.5, 12),
            p_tab_hide_per_q=rng.uniform(0.10, 0.35),
            tab_hide_median_s=rng.uniform(12, 50),
            tab_hide_sigma=rng.uniform(0.7, 1.2),
            p_paste_per_q=rng.uniform(0.05, 0.30),
            paste_lambda=rng.uniform(0.9, 2.8),
            p_copy_per_q=rng.uniform(0.03, 0.10),
            typing_burst_lambda_per_q=rng.uniform(0.4, 2.3),
            keystrokes_per_s_mean=rng.uniform(1.8, 3.8),
            keystrokes_per_s_sigma=rng.uniform(0.30, 0.60),
            p_idle_per_q=rng.uniform(0.10, 0.35),
            idle_median_s=rng.uniform(12, 45),
            idle_sigma=rng.uniform(0.65, 1.2),
            p_idle_spike=rng.uniform(0.18, 0.40),
            answer_change_lambda_per_q=rng.uniform(0.20, 0.80),
        )

    # Archetype tweaks: modify 2-3 dimensions strongly, leave others moderate.
    if archetype == "paste_heavy":
        base.p_paste_per_q = clamp(base.p_paste_per_q + rng.uniform(0.10, 0.22), 0.0, 0.65)
        base.paste_lambda += rng.uniform(0.8, 2.2)
        base.p_fast_mix = clamp(base.p_fast_mix + rng.uniform(0.05, 0.15), 0.0, 0.8)
    elif archetype == "tab_heavy":
        base.p_tab_hide_per_q = clamp(base.p_tab_hide_per_q + rng.uniform(0.12, 0.25), 0.0, 0.9)
        base.tab_hide_median_s += rng.uniform(20, 70)
        base.p_idle_spike = clamp(base.p_idle_spike + rng.uniform(0.05, 0.15), 0.0, 0.95)
    elif archetype == "speed_run":
        base.median_q_s = max(10.0, base.median_q_s - rng.uniform(10, 25))
        base.p_fast_mix = clamp(base.p_fast_mix + rng.uniform(0.15, 0.30), 0.0, 0.95)
        base.typing_burst_lambda_per_q = max(0.1, base.typing_burst_lambda_per_q - rng.uniform(0.3, 1.3))
        base.keystrokes_per_s_mean = max(0.8, base.keystrokes_per_s_mean - rng.uniform(0.4, 1.2))
        base.p_paste_per_q = clamp(base.p_paste_per_q + rng.uniform(0.05, 0.15), 0.0, 0.65)
    elif archetype == "answer_churn":
        base.answer_change_lambda_per_q += rng.uniform(0.4, 1.2)
        base.p_tab_hide_per_q = clamp(base.p_tab_hide_per_q + rng.uniform(0.05, 0.15), 0.0, 0.9)
        base.p_idle_per_q = clamp(base.p_idle_per_q + rng.uniform(0.05, 0.18), 0.0, 0.9)

    return base


def simulate_attempt_events(
    *,
    ctx: AttemptContext,
    rng: random.Random,
    question_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate an attempt's raw events and attempt-level summary."""

    events: List[Dict[str, Any]] = []
    seq = [0]
    client_meta = base_client_meta(rng)

    params = params_for(ctx.label, ctx.archetype, rng)

    now = ctx.start_at

    emit_event(
        events,
        ctx=ctx,
        event_type="attempt_state",
        occurred_at=now,
        client_seq_ref=seq,
        payload={"action": "start"},
        client_meta=client_meta,
        rng=rng,
    )

    # Track whether currently hidden/idle to emit transitions correctly.
    is_hidden = False
    is_idle = False

    # Keep a light per-question paste counter for the manifest.
    paste_by_q: Dict[str, int] = {qid: 0 for qid in question_ids}
    answer_change_by_q: Dict[str, int] = {qid: 0 for qid in question_ids}

    for qid in question_ids:
        emit_event(
            events,
            ctx=ctx,
            event_type="question_view",
            occurred_at=now,
            client_seq_ref=seq,
            payload={"question_id": qid, "action": "enter"},
            client_meta=client_meta,
            rng=rng,
        )

        # Sample question time (mixture: sometimes very fast).
        if rng.random() < params.p_fast_mix:
            q_time_s = lognorm_seconds(params.fast_median_q_s, params.sigma_q, rng)
        else:
            q_time_s = lognorm_seconds(params.median_q_s, params.sigma_q, rng)

        # Within-question behavior timeline is simulated as events spread across q_time_s.
        q_start = now
        q_end = q_start + timedelta(seconds=q_time_s)

        # Tab hidden episode (optional). Emit hidden->visible transitions.
        if rng.random() < params.p_tab_hide_per_q:
            # Hide starts sometime within the question.
            hide_start = q_start + timedelta(seconds=rng.uniform(0.2, max(0.3, q_time_s * 0.7)))
            hide_dur = lognorm_seconds(params.tab_hide_median_s, params.tab_hide_sigma, rng)
            hide_end = min(q_end - timedelta(seconds=0.2), hide_start + timedelta(seconds=hide_dur))
            if hide_end > hide_start + timedelta(seconds=0.5):
                if not is_hidden:
                    emit_event(
                        events,
                        ctx=ctx,
                        event_type="visibility_change",
                        occurred_at=hide_start,
                        client_seq_ref=seq,
                        payload={"state": "hidden"},
                        client_meta=client_meta,
                        rng=rng,
                    )
                    is_hidden = True

                # Often, hidden implies idle (no input) for a while.
                if not is_idle and rng.random() < 0.75:
                    emit_event(
                        events,
                        ctx=ctx,
                        event_type="idle_state",
                        occurred_at=hide_start + timedelta(milliseconds=120),
                        client_seq_ref=seq,
                        payload={"state": "idle", "idle_reason": "tab_hidden"},
                        client_meta=client_meta,
                        rng=rng,
                    )
                    is_idle = True

                emit_event(
                    events,
                    ctx=ctx,
                    event_type="visibility_change",
                    occurred_at=hide_end,
                    client_seq_ref=seq,
                    payload={"state": "visible"},
                    client_meta=client_meta,
                    rng=rng,
                )
                is_hidden = False

                if is_idle:
                    emit_event(
                        events,
                        ctx=ctx,
                        event_type="idle_state",
                        occurred_at=hide_end + timedelta(milliseconds=80),
                        client_seq_ref=seq,
                        payload={"state": "active", "idle_reason": "tab_hidden"},
                        client_meta=client_meta,
                        rng=rng,
                    )
                    is_idle = False

        # Idle episode (independent of tab hidden).
        if rng.random() < params.p_idle_per_q:
            idle_start = q_start + timedelta(seconds=rng.uniform(0.1, max(0.2, q_time_s * 0.8)))
            idle_dur = lognorm_seconds(params.idle_median_s, params.idle_sigma, rng)
            # spikes: occasionally much longer
            if rng.random() < params.p_idle_spike:
                idle_dur *= rng.uniform(2.0, 5.0)
            idle_end = min(q_end - timedelta(seconds=0.2), idle_start + timedelta(seconds=idle_dur))
            if idle_end > idle_start + timedelta(seconds=0.6):
                if not is_idle:
                    emit_event(
                        events,
                        ctx=ctx,
                        event_type="idle_state",
                        occurred_at=idle_start,
                        client_seq_ref=seq,
                        payload={"state": "idle", "idle_reason": "no_input"},
                        client_meta=client_meta,
                        rng=rng,
                    )
                    is_idle = True

                emit_event(
                    events,
                    ctx=ctx,
                    event_type="idle_state",
                    occurred_at=idle_end,
                    client_seq_ref=seq,
                    payload={"state": "active", "idle_reason": "no_input"},
                    client_meta=client_meta,
                    rng=rng,
                )
                is_idle = False

        # Clipboard events (copy/paste) - no content.
        if rng.random() < params.p_copy_per_q:
            t = q_start + timedelta(seconds=rng.uniform(0.2, max(0.3, q_time_s * 0.9)))
            emit_event(
                events,
                ctx=ctx,
                event_type="clipboard",
                occurred_at=t,
                client_seq_ref=seq,
                payload={"action": "copy", "context": "answer_field", "question_id": qid},
                client_meta=client_meta,
                rng=rng,
            )

        if rng.random() < params.p_paste_per_q:
            n_pastes = max(1, poisson(params.paste_lambda, rng))
            paste_by_q[qid] += n_pastes
            for _ in range(n_pastes):
                t = q_start + timedelta(seconds=rng.uniform(0.2, max(0.3, q_time_s * 0.95)))
                emit_event(
                    events,
                    ctx=ctx,
                    event_type="clipboard",
                    occurred_at=t,
                    client_seq_ref=seq,
                    payload={"action": "paste", "context": "answer_field", "question_id": qid},
                    client_meta=client_meta,
                    rng=rng,
                )

        # Typing bursts (timing only): 0..k bursts per question.
        n_bursts = poisson(params.typing_burst_lambda_per_q, rng)
        for _ in range(n_bursts):
            burst_dur = rng.uniform(0.8, 6.0)
            # typing speed lognormal-ish around mean
            kps = max(0.2, rng.lognormvariate(math.log(params.keystrokes_per_s_mean), params.keystrokes_per_s_sigma) - 0.5)
            n_keys = int(max(0, kps * burst_dur))
            if n_keys == 0:
                continue
            t = q_start + timedelta(seconds=rng.uniform(0.1, max(0.2, q_time_s * 0.95)))
            emit_event(
                events,
                ctx=ctx,
                event_type="typing_burst",
                occurred_at=t,
                client_seq_ref=seq,
                payload={
                    "question_id": qid,
                    "n_keystrokes": n_keys,
                    "duration_ms": int(burst_dur * 1000),
                },
                client_meta=client_meta,
                rng=rng,
            )

        # Answer changes: Poisson distributed edits.
        n_changes = poisson(params.answer_change_lambda_per_q, rng)
        if n_changes > 0:
            answer_change_by_q[qid] += n_changes
        for _ in range(n_changes):
            t = q_start + timedelta(seconds=rng.uniform(0.2, max(0.3, q_time_s * 0.98)))
            emit_event(
                events,
                ctx=ctx,
                event_type="answer_change",
                occurred_at=t,
                client_seq_ref=seq,
                payload={"question_id": qid, "action": rng.choice(["set", "edit", "edit", "clear"])},
                client_meta=client_meta,
                rng=rng,
            )

        # Leave question.
        emit_event(
            events,
            ctx=ctx,
            event_type="question_view",
            occurred_at=q_end,
            client_seq_ref=seq,
            payload={"question_id": qid, "action": "leave"},
            client_meta=client_meta,
            rng=rng,
        )

        now = q_end

        # Small transition time between questions.
        now += timedelta(seconds=rng.uniform(0.3, 2.0))

    # Ensure we end in visible/active.
    if is_hidden:
        emit_event(
            events,
            ctx=ctx,
            event_type="visibility_change",
            occurred_at=now,
            client_seq_ref=seq,
            payload={"state": "visible"},
            client_meta=client_meta,
            rng=rng,
        )
        is_hidden = False
    if is_idle:
        emit_event(
            events,
            ctx=ctx,
            event_type="idle_state",
            occurred_at=now,
            client_seq_ref=seq,
            payload={"state": "active", "idle_reason": "no_input"},
            client_meta=client_meta,
            rng=rng,
        )
        is_idle = False

    emit_event(
        events,
        ctx=ctx,
        event_type="attempt_state",
        occurred_at=now,
        client_seq_ref=seq,
        payload={"action": "submit"},
        client_meta=client_meta,
        rng=rng,
    )

    attempt_summary = {
        "attempt_id": ctx.attempt_id,
        "user_id": ctx.user_id,
        "assessment_id": ctx.assessment_id,
        "label": ctx.label,
        "archetype": ctx.archetype,
        "n_questions": ctx.n_questions,
        "start_at": utc_iso(ctx.start_at),
        "end_at": utc_iso(now),
        # Light diagnostics to help sanity-check the generator.
        "paste_total": int(sum(paste_by_q.values())),
        "paste_questions_affected": int(sum(1 for v in paste_by_q.values() if v > 0)),
        "answer_change_total": int(sum(answer_change_by_q.values())),
    }

    return events, attempt_summary


def generate_dataset(
    *,
    out_dir: str,
    n_attempts: int,
    seed: int,
    p_normal: float,
    p_borderline: float,
    question_count_mode: str,
) -> None:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    events_path = os.path.join(out_dir, "events.jsonl")
    attempts_path = os.path.join(out_dir, "attempts.jsonl")

    # Generate a small set of assessments to create some variability.
    assessment_ids = [f"asm_{i:03d}" for i in range(1, 6)]

    # Spread attempt start times over a day.
    base_day = datetime(2026, 4, 15, 9, 0, 0, tzinfo=timezone.utc)

    with open(events_path, "w", encoding="utf-8") as f_events, open(
        attempts_path, "w", encoding="utf-8"
    ) as f_attempts:
        for i in range(n_attempts):
            label = pick_label(rng, p_normal=p_normal, p_borderline=p_borderline)
            archetype = pick_archetype(label, rng)

            assessment_id = rng.choice(assessment_ids)
            user_id = f"user_{rng.randint(1, max(30, n_attempts // 3)):04d}"
            attempt_id = str(uuid.uuid4())

            n_questions = choose_question_count(rng, mode=question_count_mode)
            question_ids = [f"q_{j:03d}" for j in range(1, n_questions + 1)]

            start_at = base_day + timedelta(seconds=rng.uniform(0, 10 * 60 * 60))

            ctx = AttemptContext(
                attempt_id=attempt_id,
                user_id=user_id,
                assessment_id=assessment_id,
                label=label,
                archetype=archetype,
                n_questions=n_questions,
                start_at=start_at,
            )

            events, attempt_summary = simulate_attempt_events(ctx=ctx, rng=rng, question_ids=question_ids)

            # Write attempts manifest first (one row per attempt).
            f_attempts.write(json.dumps(attempt_summary) + "\n")

            # Append events; keep them grouped per attempt for easy debugging.
            for ev in events:
                f_events.write(json.dumps(ev) + "\n")

            # Add a small delimiter blank line? No: JSONL should be strict.


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic assessment event logs.")
    p.add_argument("--out_dir", type=str, default="data/synth", help="Output directory")
    p.add_argument("--n_attempts", type=int, default=600, help="Number of attempts")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--p_normal", type=float, default=0.70, help="Probability of normal attempts")
    p.add_argument("--p_borderline", type=float, default=0.20, help="Probability of borderline attempts")
    p.add_argument(
        "--question_count_mode",
        type=str,
        default="variable",
        choices=["variable", "fixed_20"],
        help="Question count mode",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.p_normal < 0 or args.p_borderline < 0 or (args.p_normal + args.p_borderline) > 1.0:
        raise SystemExit("Invalid label probabilities: require p_normal>=0, p_borderline>=0, p_normal+p_borderline<=1")

    generate_dataset(
        out_dir=args.out_dir,
        n_attempts=args.n_attempts,
        seed=args.seed,
        p_normal=args.p_normal,
        p_borderline=args.p_borderline,
        question_count_mode=args.question_count_mode,
    )


if __name__ == "__main__":
    main()
