from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components


API_URL = "http://127.0.0.1:8000/v1/score"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _copy_button(label: str, text: str, *, key: str) -> None:
        escaped = json.dumps(text)
        html = f"""
        <div style=\"display:flex; gap:8px; align-items:center;\">
            <button
                id=\"btn-{key}\"
                style=\"padding:6px 10px; border-radius:8px; border:1px solid rgba(0,0,0,0.14); background:white; cursor:pointer;\"
            >
                {label}
            </button>
            <span id=\"msg-{key}\" style=\"font-size:12px; opacity:0.75;\"></span>
        </div>
        <script>
            const btn = document.getElementById('btn-{key}');
            const msg = document.getElementById('msg-{key}');
            const txt = {escaped};
            btn.addEventListener('click', async () => {{
                msg.textContent = '';
                try {{
                    await navigator.clipboard.writeText(txt);
                    msg.textContent = 'Copied.';
                    setTimeout(() => msg.textContent = '', 1200);
                }} catch (e) {{
                    msg.textContent = 'Copy failed (browser blocked).';
                }}
            }});
        </script>
        """
        components.html(html, height=44)


def _safe_parse_json(text: str) -> Tuple[bool, Any, str]:
    try:
        return True, json.loads(text), ""
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e.msg} (line {e.lineno}, column {e.colno})"


def _risk_style(risk: str) -> Tuple[str, str]:
    r = (risk or "").upper().strip()
    if r == "LOW":
        return "LOW", "#15803d"  # green
    if r == "MEDIUM":
        return "MEDIUM", "#a16207"  # yellow-ish
    if r == "HIGH":
        return "HIGH", "#b91c1c"  # red
    return r or "UNKNOWN", "#334155"  # slate


def _flatten_reasons(explanation: Dict[str, List[str]] | None) -> List[str]:
    if not explanation:
        return []
    reasons: List[str] = []
    for _, items in explanation.items():
        if not items:
            continue
        for s in items:
            if s and isinstance(s, str):
                reasons.append(s)
    return reasons


@dataclass(frozen=True)
class CaseParams:
    name: str
    n_questions: int
    base_time_s: float
    fast_fraction: float
    tab_hidden_segments: int
    tab_hidden_mean_s: float
    paste_events: int
    idle_spikes: int
    answer_changes: int
    last_min_changes: int


def _build_case(params: CaseParams, *, seed: int | None = None) -> Dict[str, Any]:
    rng = random.Random(seed)

    attempt_id = f"demo-{params.name.lower()}-{uuid.uuid4().hex[:8]}"
    user_id = "demo_user"
    assessment_id = "demo_assessment"

    t0 = datetime.now(timezone.utc) - timedelta(minutes=25)
    t = t0
    client_seq = 1

    events: List[Dict[str, Any]] = []

    def add_event(event_type: str, occurred_at: datetime, payload: Dict[str, Any] | None = None):
        nonlocal client_seq
        ev: Dict[str, Any] = {
            "attempt_id": attempt_id,
            "user_id": user_id,
            "assessment_id": assessment_id,
            "client_seq": client_seq,
            "event_type": event_type,
            "occurred_at": _iso(occurred_at),
            "payload": payload or {},
        }
        events.append(ev)
        client_seq += 1

    # Attempt start
    add_event("attempt_state", t, {"action": "start"})

    # Decide which questions are "fast"
    qids = [f"q{i+1}" for i in range(params.n_questions)]
    fast_n = int(round(params.fast_fraction * params.n_questions))
    fast_set = set(rng.sample(qids, k=min(max(fast_n, 0), len(qids))))

    # Plan tab-hidden segments at random offsets across the attempt
    hidden_q_indices = sorted(rng.sample(range(params.n_questions), k=min(params.tab_hidden_segments, params.n_questions)))
    hidden_idx_set = set(hidden_q_indices)

    # Paste distribution across questions
    paste_qs = rng.choices(qids, k=max(0, params.paste_events)) if params.paste_events else []

    # Answer change distribution across questions
    change_qs = rng.choices(qids, k=max(0, params.answer_changes)) if params.answer_changes else []

    for i, qid in enumerate(qids):
        # Enter question
        add_event("question_view", t, {"action": "enter", "question_id": qid})

        # Within-question behavior
        if qid in paste_qs:
            # A couple of pastes can happen on the same question; spread them.
            add_event("clipboard", t + timedelta(seconds=1), {"action": "paste", "question_id": qid})

        # Some typing bursts (helps look realistic, but isn't required)
        if rng.random() < 0.6:
            dur_ms = int(rng.uniform(600, 2200))
            n_keys = int(rng.uniform(25, 140))
            add_event(
                "typing_burst",
                t + timedelta(seconds=2),
                {"duration_ms": dur_ms, "n_keystrokes": n_keys, "question_id": qid},
            )

        if qid in change_qs and rng.random() < 0.8:
            add_event("answer_change", t + timedelta(seconds=3), {"question_id": qid, "to": "choice_b"})

        # Leave question after some time
        if qid in fast_set:
            dt_s = rng.uniform(3.5, 8.5)
        else:
            dt_s = max(6.0, rng.gauss(params.base_time_s, params.base_time_s * 0.18))
        t_leave = t + timedelta(seconds=float(dt_s))
        add_event("question_view", t_leave, {"action": "leave", "question_id": qid})
        t = t_leave

        # Optional tab-hidden segment after this question
        if i in hidden_idx_set:
            hidden_s = max(2.0, rng.gauss(params.tab_hidden_mean_s, params.tab_hidden_mean_s * 0.25))
            add_event("visibility_change", t + timedelta(seconds=0.3), {"state": "hidden"})
            add_event("visibility_change", t + timedelta(seconds=0.3 + float(hidden_s)), {"state": "visible"})
            t = t + timedelta(seconds=0.3 + float(hidden_s))

        # Optional idle spikes
        if params.idle_spikes and rng.random() < (params.idle_spikes / max(1, params.n_questions)):
            idle_s = rng.uniform(35.0, 80.0)
            add_event("idle_state", t + timedelta(seconds=0.2), {"state": "idle"})
            add_event("idle_state", t + timedelta(seconds=0.2 + idle_s), {"state": "active"})
            t = t + timedelta(seconds=0.2 + idle_s)

        # Small gap between questions
        t = t + timedelta(seconds=rng.uniform(0.8, 2.2))

    # Submit
    submit_ts = t
    add_event("attempt_state", submit_ts, {"action": "submit"})

    # Inject extra last-minute changes if requested (place just before submit)
    if params.last_min_changes > 0:
        for j in range(params.last_min_changes):
            qid = rng.choice(qids)
            when = submit_ts - timedelta(seconds=rng.uniform(5.0, 55.0))
            add_event("answer_change", when, {"question_id": qid, "to": f"choice_{j%4}"})

        # Keep events in time order (client_seq order is still monotonic)
        events.sort(key=lambda e: e["occurred_at"])

    return {"attempt_id": attempt_id, "events": events}


def _sample_cases() -> Dict[str, Dict[str, Any]]:
    normal = CaseParams(
        name="Normal",
        n_questions=24,
        base_time_s=38.0,
        fast_fraction=0.05,
        tab_hidden_segments=1,
        tab_hidden_mean_s=6.0,
        paste_events=0,
        idle_spikes=0,
        answer_changes=5,
        last_min_changes=0,
    )

    borderline = CaseParams(
        name="Borderline",
        n_questions=22,
        base_time_s=26.0,
        fast_fraction=0.22,
        tab_hidden_segments=3,
        tab_hidden_mean_s=14.0,
        paste_events=3,
        idle_spikes=1,
        answer_changes=10,
        last_min_changes=2,
    )

    suspicious = CaseParams(
        name="Suspicious",
        n_questions=28,
        base_time_s=18.0,
        fast_fraction=0.55,
        tab_hidden_segments=7,
        tab_hidden_mean_s=28.0,
        paste_events=12,
        idle_spikes=3,
        answer_changes=18,
        last_min_changes=8,
    )

    return {
        "Normal": _build_case(normal, seed=7),
        "Borderline": _build_case(borderline, seed=11),
        "Suspicious": _build_case(suspicious, seed=23),
    }


def _random_case() -> Dict[str, Any]:
    # Simple generator: randomly sample plausible ranges.
    level = random.choices(["Normal", "Borderline", "Suspicious"], weights=[0.45, 0.35, 0.20], k=1)[0]
    if level == "Normal":
        params = CaseParams(
            name="Random",
            n_questions=random.randint(18, 30),
            base_time_s=random.uniform(28.0, 55.0),
            fast_fraction=random.uniform(0.0, 0.08),
            tab_hidden_segments=random.randint(0, 2),
            tab_hidden_mean_s=random.uniform(4.0, 10.0),
            paste_events=random.randint(0, 1),
            idle_spikes=0,
            answer_changes=random.randint(2, 8),
            last_min_changes=random.randint(0, 1),
        )
    elif level == "Borderline":
        params = CaseParams(
            name="Random",
            n_questions=random.randint(18, 28),
            base_time_s=random.uniform(18.0, 32.0),
            fast_fraction=random.uniform(0.12, 0.30),
            tab_hidden_segments=random.randint(2, 5),
            tab_hidden_mean_s=random.uniform(10.0, 18.0),
            paste_events=random.randint(2, 5),
            idle_spikes=random.randint(0, 2),
            answer_changes=random.randint(7, 14),
            last_min_changes=random.randint(1, 4),
        )
    else:
        params = CaseParams(
            name="Random",
            n_questions=random.randint(22, 34),
            base_time_s=random.uniform(12.0, 22.0),
            fast_fraction=random.uniform(0.40, 0.70),
            tab_hidden_segments=random.randint(5, 10),
            tab_hidden_mean_s=random.uniform(18.0, 35.0),
            paste_events=random.randint(8, 18),
            idle_spikes=random.randint(1, 4),
            answer_changes=random.randint(12, 26),
            last_min_changes=random.randint(5, 10),
        )
    payload = _build_case(params)
    payload["_generated_level_hint"] = level
    return payload


def main() -> None:
    st.set_page_config(page_title="MMCDS v1 Demo", layout="wide")

    st.markdown("# MMCDS v1 — Cheating Detection Demo")
    st.caption("Paste an event batch, load a sample case, or generate a synthetic one. Then score it via the local API.")

    if "input_json" not in st.session_state:
        st.session_state.input_json = _pretty_json(_sample_cases()["Normal"])

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown("## Input")
        with st.container(border=True):
            samples = _sample_cases()

            b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
            with b1:
                if st.button("Load Normal", use_container_width=True):
                    st.session_state.input_json = _pretty_json(samples["Normal"])
            with b2:
                if st.button("Load Borderline", use_container_width=True):
                    st.session_state.input_json = _pretty_json(samples["Borderline"])
            with b3:
                if st.button("Load Suspicious", use_container_width=True):
                    st.session_state.input_json = _pretty_json(samples["Suspicious"])
            with b4:
                if st.button("Generate Synthetic Test Case", use_container_width=True):
                    st.session_state.input_json = _pretty_json(_random_case())

            st.text_area(
                "Event batch JSON (must include `attempt_id` and `events`) ",
                key="input_json",
                height=420,
            )

            score_clicked = st.button("Score", type="primary", use_container_width=True)

    with right:
        st.markdown("## Output")
        with st.container(border=True):
            if not score_clicked:
                st.info("Click **Score** to see results.")
                return

            ok, payload, err = _safe_parse_json(st.session_state.input_json)
            if not ok:
                st.error(err)
                return

            if not isinstance(payload, dict):
                st.error("Input JSON must be an object with keys `attempt_id` and `events`.")
                return

            attempt_id = payload.get("attempt_id")
            events = payload.get("events")
            if not attempt_id or not isinstance(events, list):
                st.error("Missing required fields: `attempt_id` (string) and `events` (list).")
                return

            # Call API
            try:
                t_start = time.perf_counter()
                resp = requests.post(API_URL, json=payload, timeout=15)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            except requests.RequestException:
                st.error(
                    "Could not reach the API. Make sure it is running locally: "
                    "`uvicorn app.main:app --reload` (default at http://127.0.0.1:8000)."
                )
                return

            if resp.status_code >= 400:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    detail = resp.text.strip()
                st.error(f"API error ({resp.status_code}). {detail}")
                return

            try:
                out = resp.json()
            except ValueError:
                st.error("API returned a non-JSON response.")
                return

            risk = str(out.get("risk", "UNKNOWN"))
            confidence = float(out.get("confidence", 0.0) or 0.0)
            combined_score = float(out.get("combined_score", 0.0) or 0.0)
            base_score = float(out.get("base_score", 0.0) or 0.0)
            explanation = out.get("explanation")
            signals = out.get("signals")

            label, color = _risk_style(risk)
            st.markdown(
                f"""
                <div style="padding: 14px 14px; border-radius: 10px; border: 1px solid rgba(0,0,0,0.08);">
                  <div style="font-size: 13px; opacity: 0.75;">Risk Level</div>
                  <div style="font-size: 44px; font-weight: 800; line-height: 1.0; color: {color};">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
            m1.metric("Confidence", f"{confidence:.2f}")
            m2.metric("Combined", f"{combined_score:.2f}")
            m3.metric("Base", f"{base_score:.2f}")
            m4.metric("Attempt ID", str(out.get("attempt_id", attempt_id)))

            st.caption(f"API: {resp.status_code} • {elapsed_ms:.0f} ms")

            reasons = _flatten_reasons(explanation if isinstance(explanation, dict) else None)
            st.markdown("### Reasons")
            if reasons:
                st.markdown("\n".join([f"- {r}" for r in reasons]))
            else:
                st.caption("No reasons returned.")

            st.write("")
            with st.expander("Details", expanded=False):
                if isinstance(signals, dict) and signals:
                    st.markdown("#### Signals")
                    st.json(signals)
                st.markdown("#### Raw API response")
                st.json(out)

                out_text = _pretty_json(out)
                st.download_button(
                    "Download API response JSON",
                    data=out_text,
                    file_name="mmcds_score_response.json",
                    mime="application/json",
                    use_container_width=True,
                )

                st.markdown("#### Utilities")
                u1, u2 = st.columns(2)
                with u1:
                    _copy_button("Copy response JSON", out_text, key="copy-out")
                with u2:
                    _copy_button("Copy input JSON", st.session_state.input_json, key="copy-in")


if __name__ == "__main__":
    main()
