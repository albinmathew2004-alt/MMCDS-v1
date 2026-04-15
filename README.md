# MMCDS v1 — Explainable Cheating Detection (Behavioral)

A minimal, production-minded, explainable cheating detection pipeline for online assessments.

## What it does
- Ingests raw behavioral events (no webcam/audio/mouse tracking/keystroke content)
- Computes deterministic attempt-level features
- Normalizes features into 0–1 signals
- Combines signals with weighted reasoning + explicit multi-signal gating
- Outputs risk (LOW/MEDIUM/HIGH), confidence, and human-readable explanations

## Quickstart
### 1) Generate synthetic data
```bash
python scripts/generate_synthetic.py --out_dir data/synth --n_attempts 600 --seed 42
```

### 2) Score attempts
```bash
python scripts/score_synth.py --events data/synth/events.jsonl --attempts data/synth/attempts.jsonl --out data/synth/scored.jsonl
```

### 3) Evaluate scoring vs synthetic labels
```bash
python scripts/evaluate_scored.py --scored data/synth/scored.jsonl
```

### Optional: add robust anomaly signal
```bash
python scripts/score_synth.py --events data/synth/events.jsonl --attempts data/synth/attempts.jsonl --out data/synth/scored_anomaly.jsonl --use_anomaly
```

## Minimal API skeleton
- FastAPI app: `app/main.py`
- Install deps: `pip install -r requirements.txt`
- Run: `uvicorn app.main:app --reload`

## Repo layout
- `mmcds/`: core scoring logic (features, signals, reasoning, risk, explanations)
- `scripts/`: generator, scoring, evaluation
- `app/`: minimal API skeleton
