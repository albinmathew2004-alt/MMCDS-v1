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

## Deploy to AWS (fastest path: App Runner)

This repo includes a `Dockerfile` and `start.sh` so you can deploy the API quickly.

### Local Docker sanity check (optional)
```bash
docker build -t mmcds .
docker run --rm -p 8000:8000 mmcds
```
Then open:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

### AWS App Runner (container from source)
1) Push this repo to GitHub.
2) In AWS Console: **App Runner** → **Create service**.
3) Source: **Source code repository** (connect your GitHub) → pick the repo/branch.
4) Deployment: **Use a Dockerfile** (App Runner will build the image).
5) Configure service:
	- Port: `8000` (the container listens on `PORT` env var, defaulting to 8000)
6) Create service and wait for build/deploy.
7) Use the service URL:
	- `GET /health`
	- `POST /v1/score`

## Repo layout
- `mmcds/`: core scoring logic (features, signals, reasoning, risk, explanations)
- `scripts/`: generator, scoring, evaluation
- `app/`: minimal API skeleton
