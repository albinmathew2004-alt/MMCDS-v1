#!/usr/bin/env sh
set -eu

# AWS-style platforms often provide PORT; default to 8000 for local Docker.
PORT="${PORT:-8000}"

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
