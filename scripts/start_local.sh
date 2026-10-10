#!/usr/bin/env bash
# Starts Control Finder on this computer and opens its website at http://127.0.0.1:8000/
#
#   scripts/start_local.sh
#
# Ctrl+C stops the website and the API. The database keeps running until: docker compose stop
# Needs Docker (Docker Desktop on a Mac), Ollama with nomic-embed-text and qwen3:8b, Node 24, and the
# Python setup in the README (.venv). Everything listens on 127.0.0.1 only, so nothing outside this
# computer can reach it.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "The Python setup is missing. Follow \"Run it\" in the README first." >&2
  exit 1
fi

health=$(curl -sf http://127.0.0.1:8000/health 2> /dev/null || true)
if [[ "$health" == *'"controls"'* ]]; then
  echo "Control Finder is already running at http://127.0.0.1:8000/ (scripts/stop_local.sh stops it)."
  open http://127.0.0.1:8000/ 2> /dev/null || true
  exit 0
fi
if lsof -nP -iTCP:8000 -sTCP:LISTEN > /dev/null 2>&1; then
  echo "Another program is using port 8000. Close it, then run this again." >&2
  exit 1
fi

echo "1/4 The database: PostgreSQL with pgvector, on 127.0.0.1:55432"
if ! docker info > /dev/null 2>&1; then
  echo "Docker isn't running. Open Docker Desktop and wait until it's running, then run this again." >&2
  exit 1
fi
docker compose up -d --wait

echo "2/4 The local models: Ollama"
if ! curl -sf http://127.0.0.1:11434/api/version > /dev/null; then
  echo "Ollama isn't running. Open the Ollama app (or run: ollama serve), then run this again." >&2
  exit 1
fi
tags=$(curl -sf http://127.0.0.1:11434/api/tags)
for model in nomic-embed-text qwen3:8b; do
  if ! grep -Eq "\"name\":\"$model(:latest)?\"" <<< "$tags"; then  # exactly: qwen3:8b-q8_0 isn't qwen3:8b
    echo "Ollama doesn't have $model yet. Run: ollama pull $model" >&2
    exit 1
  fi
done

echo "3/4 The website"
if [ ! -f web/dist/index.html ] || [ -n "$(find web/src web/index.html web/package-lock.json web/vite.config.ts -newer web/dist/index.html -print -quit)" ]; then
  if ! command -v npm > /dev/null; then
    echo "The website needs building, and that needs Node 24 (npm). Install it from nodejs.org, then run this again." >&2
    exit 1
  fi
  (cd web && npm ci --no-audit --no-fund && npm run build)
else
  echo "    already built"
fi

echo "4/4 The API, serving the website at http://127.0.0.1:8000/ (Ctrl+C to stop)"
(
  for _ in $(seq 1 60); do
    kill -0 $$ 2> /dev/null || break  # after exec, $$ is the API: if it stopped, don't open anything
    if curl -sf http://127.0.0.1:8000/health > /dev/null; then
      open http://127.0.0.1:8000/ 2> /dev/null || true
      break
    fi
    sleep 1
  done
) &
exec .venv/bin/python -m uvicorn controlfinder.api:app --app-dir src --host 127.0.0.1 --port 8000
