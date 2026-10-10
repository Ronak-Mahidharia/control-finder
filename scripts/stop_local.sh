#!/usr/bin/env bash
# Stops what scripts/start_local.sh started: the API and website on port 8000, then the database.
# Ollama is left running; it's a separate app. Only the API on port 8000 is stopped, so a check
# running its own API on another port (scripts/run_api_check.py) isn't touched.
set -euo pipefail
cd "$(dirname "$0")/.."
stopped=no
for pid in $(lsof -tiTCP:8000 -sTCP:LISTEN 2> /dev/null || true); do
  if ps -o command= -p "$pid" | grep -q "uvicorn controlfinder.api:app"; then
    kill "$pid" && stopped=yes
  fi
done
if [ "$stopped" = yes ]; then
  echo "Stopped the API and the website."
else
  echo "The API wasn't running."
fi
if docker info > /dev/null 2>&1; then
  docker compose stop
else
  echo "Docker isn't running, so the database is stopped already."
fi
