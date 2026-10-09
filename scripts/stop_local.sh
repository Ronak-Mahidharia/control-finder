#!/usr/bin/env bash
# Stops what scripts/start_local.sh started: the API and website on port 8000, then the database.
# Ollama is left running; it's a separate app.
set -euo pipefail
cd "$(dirname "$0")/.."
if pkill -f "uvicorn controlfinder.api:app"; then
  echo "Stopped the API and the website."
else
  echo "The API wasn't running."
fi
docker compose stop
