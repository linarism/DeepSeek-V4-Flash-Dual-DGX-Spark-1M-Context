#!/usr/bin/env bash
# =============================================================================
# start-deepseek-v4-flash.sh
#
# Starts the DeepSeek V4 Flash vLLM server across two DGX Spark nodes.
#
# Prerequisites:
#   - Docker & docker compose plugin installed on both nodes
#   - Passwordless SSH from head to worker node configured
#   - .env file configured (see .env.example)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in .env}"

API_URL="http://127.0.0.1:8000/v1/models"

cd "$SCRIPT_DIR"

echo "Starting worker on ${WORKER_HOST}..."
ssh "${WORKER_HOST}" "cd '${SCRIPT_DIR}' && docker compose up -d"

echo "Starting head on spark1..."
docker compose up -d

echo "Waiting for vLLM API..."
for _ in $(seq 1 80); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null; then
    echo "DeepSeek V4 Flash is running: $API_URL"
    docker compose ps
    ssh "${WORKER_HOST}" "cd '${SCRIPT_DIR}' && docker compose ps"
    exit 0
  fi
  sleep 15
done

echo "Timed out waiting for API. Recent spark1 logs:"
docker logs --tail=120 deepseek-v4-flash-vllm-1 2>&1 || true
echo "Recent spark2 logs:"
ssh "${WORKER_HOST}" "docker logs --tail=120 deepseek-v4-flash-vllm-1 2>&1" || true
exit 1
