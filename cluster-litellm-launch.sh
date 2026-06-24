#!/usr/bin/env bash
# LiteLLM proxy launcher for systemd: wait for vLLM, then health monitor loop.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Environment is injected by systemd via EnvironmentFile=/etc/default/deepseek-v4-flash-1m-litellm

REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
LITELLM_CONTAINER_NAME="${LITELLM_CONTAINER_NAME:-deepseek-v4-flash-1m-litellm}"
LITELLM_IMAGE="${LITELLM_IMAGE:-ghcr.io/berriai/litellm:main-latest}"
LITELLM_PORT="${LITELLM_PORT:-4000}"
LITELLM_CONFIG="${LITELLM_CONFIG:-/etc/litellm/deepseek-v4-flash-1m-litellm/config.yaml}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-local-vllm}"
LITELLM_MODEL_NAME="${LITELLM_MODEL_NAME:-deepseek-v4-flash}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:8000/v1/models}"
LITELLM_HEALTH_URL="${LITELLM_HEALTH_URL:-http://127.0.0.1:${LITELLM_PORT}/v1/models}"
LITELLM_LIVENESS_URL="${LITELLM_LIVENESS_URL:-http://127.0.0.1:${LITELLM_PORT}/health/liveliness}"
VLLM_STARTUP_TIMEOUT_SEC="${VLLM_STARTUP_TIMEOUT_SEC:-3600}"
HEALTH_INTERVAL_SEC="${HEALTH_INTERVAL_SEC:-30}"
HEALTH_FAILURES="${HEALTH_FAILURES:-4}"

export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

vllm_health_ok() {
  curl -fsS --max-time 10 "$VLLM_HEALTH_URL" 2>/dev/null | python3 -c \
    'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); ids={item.get("id") for item in data.get("data", []) if isinstance(item, dict)}; raise SystemExit(0 if expected in ids else 1)' \
    "$LITELLM_MODEL_NAME" 2>/dev/null
}

litellm_health_ok() {
  if [[ "$(docker inspect -f '{{.State.Running}}' "$LITELLM_CONTAINER_NAME" 2>/dev/null || true)" != "true" ]]; then
    return 1
  fi
  curl -fsS --max-time 5 "$LITELLM_LIVENESS_URL" >/dev/null 2>&1 || return 1
  curl -fsS --max-time 10 -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" "$LITELLM_HEALTH_URL" 2>/dev/null | python3 -c \
    'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); ids={item.get("id") for item in data.get("data", []) if isinstance(item, dict)}; raise SystemExit(0 if expected in ids else 1)' \
    "$LITELLM_MODEL_NAME" 2>/dev/null
}

stop_litellm() {
  docker rm -f "$LITELLM_CONTAINER_NAME" >/dev/null 2>&1 || true
}

start_litellm() {
  stop_litellm
  docker run -d \
    --name "$LITELLM_CONTAINER_NAME" \
    --network host \
    -v "${LITELLM_CONFIG}:/app/config.yaml:ro" \
    "$LITELLM_IMAGE" \
    --config /app/config.yaml \
    --host 0.0.0.0 \
    --port "$LITELLM_PORT"
}

trap stop_litellm EXIT INT TERM HUP

echo "Waiting for vLLM at ${VLLM_HEALTH_URL}..."
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SEC))
until vllm_health_ok; do
  if (( SECONDS >= deadline )); then
    echo "vLLM did not become healthy within ${VLLM_STARTUP_TIMEOUT_SEC}s" >&2
    exit 1
  fi
  sleep 10
done

echo "vLLM healthy; starting LiteLLM on port ${LITELLM_PORT}..."
start_litellm

echo "Waiting for LiteLLM health..."
deadline=$((SECONDS + 300))
until litellm_health_ok; do
  if (( SECONDS >= deadline )); then
    echo "LiteLLM did not become healthy within 300s" >&2
    docker logs --tail=80 "$LITELLM_CONTAINER_NAME" 2>&1 || true
    exit 1
  fi
  sleep 5
done

echo "LiteLLM is healthy; entering health monitor."
failures=0
while true; do
  sleep "$HEALTH_INTERVAL_SEC"
  if vllm_health_ok && litellm_health_ok; then
    failures=0
  else
    failures=$((failures + 1))
    echo "LiteLLM/vLLM health check failed (${failures}/${HEALTH_FAILURES})" >&2
    if (( failures >= HEALTH_FAILURES )); then
      echo "LiteLLM stack unhealthy; exiting for systemd restart." >&2
      exit 1
    fi
  fi
done
