#!/usr/bin/env bash
# Sync cluster config to the worker node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_USER="${REMOTE_USER:-linar}"
WORKER_HOST="${WORKER_HOST:-192.168.0.127}"
WORKER_DIR="${WORKER_DIR:-agent/deepseek-v4-flash-1m-context}"

target="${REMOTE_USER}@${WORKER_HOST}"
remote="~/${WORKER_DIR}"

echo "Syncing to ${target}:${remote}..."
ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "mkdir -p ${remote}"
rsync -az \
  "${SCRIPT_DIR}/docker-compose.yml" \
  "${SCRIPT_DIR}/worker.env" \
  "${SCRIPT_DIR}/head.env" \
  "${SCRIPT_DIR}/.env" \
  "${SCRIPT_DIR}/start-deepseek-v4-flash.sh" \
  "${SCRIPT_DIR}/stop-deepseek-v4-flash.sh" \
  "${SCRIPT_DIR}/cluster-launch.sh" \
  "${SCRIPT_DIR}/run-forum-benchmark.sh" \
  "${target}:${remote}/"

echo "Sync complete."
