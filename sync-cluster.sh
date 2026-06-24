#!/usr/bin/env bash
# Sync cluster config to worker and node3.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_USER="${REMOTE_USER:-linar}"
WORKER_HOST="${WORKER_HOST:-192.168.0.127}"
NODE3_HOST="${NODE3_HOST:-192.168.0.36}"
REMOTE_DIR="${REMOTE_DIR:-agent/deepseek-v4-flash-1m-context}"

FILES=(
  docker-compose.yml
  head.env
  worker.env
  node3.env
  .env
  start-deepseek-v4-flash.sh
  stop-deepseek-v4-flash.sh
  cluster-launch.sh
  run-forum-benchmark.sh
  run-clock-power-bench.sh
  install-gpu-clock-cap.sh
)

sync_one() {
  local host="$1"
  local target="${REMOTE_USER}@${host}"
  local remote="~/${REMOTE_DIR}"
  echo "Syncing to ${target}:${remote}..."
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "mkdir -p ${remote}"
  rsync -az "${FILES[@]/#/${SCRIPT_DIR}/}" "${target}:${remote}/"
}

sync_one "$WORKER_HOST"
sync_one "$NODE3_HOST"
echo "Sync complete."
