#!/usr/bin/env bash
# Robust 3-node launcher for systemd: head master first, then workers, health monitor.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
HEAD_ENV="${HEAD_ENV:-${REPO_DIR}/head.env}"
WORKER_ENV="${WORKER_ENV:-${REPO_DIR}/worker.env}"
NODE3_ENV="${NODE3_ENV:-${REPO_DIR}/node3.env}"
REMOTE_USER="${REMOTE_USER:-linar}"
WORKER_HOST="${WORKER_HOST:-192.168.0.127}"
NODE3_HOST="${NODE3_HOST:-192.168.0.36}"
WORKER_QSFP_IP="${WORKER_QSFP_IP:-192.168.179.127}"
NODE3_QSFP_IP="${NODE3_QSFP_IP:-192.168.177.36}"
HEAD_QSFP_IP_NODE3="${HEAD_QSFP_IP_NODE3:-192.168.177.27}"
HEAD_QSFP_IP_WORKER="${HEAD_QSFP_IP_WORKER:-192.168.179.27}"
CLUSTER_NNODES="${CLUSTER_NNODES:-2}"
REMOTE_DIR="${REMOTE_DIR:-agent/deepseek-v4-flash-1m-context}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_dsf1m_prod}"
VLLM_IMAGE="${VLLM_IMAGE:-aidendle94/sparkrun-vllm-ds4-gb10:production-ready}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/v1/models}"
HEALTH_INTERVAL_SEC="${HEALTH_INTERVAL_SEC:-30}"
HEALTH_FAILURES="${HEALTH_FAILURES:-4}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-3600}"
WORKER_STAGGER_SEC="${WORKER_STAGGER_SEC:-45}"
MIN_CUDA_FREE_GB="${MIN_CUDA_FREE_GB:-100}"
MAX_LAUNCH_ATTEMPTS="${MAX_LAUNCH_ATTEMPTS:-3}"
ROCE_WAIT_SEC="${ROCE_WAIT_SEC:-300}"

export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ssh_host() {
  echo "${REMOTE_USER}@${1}"
}

run_on() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$(ssh_host "$host")" "$@"
}

cuda_free_gb_local() {
  docker run --rm --gpus all "$VLLM_IMAGE" python3 -c \
    'import torch; f,_=torch.cuda.mem_get_info(); print(f"{f/1e9:.1f}")' 2>/dev/null || echo 0
}

cuda_free_gb_remote() {
  local host="$1"
  run_on "$host" "docker run --rm --gpus all '$VLLM_IMAGE' python3 -c \
    'import torch; f,_=torch.cuda.mem_get_info(); print(f\"{f/1e9:.1f}\")'" 2>/dev/null || echo 0
}

drop_caches_local() {
  if sudo -n true 2>/dev/null; then
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true
  fi
}

drop_caches_remote() {
  local host="$1"
  run_on "$host" 'if sudo -n true 2>/dev/null; then sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"; fi' \
    >/dev/null 2>&1 || true
}

ensure_cuda_memory() {
  local head_free worker_free node3_free=""
  head_free="$(cuda_free_gb_local)"
  worker_free="$(cuda_free_gb_remote "$WORKER_HOST")"
  if (( CLUSTER_NNODES >= 3 )); then
    node3_free="$(cuda_free_gb_remote "$NODE3_HOST")"
  fi
  if (( CLUSTER_NNODES >= 3 )); then
    echo "CUDA free: head=${head_free}GB worker=${worker_free}GB node3=${node3_free}GB (need ${MIN_CUDA_FREE_GB}GB)"
  else
    echo "CUDA free: head=${head_free}GB worker=${worker_free}GB (need ${MIN_CUDA_FREE_GB}GB)"
  fi

  if awk -v free="$head_free" -v min="$MIN_CUDA_FREE_GB" 'BEGIN{exit !(free+0 < min+0)}'; then
    echo "Low CUDA on head; dropping cache..."
    drop_caches_local
    head_free="$(cuda_free_gb_local)"
  fi
  if awk -v free="$worker_free" -v min="$MIN_CUDA_FREE_GB" 'BEGIN{exit !(free+0 < min+0)}'; then
    echo "Low CUDA on worker; dropping cache..."
    drop_caches_remote "$WORKER_HOST"
    worker_free="$(cuda_free_gb_remote "$WORKER_HOST")"
  fi
  if (( CLUSTER_NNODES >= 3 )); then
    if awk -v free="$node3_free" -v min="$MIN_CUDA_FREE_GB" 'BEGIN{exit !(free+0 < min+0)}'; then
      echo "Low CUDA on node3; dropping cache..."
      drop_caches_remote "$NODE3_HOST"
      node3_free="$(cuda_free_gb_remote "$NODE3_HOST")"
    fi
  fi

  if (( CLUSTER_NNODES >= 3 )); then
    if awk -v h="$head_free" -v w="$worker_free" -v n="$node3_free" -v min="$MIN_CUDA_FREE_GB" \
      'BEGIN{exit !((h+0<min+0)||(w+0<min+0)||(n+0<min+0))}'; then
      echo "Insufficient CUDA memory (head=${head_free}GB worker=${worker_free}GB node3=${node3_free}GB)" >&2
      return 1
    fi
  elif awk -v h="$head_free" -v w="$worker_free" -v min="$MIN_CUDA_FREE_GB" \
    'BEGIN{exit !((h+0<min+0)||(w+0<min+0))}'; then
    echo "Insufficient CUDA memory (head=${head_free}GB worker=${worker_free}GB)" >&2
    return 1
  fi
}

ib_link_up_on() {
  local host="$1"
  if [[ "$host" == local ]]; then
    ibdev2netdev -v 2>/dev/null | grep -q '(ACTIVE)'
  else
    run_on "$host" 'ibdev2netdev -v 2>/dev/null | grep -q "(ACTIVE)"'
  fi
}

cluster_network_ready() {
  if (( CLUSTER_NNODES >= 3 )); then
    ib_link_up_on local \
      && ib_link_up_on "$WORKER_HOST" \
      && ib_link_up_on "$NODE3_HOST" \
      && ping -c 1 -W 2 "$WORKER_QSFP_IP" >/dev/null 2>&1 \
      && ping -c 1 -W 2 "$NODE3_QSFP_IP" >/dev/null 2>&1 \
      && run_on "$WORKER_HOST" "ping -c 1 -W 2 '$HEAD_QSFP_IP_WORKER' >/dev/null 2>&1" \
      && run_on "$NODE3_HOST" "ping -c 1 -W 2 '$HEAD_QSFP_IP_NODE3' >/dev/null 2>&1"
  else
    ib_link_up_on local \
      && ib_link_up_on "$WORKER_HOST" \
      && ping -c 1 -W 2 "$WORKER_QSFP_IP" >/dev/null 2>&1 \
      && run_on "$WORKER_HOST" "ping -c 1 -W 2 '$HEAD_QSFP_IP_WORKER' >/dev/null 2>&1"
  fi
}

wait_for_nodes() {
  local hosts=("$WORKER_HOST")
  if (( CLUSTER_NNODES >= 3 )); then
    echo "Waiting for worker and node3 SSH/Docker..."
    hosts+=("$NODE3_HOST")
  else
    echo "Waiting for worker SSH/Docker..."
  fi
  for host in "${hosts[@]}"; do
    for _ in $(seq 1 60); do
      if run_on "$host" "docker ps >/dev/null" >/dev/null 2>&1; then
        break
      fi
      sleep 5
    done
    run_on "$host" "docker ps >/dev/null" >/dev/null 2>&1 || {
      echo "Node ${host} not reachable" >&2
      return 1
    }
  done
}

wait_for_roce() {
  echo "Waiting for RoCE/QSFP connectivity (up to ${ROCE_WAIT_SEC}s)..."
  local deadline=$((SECONDS + ROCE_WAIT_SEC))
  until cluster_network_ready; do
    if (( SECONDS >= deadline )); then
      echo "Cluster network not ready within ${ROCE_WAIT_SEC}s" >&2
      return 1
    fi
    sleep 5
  done
  echo "RoCE links and QSFP ping OK."
  sleep 5
}

container_running_on() {
  local where="$1"
  if [[ "$where" == head ]]; then
    [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]
  elif [[ "$where" == worker ]]; then
    [[ "$(run_on "$WORKER_HOST" "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null || true" 2>/dev/null)" == "true" ]]
  else
    [[ "$(run_on "$NODE3_HOST" "docker inspect -f '{{.State.Running}}' '$CONTAINER_NAME' 2>/dev/null || true" 2>/dev/null)" == "true" ]]
  fi
}

containers_running() {
  if (( CLUSTER_NNODES >= 3 )); then
    container_running_on head && container_running_on worker && container_running_on node3
  else
    container_running_on head && container_running_on worker
  fi
}

cleanup_cluster() {
  echo "Stopping vLLM containers on all nodes..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker compose -f "${REPO_DIR}/docker-compose.yml" --env-file "$HEAD_ENV" down >/dev/null 2>&1 || true
  run_on "$WORKER_HOST" "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 || true; cd ~/${REMOTE_DIR} && docker compose -f docker-compose.yml --env-file worker.env down >/dev/null 2>&1 || true" \
    >/dev/null 2>&1 || true
  if (( CLUSTER_NNODES >= 3 )); then
    run_on "$NODE3_HOST" "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 || true; cd ~/${REMOTE_DIR} && docker compose -f docker-compose.yml --env-file node3.env down >/dev/null 2>&1 || true" \
      >/dev/null 2>&1 || true
  fi
}

health_ok() {
  containers_running || return 1
  curl -fsS --max-time 10 "$HEALTH_URL" 2>/dev/null | python3 -c \
    'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); ids={item.get("id") for item in data.get("data", []) if isinstance(item, dict)}; raise SystemExit(0 if expected in ids else 1)' \
    "$SERVED_MODEL_NAME" 2>/dev/null
}

dump_crash_logs() {
  echo "=== head logs ===" >&2
  docker logs --tail=60 "$CONTAINER_NAME" 2>&1 || true
  echo "=== worker logs ===" >&2
  run_on "$WORKER_HOST" "docker logs --tail=60 '$CONTAINER_NAME'" 2>&1 || true
  if (( CLUSTER_NNODES >= 3 )); then
    echo "=== node3 logs ===" >&2
    run_on "$NODE3_HOST" "docker logs --tail=60 '$CONTAINER_NAME'" 2>&1 || true
  fi
}

launch_cluster_once() {
  drop_caches_remote "$WORKER_HOST"
  if (( CLUSTER_NNODES >= 3 )); then
    drop_caches_remote "$NODE3_HOST"
  fi
  drop_caches_local

  echo "Starting head (master) on ${MASTER_ADDR:-192.168.0.27}:25000..."
  docker compose -f "${REPO_DIR}/docker-compose.yml" --env-file "$HEAD_ENV" up -d

  echo "Waiting for head master port 25000..."
  local port_deadline=$((SECONDS + 600))
  until ss -tln 2>/dev/null | grep -q ':25000'; do
    if (( SECONDS >= port_deadline )); then
      echo "Head master port 25000 did not open within 600s" >&2
      dump_crash_logs
      return 1
    fi
    if ! container_running_on head; then
      echo "Head container exited while waiting for master port" >&2
      dump_crash_logs
      return 1
    fi
    sleep 5
  done
  echo "Head master port 25000 is up."

  echo "Starting worker${CLUSTER_NNODES:+ and node3}..."
  run_on "$WORKER_HOST" "cd ~/${REMOTE_DIR} && docker compose -f docker-compose.yml --env-file worker.env up -d" &
  local wp=$!
  if (( CLUSTER_NNODES >= 3 )); then
    run_on "$NODE3_HOST" "cd ~/${REMOTE_DIR} && docker compose -f docker-compose.yml --env-file node3.env up -d" &
    local np=$!
    wait "$wp" "$np"
  else
    wait "$wp"
  fi

  echo "Waiting ${WORKER_STAGGER_SEC}s for workers to join..."
  sleep "$WORKER_STAGGER_SEC"

  local nodes=(worker head)
  if (( CLUSTER_NNODES >= 3 )); then
    nodes=(worker node3 head)
  fi
  for node in "${nodes[@]}"; do
    if ! container_running_on "$node"; then
      echo "Container not running on ${node}" >&2
      dump_crash_logs
      return 1
    fi
  done
}

wait_for_health() {
  echo "Waiting for vLLM health at ${HEALTH_URL}..."
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  until health_ok; do
    if (( SECONDS >= deadline )); then
      echo "vLLM did not become healthy within ${STARTUP_TIMEOUT_SEC}s" >&2
      dump_crash_logs
      return 1
    fi
    if ! containers_running; then
      echo "vLLM container crashed during startup" >&2
      dump_crash_logs
      return 1
    fi
    sleep 10
  done
  echo "vLLM is healthy (${CLUSTER_NNODES}-node TP=2)."
}

launch_with_retries() {
  local attempt
  for attempt in $(seq 1 "$MAX_LAUNCH_ATTEMPTS"); do
    echo "Launch attempt ${attempt}/${MAX_LAUNCH_ATTEMPTS}..."
    cleanup_cluster
    sleep 5
    ensure_cuda_memory || {
      echo "CUDA memory check failed on attempt ${attempt}" >&2
      if (( attempt >= MAX_LAUNCH_ATTEMPTS )); then
        return 1
      fi
      sleep 30
      continue
    }
    if launch_cluster_once && wait_for_health; then
      return 0
    fi
    echo "Launch attempt ${attempt} failed." >&2
    cleanup_cluster
    if (( attempt < MAX_LAUNCH_ATTEMPTS )); then
      echo "Retrying in 30s..."
      sleep 30
      wait_for_roce || true
    fi
  done
  return 1
}

trap cleanup_cluster EXIT INT TERM HUP

wait_for_nodes
wait_for_roce
cleanup_cluster
sleep 3
launch_with_retries

echo "Entering health monitor."
failures=0
while true; do
  sleep "$HEALTH_INTERVAL_SEC"
  if health_ok; then
    failures=0
  else
    failures=$((failures + 1))
    echo "vLLM health check failed (${failures}/${HEALTH_FAILURES})" >&2
    if ! containers_running; then
      dump_crash_logs
    fi
    if (( failures >= HEALTH_FAILURES )); then
      echo "vLLM is unhealthy; exiting for systemd restart." >&2
      exit 1
    fi
  fi
done
