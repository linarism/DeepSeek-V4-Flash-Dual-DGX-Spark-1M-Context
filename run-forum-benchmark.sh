#!/usr/bin/env bash
# Forum-style llama-benchy throughput replay for the MiaAI 1M stack.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
RUNS="${RUNS:-3}"
PREWARM="${PREWARM:-1}"
BENCH_VENV="${BENCH_VENV:-${HOME}/.cache/dsf1m-bench-venv}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/benchmark-results/${SERVED_MODEL_NAME}/forum-${RUN_TIMESTAMP}}"

ensure_benchy() {
  if [[ -x "${BENCH_VENV}/bin/llama-benchy" ]]; then
    return 0
  fi
  echo "Installing llama-benchy into ${BENCH_VENV}..."
  python3 -m venv "$BENCH_VENV"
  "${BENCH_VENV}/bin/pip" install -q llama-benchy
}

health_ok() {
  curl -fsS --max-time 10 "${BASE_URL%/}/models" 2>/dev/null | python3 -c \
    'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); ids={item.get("id") for item in data.get("data", []) if isinstance(item, dict)}; raise SystemExit(0 if expected in ids else 1)' \
    "$SERVED_MODEL_NAME" 2>/dev/null
}

health_ok || {
  echo "Stack is not healthy on ${BASE_URL}" >&2
  exit 1
}

ensure_benchy
mkdir -p "$OUT_DIR"

benchy() {
  local label="$1"
  shift
  local log="${OUT_DIR}/${label}.log"
  echo "==> ${label}"
  "${BENCH_VENV}/bin/llama-benchy" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL" \
    --runs "$RUNS" \
    --latency-mode none \
    "$@" 2>&1 | tee "$log"
}

if [[ "$PREWARM" == "1" ]]; then
  echo "Prewarming chat path via llama-benchy (pp1024/tg32 c1)..."
  "${BENCH_VENV}/bin/llama-benchy" \
    --base-url "$BASE_URL" --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL" --runs 1 --latency-mode none \
    --pp 1024 --tg 32 --depth 0 --concurrency 1 >/dev/null 2>&1 || true
fi

benchy pp1024-tg128-c1 --pp 1024 --tg 128 --depth 0 --concurrency 1
benchy pp2048-tg128-c2 --pp 2048 --tg 128 --depth 0 --concurrency 2

cat > "${OUT_DIR}/summary.md" <<EOF
# MiaAI DeepSeek V4 Flash 1M benchmark (${RUN_TIMESTAMP})

Tool: llama-benchy
Base URL: ${BASE_URL}
Runs per case: ${RUNS}

Forum reference (NVIDIA post 372268):
- pp1024/tg128 c1 decode: 45.7 tok/s
- pp2048/tg128 c2 decode: 54.4 tok/s

See individual logs in this directory.
EOF

echo "Forum benchmark complete: ${OUT_DIR}"
