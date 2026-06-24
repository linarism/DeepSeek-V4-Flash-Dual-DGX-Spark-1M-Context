#!/usr/bin/env bash
# Compare llama-benchy throughput + GPU power at stock vs 2000 MHz clocks (forum 372662).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER_HOST="${WORKER_HOST:-192.168.0.127}"
REMOTE_USER="${REMOTE_USER:-linar}"
STAMP="$(date +%Y%m%d%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/benchmark-results/deepseek-v4-flash/clock-power-${STAMP}}"
SAMPLE_SEC="${SAMPLE_SEC:-1}"
RUNS="${RUNS:-3}"

ssh_worker() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_USER}@${WORKER_HOST}" "$@"
}

set_clocks() {
  local label="$1"
  shift
  local cmd="$*"
  echo "=== Setting clocks (${label}) on head + worker: ${cmd}"
  sudo bash -c "$cmd"
  ssh_worker "sudo bash -c '$cmd'"
  sleep 3
  echo "Head clocks:"; nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv,noheader
  echo "Worker clocks:"; ssh_worker "nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv,noheader"
}

start_power_log() {
  local tag="$1"
  local head_log="${OUT_ROOT}/${tag}-power-head.csv"
  local worker_log="${OUT_ROOT}/${tag}-power-worker.csv"
  mkdir -p "$OUT_ROOT"
  (
    while true; do
      echo "$(date -Iseconds),$(nvidia-smi --query-gpu=power.draw,clocks.current.graphics,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ' | sed 's/ $//')"
      sleep "$SAMPLE_SEC"
    done
  ) >"$head_log" &
  local hp=$!
  ssh_worker "while true; do echo \$(date -Iseconds),\$(nvidia-smi --query-gpu=power.draw,clocks.current.graphics,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ' | sed 's/ \$//'); sleep ${SAMPLE_SEC}; done" >"$worker_log" &
  local wp=$!
  echo "$hp $wp"
}

stop_power_log() {
  local pids="$1"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  wait $pids 2>/dev/null || true
}

summarize_power() {
  local tag="$1"
  python3 - "$OUT_ROOT" "$tag" <<'PY'
import sys, pathlib, statistics
root, tag = sys.argv[1], sys.argv[2]
for node in ("head", "worker"):
    path = pathlib.Path(root) / f"{tag}-power-{node}.csv"
    if not path.exists():
        print(f"{node}: no data")
        continue
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("20") is False:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                rows.append({
                    "power_w": float(parts[1]),
                    "clock_mhz": float(parts[2]),
                    "temp_c": float(parts[3]),
                    "util_pct": float(parts[4]),
                })
            except ValueError:
                pass
    if not rows:
        print(f"{node}: empty")
        continue
    p = [r["power_w"] for r in rows]
    t = [r["temp_c"] for r in rows]
    c = [r["clock_mhz"] for r in rows]
    u = [r["util_pct"] for r in rows]
    print(f"{node}: power avg={statistics.mean(p):.1f}W max={max(p):.1f}W | temp avg={statistics.mean(t):.1f}C max={max(t):.1f}C | clock avg={statistics.mean(c):.0f}MHz | util avg={statistics.mean(u):.1f}% | samples={len(rows)}")
PY
}

run_bench_pass() {
  local tag="$1"
  local clock_cmd="$2"
  set_clocks "$tag" "$clock_cmd"
  local pids
  pids="$(start_power_log "$tag")"
  local bench_out="${OUT_ROOT}/${tag}-bench"
  mkdir -p "$bench_out"
  echo "=== Running llama-benchy (${tag}) ==="
  RUN_TIMESTAMP="${tag}" OUT_DIR="$bench_out" RUNS="$RUNS" PREWARM=1 \
    "${SCRIPT_DIR}/run-forum-benchmark.sh" 2>&1 | tee "${OUT_ROOT}/${tag}-bench.log"
  stop_power_log "$pids"
  summarize_power "$tag"
}

mkdir -p "$OUT_ROOT"
echo "Output: ${OUT_ROOT}"

# Stock clocks (forum: -lgc 0,3000)
run_bench_pass stock "nvidia-smi -pm 1 && nvidia-smi -lgc 0,3000"

# 2000 MHz cap (forum: -lgc 0,2000)
run_bench_pass mhz2000 "nvidia-smi -pm 1 && nvidia-smi -lgc 0,2000"

# Restore stock after test
set_clocks restore "nvidia-smi -pm 1 && nvidia-smi -rgc"

python3 - "$OUT_ROOT" <<'PY'
import re, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for tag in ("stock", "mhz2000"):
    log = root / f"{tag}-bench.log"
    if not log.exists():
        continue
    text = log.read_text()
    for m in re.finditer(r"\|\s*deepseek-ai/DeepSeek-V4-Flash\s*\|\s*(pp2048 \(c2\)|tg128 \(c2\)|pp1024 \(c1\)|tg128 \(c1\))\s*\|\s*([0-9.]+)", text):
        rows.append((tag, m.group(1), float(m.group(2))))
print("\n=== Throughput summary (tok/s total) ===")
for tag in ("stock", "mhz2000"):
    print(f"\n{tag}:")
    for t, case, val in rows:
        if t == tag:
            print(f"  {case}: {val:.2f}")
if len({r[0] for r in rows}) == 2:
    stock = {c: v for t, c, v in rows if t == "stock" and "tg128" in c}
    low = {c: v for t, c, v in rows if t == "mhz2000" and "tg128" in c}
    for case in stock:
        key = case.replace("pp", "tg") if "pp" in case else case
        for lk, lv in low.items():
            if "tg128" in lk and case.split()[0].replace("pp", "tg") in lk or ("tg128" in case and "tg128" in lk):
                pct = (lv / stock[case] - 1) * 100 if case in stock else 0
                print(f"\nDecode delta ({case} -> {lk}): {pct:+.1f}%")
PY

echo "Done. Results in ${OUT_ROOT}"
