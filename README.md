# DeepSeek V4 Flash – Dual DGX Spark (1M Context)

Deploy [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) — a Mixture-of-Experts (MoE) reasoning model — across **two NVIDIA DGX Spark** nodes with **1 million token context length**, InfiniBand interconnect, and FP8 KV-cache.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

## Overview

This repository provides a ready-to-run `docker-compose.yml` and helper scripts that launch a vLLM inference server serving DeepSeek V4 Flash. The setup uses:

- **2 × DGX Spark** nodes (spark1 = head, spark2 = worker)
- **Tensor parallelism (TP)** across both GPUs
- **InfiniBand** (NCCL over IB) for inter-node communication
- **FP8 KV-cache** (`--kv-cache-dtype fp8`) for memory efficiency
- **Multi-Token Prediction (MTP)** speculative decoding (2 draft tokens)
- **Prefix caching** and **FlashInfer autotune**
- **Tool calling** and **reasoning** support (DeepSeek V4 parsers)

> **Default generation config:** The `docker-compose.yml` sets `--override-generation-config '{"temperature":0.6,"top_p":0.95}'` — tuned for coding tasks. Remove this line to use DeepSeek V4 Flash's default sampling parameters.

## Requirements

### Hardware

| Component | Requirement |
|-----------|------------|
| Nodes     | 2 × DGX Spark (Grace Hopper, GB10) |
| Interconnect | InfiniBand (e.g., ConnectX-7) between nodes |
| Storage   | Sufficient for model weights (~400 GB with HF cache) |

### Software

- **Docker** with `docker compose` plugin (v2.24+)
- **Passwordless SSH** from spark1 → spark2
- **NVIDIA Container Toolkit** (`nvidia-ctk`) — installed by default on DGX Spark
- **InfiniBand drivers** (`ibdev2netdev`, `ibstat`) — installed by default
- **Git** and **curl**

## Quick Start

### 1. Clone the repo on both nodes

```bash
git clone https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context.git
cd DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context
```

Run this on **both** spark1 (head) and spark2 (worker).

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to match your cluster:

| Variable | Description | Example |
|----------|-------------|---------|
| `NODE_RANK` | `0` for head (spark1), `1` for worker (spark2) | `0` |
| `HEADLESS` | Set to `1` on worker nodes | *(empty for head)* |
| `MASTER_ADDR` | IP address of the head node (spark1) | `192.168.1.100` |
| `WORKER_HOST` | IP address of the worker node (spark2) | `192.168.1.101` |
| `HF_CACHE` | Path to your HuggingFace cache | `${HOME}/.cache/huggingface` |
| `NCCL_IB_HCA` | InfiniBand HCA device (run `ibdev2netdev -v`) | `rocep1s0f1` |
| `NCCL_SOCKET_IFNAME` | Network interface for socket comms | `enp1s0f1np1` |

> **Finding your InfiniBand interface:**
> ```bash
> ibdev2netdev -v
> ```
> Pick the `rocep*` device connected to your IB link.

> **Finding your socket interface:**
> ```bash
> ip addr show | grep 169.254
> ```
> Pick the interface with the link-local IP used for NCCL out-of-band communication.

### 3. Set NODE_RANK on each node

On **spark1** (head):
```bash
NODE_RANK=0
```

On **spark2** (worker):
```bash
NODE_RANK=1
HEADLESS=1
```

Update `.env` accordingly on each node.

### 4. Start the server

From **spark1** only:

```bash
./start-deepseek-v4-flash.sh
```

This script:
1. SSHs into spark2 and starts the container there
2. Starts the container on spark1
3. Polls `http://127.0.0.1:8000/v1/models` until the API is ready (up to ~20 minutes)

### 5. Stop the server

From **spark1** only:

```bash
./stop-deepseek-v4-flash.sh
```

## API Usage

Once running, the OpenAI-compatible vLLM API is available on both nodes at `http://localhost:8000`.

### Chat completion (with reasoning)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "What is 47 × 89?"}
    ],
    "temperature": 0.0
  }'
```

### Streaming

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "Write a haiku about distributed inference"}
    ],
    "stream": true
  }'
```

### Tool calling

The server is configured with `--tool-call-parser deepseek_v4` and `--enable-auto-tool-choice` for function-calling tasks.

## Architecture

```
┌─────────────────────────┐       InfiniBand        ┌─────────────────────────┐
│     spark1 (head)       │◄───────────────────────►│    spark2 (worker)      │
│  NODE_RANK=0            │   NCCL (IB / sockets)    │  NODE_RANK=1            │
│  ┌───────────────────┐  │                          │  ┌───────────────────┐  │
│  │ vLLM container    │  │                          │  │ vLLM container    │  │
│  │ TP rank 0 (GPU 0) │  │                          │  │ TP rank 1 (GPU 1) │  │
│  │ Port 8000 (API)   │  │                          │  │ Port 8000 (API)   │  │
│  └───────────────────┘  │                          │  └───────────────────┘  │
└─────────────────────────┘                          └─────────────────────────┘
```

### Key vLLM parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--tensor-parallel-size` | `2` | Split model across 2 GPUs |
| `--pipeline-parallel-size` | `1` | No pipeline parallelism |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache for memory savings |
| `--max-model-len` | `1000000` | 1M token context |
| `--max-num-seqs` | `4` | Max concurrent sequences |
| `--block-size` | `256` | PagedAttention block size |
| `--gpu-memory-utilization` | `0.9` | GPU memory budget |
| `--enable-prefix-caching` | enabled | Reuse KV cache across requests |
| `--speculative-config` | MTP, 2 tokens | Multi-Token Prediction |
| `--distributed-executor-backend` | `mp` | Multi-process backend |
| `--nnodes` | `2` | Two-node deployment |

## Docker Images

The compose file uses `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` — a custom vLLM build optimized for DGX Spark (GB10, CUDA arch 12.1a, FlashInfer for Hopper).

If you need to rebuild or use a different image, update the `image:` field in `docker-compose.yml`.

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Container definition with volume mounts, env vars, and entrypoint |
| `head.env` / `worker.env` | Per-node env for head and worker (see `node3.env` for 3rd node) |
| `.env.example` | Template for environment configuration |
| `start-deepseek-v4-flash.sh` | Start server on both nodes (manual) |
| `stop-deepseek-v4-flash.sh` | Stop server on both nodes |
| `cluster-launch.sh` | Systemd launcher with RoCE checks and retries |
| `sync-cluster.sh` | Rsync config to worker and node3 |
| `install-systemd-service.sh` | Install `deepseek-v4-flash-1m.service` |
| `install-gpu-clock-cap.sh` | Persistent GB10 GPU clock cap (2000 MHz) |
| `run-clock-power-bench.sh` | Benchmark throughput vs power at stock vs capped clocks |
| `litellm-config.yaml` | Optional LiteLLM proxy on `:4000` |
| `docs/CLUSTER.md` | **Cluster network, 3-node findings, NCCL notes** |
| `.gitignore` | Files excluded from version control |

> For Promax GB10 3-node ring setup, RoCE addressing, and why TP=3/PP=3 fail for
> DeepSeek V4 Flash, see **[docs/CLUSTER.md](docs/CLUSTER.md)**.

## Troubleshooting

### NCCL / InfiniBand

- Verify IB link state: `ibstat` (should show `LinkUp: true`)
- Test passwordless SSH: `ssh <WORKER_HOST> hostname`
- Check NCCL debug logs (set `NCCL_DEBUG=INFO` in compose env)

### Container fails to start

```bash
docker compose logs
```

### CUDA out of memory

Reduce `--gpu-memory-utilization` (e.g., `0.75`) or `--max-model-len`.

### Timeout waiting for API

The model loads from HuggingFace on first run. Check logs:
```bash
docker logs deepseek-v4-flash-vllm-1
```

## License

This repository's code is provided under the MIT License. The DeepSeek V4 Flash model weights are subject to [DeepSeek's license](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).
