# Cluster setup — 3-node ring + 2-node vLLM

This document records the Promax GB10 cluster configuration, network changes, vLLM
deployment choices, and findings from the 3-node migration attempt (June 2026).

## Cluster inventory

| Role   | Hostname           | Mgmt IP       | Notes                          |
|--------|--------------------|---------------|--------------------------------|
| HEAD   | `promaxgb10-ee6d`  | `192.168.0.27` | Launch point, systemd, API     |
| WORKER | `promaxgb10-081c`  | `192.168.0.127`| vLLM TP rank 1                 |
| NODE3  | `test-gb10`        | `192.168.0.36` | Ring member; not in inference  |

SSH from HEAD uses mgmt aliases (`promax-worker`, `test-gb10`) and IPv6 link-local
fallbacks when QSFP routing is broken.

## Network topology

### Management (unchanged)

- Subnet: `192.168.0.0/24` on `enP7s7`
- Used for SSH, systemd orchestration, and (optionally) torch distributed master

### RoCE ring (rig-specific)

NVIDIA forum recipes use `192.168.0.x` on QSFP ports, which **collides with mgmt**
on this rig. We use **`192.168.177–182/24`** instead; last octet = mgmt host id:

| Link              | Subnets   | HEAD      | node3     | worker    |
|-------------------|-----------|-----------|-----------|-----------|
| HEAD ↔ node3 (f0) | 177, 178  | `.177.27` | `.177.36` | —         |
| HEAD ↔ worker (f1)| 179, 180  | `.179.27` | —         | `.179.127`|
| node3 ↔ worker    | 181, 182  | —         | `.181.36` | `.181.127`|

Netplan YAML lives outside this repo at `~/agent/cluster-ring-netplan/` on HEAD.

**Do not re-run** `spark_cluster_setup --run-setup` on all nodes in parallel — it
can overwrite netplan and break mgmt routing. Apply netplan one node at a time with
backups.

### RoCE health checks (launcher)

In a ring, nodes do **not** all share one HEAD QSFP address:

- Worker reaches HEAD on **`192.168.179.27`** (f1 link only)
- node3 reaches HEAD on **`192.168.177.27`** (f0 link only)
- Worker **cannot** ping `192.168.177.27` (no direct L3 path)

The systemd launcher (`cluster-launch.sh`) uses per-link variables:

| Variable              | Default            | Used by   |
|-----------------------|--------------------|-----------|
| `HEAD_QSFP_IP_WORKER` | `192.168.179.27`   | worker    |
| `HEAD_QSFP_IP_NODE3`  | `192.168.177.27`   | node3     |
| `WORKER_QSFP_IP`      | `192.168.179.127`  | head ping |
| `NODE3_QSFP_IP`       | `192.168.177.36`   | head ping |

## vLLM deployment (production: 2-node TP=2)

DeepSeek V4 Flash runs as **tensor parallel size 2** across HEAD + WORKER on the
**179.x** RoCE link (`enp1s0f1np1`).

### Per-node env files

| File         | `NODE_RANK` | `VLLM_HOST_IP`      | `NCCL_SOCKET_IFNAME` |
|--------------|-------------|---------------------|----------------------|
| `head.env`   | 0           | `192.168.179.27`    | `enp1s0f1np1`        |
| `worker.env` | 1           | `192.168.179.127`   | `enp1s0f1np1`        |
| `node3.env`  | 2           | `192.168.177.36`    | `enp1s0f0np0`        |

`MASTER_ADDR` is the head RoCE IP (`192.168.179.27`) for 2-node mode.

### Key docker-compose settings

| Parameter                    | Value   | Notes                                      |
|------------------------------|---------|---------------------------------------------|
| `--nnodes`                   | `2`     | HEAD + worker                               |
| `--tensor-parallel-size`     | `2`     | Required for 1M context                     |
| `--pipeline-parallel-size`   | `1`     | PP not supported for DeepSeek V4 in this image |
| `--max-model-len`            | `1000000` | 1M context                               |
| `--gpu-memory-utilization`   | `0.9`   | via `DSF1M_GPU_MEMORY_UTILIZATION`          |
| `--max-num-seqs`             | `4`     | via `DSF1M_MAX_NUM_SEQS`                   |
| Image                        | `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` | B12X MoE path |

### NCCL / RoCE

| Setting                    | Value                                      |
|----------------------------|--------------------------------------------|
| `NCCL_IB_GID_INDEX`        | `3` (RoCEv2 on 179 subnet, all nodes)      |
| `NCCL_IB_HCA`              | `rocep1s0f1,roceP2p1s0f1` (2-node f1 link) |
| `NCCL_CROSS_NIC`           | `1`                                        |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1`                                    |
| `VLLM_HOST_IP`             | Per-node QSFP IP (see env files)           |

**Lesson:** worker GID index **4** was tried early on; on the 179 subnet index **3**
is correct for all nodes after the ring netplan migration.

### Systemd service

```bash
./install-systemd-service.sh
sudo systemctl start deepseek-v4-flash-1m
journalctl -u deepseek-v4-flash-1m -f
```

Defaults in `/etc/default/deepseek-v4-flash-1m`:

- `CLUSTER_NNODES=2` — inference node count
- `STARTUP_TIMEOUT_SEC=5400` — model load can take ~8–10 min
- `MIN_CUDA_FREE_GB=100` — preflight GPU memory check
- `MAX_LAUNCH_ATTEMPTS=3` — retry with cleanup on failure

Launcher behavior:

1. Wait for worker SSH/Docker
2. Wait for RoCE link-up + per-link QSFP pings
3. Clean stale containers on all configured nodes
4. **Start head first**, wait for `:25000`
5. Start worker(s), stagger, wait for API health
6. Health monitor loop (systemd restart on repeated failures)

Manual start (no systemd):

```bash
./sync-cluster.sh          # rsync config to worker (+ node3)
./start-deepseek-v4-flash.sh
```

## 3-node migration — findings (June 2026)

Goal: use all three GB10 nodes for a single DeepSeek V4 Flash instance.

### Preparation completed

- Ring netplan applied on all 3 nodes (177–182 addressing)
- Docker image pulled on node3
- ~149 GB HF model cache rsynced to node3
- `node3.env`, `sync-cluster.sh`, 3-node-aware `cluster-launch.sh` added
- RoCE health-check bug fixed (per-link HEAD QSFP IPs)

### Why 3-node single-model inference fails

DeepSeek V4 Flash model config:

| Property              | Value | Divisible by 3? |
|-----------------------|-------|-----------------|
| `num_attention_heads` | 64    | No              |
| `num_hidden_layers`   | 43    | No              |
| `n_routed_experts`    | 256   | No              |

Attempted configurations:

| Config | Result |
|--------|--------|
| **TP=3, PP=1** | `ValidationError: 64 attention heads must be divisible by tensor parallel size (3)` |
| **TP=1, PP=3** | `NotImplementedError: Pipeline parallelism is not supported for this model` |

vLLM documents two viable 3-node patterns for other models ([spark-vllm-docker](https://github.com)):

- **PP=3** — model too large for 2 Sparks (not supported for DeepSeek V4 here)
- **DP=3** — full replica per node for concurrency (only if model fits one GB10; unlikely at 1M context)

**Conclusion:** keep **2-node TP=2** for 1M context. node3 remains in the ring for
future use (NCCL tests, DP at reduced context, or a separate model).

To re-enable 3-node launcher checks (without valid inference config):

```bash
# /etc/default/deepseek-v4-flash-1m
CLUSTER_NNODES=3
```

### Optional future: data-parallel 3

If shorter context is acceptable, test:

```yaml
--tensor-parallel-size 1
--pipeline-parallel-size 1
--data-parallel-size 3
--nnodes 3
```

Requires verifying single-GPU memory at target `max-model-len`. Not validated on this cluster.

## LiteLLM proxy

Optional OpenAI-compatible proxy on `:4000`:

```bash
./install-litellm-service.sh
sudo systemctl start deepseek-v4-flash-1m-litellm
```

Config: `litellm-config.yaml` → forwards to `http://127.0.0.1:8000/v1`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Cluster network not ready` | RoCE ping check wrong HEAD IP | Use per-link `HEAD_QSFP_IP_*` vars |
| `ibv_modify_qp failed` / empty GID | Wrong `NCCL_IB_GID_INDEX` | Use `3` on 179 subnet |
| Head starts, worker NCCL fails | Worker before head / wrong `MASTER_ADDR` | Head first; wait for `:25000` |
| `Insufficient CUDA memory` on boot | Stale containers or HF cache in GPU mem | Launcher cleanup + cache drop |
| TP=3 validation error | Model architecture | Use TP=2 only |
| PP=3 NotImplementedError | DeepSeek V4 + this vLLM build | Use TP=2 only |

## File reference

| File | Purpose |
|------|---------|
| `head.env` / `worker.env` / `node3.env` | Per-node vLLM/NCCL settings |
| `cluster-launch.sh` | Systemd launcher (2- or 3-node aware) |
| `sync-cluster.sh` | Rsync repo to worker + node3 |
| `install-systemd-service.sh` | Install systemd unit + defaults |
| `litellm-config.yaml` | LiteLLM proxy config |
| `docs/CLUSTER.md` | This document |
