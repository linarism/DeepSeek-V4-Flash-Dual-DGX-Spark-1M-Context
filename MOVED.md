# Moved to DSpark repo

This stack is superseded by:

**https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark**

Local clone on this cluster:

```text
~/agent/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
```

| | Legacy (this repo) | DSpark |
|--|-------------------|--------|
| Model | `DeepSeek-V4-Flash` | `DeepSeek-V4-Flash-DSpark` |
| KV cache | FP8 | `nvfp4_ds_mla` |
| Speculative | MTP (2) | DSpark (5) |
| API port | 8000 | 8888 |
| Image | `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` | `vllm-dspark-runtime:dspark-nvfp4-stage-c` (local build) |

Cluster-specific notes: `docs/cluster/promax-gb10.md` in the DSpark repo.
