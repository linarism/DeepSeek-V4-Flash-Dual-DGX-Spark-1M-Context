# GB10 clock cap benchmark — 2026-06-24

2-node DeepSeek V4 Flash 1M (TP=2), llama-benchy forum cases, 3 runs each.

Commands ([NVIDIA forum 372662](https://forums.developer.nvidia.com/t/cooler-gb10-temps-almost-no-performance-lost/372662)):

```bash
sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 0,3000   # stock cap
sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 0,2000   # tested cap
sudo nvidia-smi -rgc                                     # restore
```

## Throughput (tok/s)

| Case | Stock (~2.5 GHz load) | 2000 MHz | Δ |
|------|----------------------|----------|---|
| pp1024 prefill (c1) | 686.8 | 676.9 | −1.4% |
| tg128 decode (c1) | 36.7 | 34.7 | −5.6% |
| pp2048 prefill (c2) | 724.8 | 697.1 | −3.8% |
| tg128 decode (c2, per-req) | 25.7 | 27.4 | +6.9% |

Forum reference: ~45.7 tok/s decode c1, ~54.4 c2.

## Power & thermals (head + worker)

| Metric | Stock | 2000 MHz | Change |
|--------|-------|----------|--------|
| Avg GPU power | 65.8 W | 37.5 W | **−43%** |
| Peak GPU power | 79.2 W | 44.4 W | **−44%** |
| Peak GPU temp | 62 °C | 58 °C | −4 °C |

Per-node averages during benchmark: head 34.2→19.9 W, worker 31.6→17.6 W.

## Conclusion

2000 MHz cap is recommended for this cluster: ~43% less power, cooler peaks, ~1–6%
throughput loss on prefill/single-stream decode. Dual-concurrency decode within noise.

Reproduce: `./run-clock-power-bench.sh`
