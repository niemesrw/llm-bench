# LLM Bench Results — M3 Max (48GB)

## Hardware
- Apple M3 Max, 48GB unified memory
- macOS 26.3 (Darwin 25.3.0)
- Memory bandwidth: ~400 GB/s

## Baseline — 2025-02-14

### All Models (Ollama, think disabled)

| Model | Size | Backend | Avg tok/s | Avg TTFT | Notes |
|-------|------|---------|-----------|----------|-------|
| llama3.2 (3B) | 2.0 GB | Ollama/llama.cpp | **106.8** | 110ms | Fastest, lowest quality |
| mistral (7B) | 4.1 GB | Ollama/llama.cpp | **65.8** | 80ms | Good speed, low latency |
| qwen2.5:32b | 19 GB | Ollama/llama.cpp | **16.8** | 275ms | Best quality of original set |
| qwen3:32b | 20 GB | Ollama/llama.cpp | **15.9** | 309ms | think disabled via native API |

### Ollama vs MLX — qwen3:32b (think enabled, 300 max tokens)

| Backend | Engine | Avg tok/s | Notes |
|---------|--------|-----------|-------|
| Ollama | llama.cpp/Metal | **16.7** | Native API with `think: true` |
| mlx-lm | MLX | **18.9** | OpenAI-compat API, non-streaming |

**MLX is ~13% faster** than Ollama's llama.cpp backend for the same model.

### Per-Prompt Breakdown (Ollama baseline, think disabled)

| Model | short | medium | long | reasoning |
|-------|-------|--------|------|-----------|
| llama3.2 | 42.4 | 148.3 | 136.7 | 99.8 |
| mistral | 48.0 | 88.3 | 66.8 | 60.0 |
| qwen2.5:32b | 11.5 | 19.9 | 20.5 | 15.4 |
| qwen3:32b | 13.4 | 17.0 | 16.9 | 16.4 |

*Values are tok/s. Short prompts show lower tok/s because the prompt eval overhead dominates on few output tokens.*

## Key Observations

1. **LLM inference is memory-bandwidth bound** on Apple Silicon. The M3 Max at 400 GB/s sets the ceiling.
2. **32B models run at ~17-19 tok/s** regardless of backend — this is close to the theoretical limit for a 20GB model on 400 GB/s bandwidth.
3. **MLX advantage is modest (~13%)** at 32B. May be larger for smaller models or longer contexts.
4. **Smaller models scale linearly** with size: 3B = ~107 tok/s, 7B = ~66 tok/s, 32B = ~17 tok/s.
5. **qwen3 thinking mode** burns tokens on internal reasoning. Use `think: false` (Ollama native API) for pure output speed. mlx-lm server doesn't support disabling thinking.
6. **All 32B models fit comfortably** in 48GB with room for OS + apps.

## Tools

- Benchmark script: `bench.py` in this directory
- Raw results: `results/` directory (JSON)
- Compare runs: `uv run bench.py --compare results/file1.json results/file2.json`

## Ollama Tuning Results — 2025-02-14

Tested all tuning knobs on qwen3:32b (think disabled, 300 max tokens, 2 iterations):

| Setting | Value | Avg tok/s | Delta vs baseline |
|---------|-------|-----------|-------------------|
| **Baseline** (defaults) | ctx=4096, batch=512 | **15.9** | — |
| Flash attention | `OLLAMA_FLASH_ATTENTION=1` | **15.9** | 0% |
| Smaller context | `num_ctx=2048` | **15.9** | 0% |
| Larger batch | `num_batch=1024` | **15.9** | 0% |

**Conclusion: Ollama tuning has zero impact on generation speed for 32B models.**

This confirms the hardware bottleneck — token generation is purely memory-bandwidth bound
at this model size. The M3 Max's 400 GB/s bandwidth is the ceiling. No software tuning
can overcome it. The ~13% MLX advantage likely comes from more efficient Metal shader
implementations, not configuration.

### What *would* help

- **Hardware**: M4 Max (546 GB/s) or Mac Studio with M2/M3/M4 Ultra (800+ GB/s)
- **Smaller models**: Qwen3 8B at ~60+ tok/s with much better quality/speed ratio
- **Quantization**: Already using Q4, going lower (Q3/Q2) trades quality for speed
- **MoE models**: Qwen3-30B-A3B — only 3B params active, so ~100+ tok/s theoretical

## MoE Results — 2025-02-14

| Model | Type | Size | Avg tok/s | Avg TTFT | vs qwen3:32b |
|-------|------|------|-----------|----------|--------------|
| qwen3:32b | Dense | 20 GB | **15.9** | 309ms | baseline |
| **qwen3:30b-a3b** | **MoE** | **18 GB** | **79.7** | **142ms** | **5x faster** |

The MoE model activates only 3B of its 30B parameters per token, giving near-3B speed
with 30B training quality. This is the **best model for local use on M3 Max 48GB**:
- 5x faster than dense 32B
- Half the TTFT latency
- Uses slightly less RAM (18 GB vs 20 GB)
- Trained on full 30B params so quality >> a true 3B model

## Recommendation

**For OpenClaw / local assistant use: `qwen3:30b-a3b` on Ollama.**

It hits 80 tok/s — fast enough to feel snappy — with quality that punches well above
its active parameter count. No need for MLX or tuning gymnastics.

## Qwen 3.5 Generation — 2026-02-27

### Performance

| Model | Type | Size (GPU) | Avg tok/s | Avg TTFT | vs qwen3:32b |
|-------|------|------------|-----------|----------|--------------|
| qwen3:32b | Dense | 20 GB | **15.9** | 309ms | baseline |
| qwen3.5:27b | Dense | 24 GB | **12.5** | 628ms | **21% slower** |

Per-prompt breakdown (think disabled):

| Model | short | medium | long | reasoning |
|-------|-------|--------|------|-----------|
| qwen3:32b | 13.4 | 17.0 | 16.9 | 16.4 |
| qwen3.5:27b | 12.1 | 12.7 | 12.6 | 12.6 |

### Quality vs Speed

Despite fewer parameters (27B vs 32B), qwen3.5:27b is slower and uses more GPU memory (24 GB vs 20 GB), suggesting a denser architecture with larger embeddings/attention heads.

In a head-to-head coding test (LIS algorithm with O(n log n) implementation):
- **qwen3.5:27b** produced correct reconstruction logic with proper predecessor tracking
- **qwen3:32b** had a subtle O(n^2) bug in its reconstruction helper, breaking the complexity guarantee

The 3.5 generation trades ~20% throughput for meaningfully better reasoning quality on coding tasks.

### Recommendation Update

- **Speed priority**: Still use `qwen3:30b-a3b` (MoE) at 80 tok/s
- **Quality priority**: `qwen3.5:27b` at 12.5 tok/s — best local reasoning, but slow
- **Balanced**: `qwen3:32b` at 15.9 tok/s — good enough quality, 27% faster than 3.5

## Next Steps

- [ ] Test smaller MLX models (8B) to see if speed gap widens
- [ ] Try Ollama's upcoming MLX backend when stable
- [ ] Benchmark with longer context windows (8K, 16K)
- [ ] Test quality (MMLU, HumanEval) to confirm MoE doesn't sacrifice too much
