# llm-bench

A lightweight Python benchmark for measuring local LLM inference performance (tokens/sec, TTFT, memory) on Apple Silicon. Works with Ollama, LM Studio, mlx-lm, or any OpenAI-compatible API.

## Quick Start

```bash
# Clone and install
git clone https://github.com/niemesrw/llm-bench.git
cd llm-bench
uv sync

# Benchmark a model (Ollama)
uv run bench.py --model qwen3:32b

# Multiple models
uv run bench.py --model qwen3:32b --model llama3.2

# List available models
uv run bench.py --list

# Non-Ollama server (LM Studio, mlx-lm, etc.)
uv run bench.py --base-url http://localhost:1234/v1 --model some-model
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Model name(s) to benchmark (repeatable) |
| `--base-url` | `http://localhost:11434/v1` | API endpoint |
| `--iterations` | 3 | Runs per prompt |
| `--max-tokens` | 300 | Max tokens to generate |
| `--temperature` | 0.0 | Sampling temperature |
| `--think` | off | Enable thinking mode (Ollama only) |
| `--tag` | none | Label for the results file |
| `--compare` | none | Compare saved JSON result files |
| `--list` | — | List models on the server |

## How It Works

The benchmark runs 4 prompts (short, medium, long, reasoning) across N iterations per model and reports:

- **Tokens/sec** — generation throughput
- **TTFT** — time to first token (Ollama streaming only)
- **Token counts** — from server metadata

Two backends:
- **Ollama native** (`/api/chat`, streaming) — auto-detected on port 11434. Supports `think: true/false` and Ollama-specific tuning options.
- **OpenAI-compatible** (`/v1/chat/completions`, non-streaming) — for LM Studio, mlx-lm, vLLM, etc.

Results are saved as JSON in `results/`.

## Community Results

> **Want to add your hardware?** Run the benchmark, then open a PR adding a row to the table below and your result JSON in `results/community/`.

### Results Table

| Hardware | Model | Type | Size | Backend | Avg tok/s | Avg TTFT | Contributor |
|----------|-------|------|------|---------|-----------|----------|-------------|
| M3 Max 48GB | llama3.2 (3B) | Dense | 2.0 GB | Ollama | **106.8** | 110ms | [@niemesrw](https://github.com/niemesrw) |
| M3 Max 48GB | mistral (7B) | Dense | 4.1 GB | Ollama | **65.8** | 80ms | [@niemesrw](https://github.com/niemesrw) |
| M3 Max 48GB | qwen2.5:32b | Dense | 19 GB | Ollama | **16.8** | 275ms | [@niemesrw](https://github.com/niemesrw) |
| M3 Max 48GB | qwen3:32b | Dense | 20 GB | Ollama | **15.9** | 309ms | [@niemesrw](https://github.com/niemesrw) |
| M3 Max 48GB | qwen3:32b | Dense | 20 GB | mlx-lm | **18.9** | — | [@niemesrw](https://github.com/niemesrw) |
| M3 Max 48GB | qwen3:30b-a3b | MoE | 18 GB | Ollama | **79.7** | 142ms | [@niemesrw](https://github.com/niemesrw) |
| M3 Max 48GB | qwen3.5:27b | Dense | 24 GB | Ollama | **12.5** | 628ms | [@niemesrw](https://github.com/niemesrw) |

### Key Findings (M3 Max 48GB)

- **LLM inference is memory-bandwidth bound** on Apple Silicon. The M3 Max at ~400 GB/s sets the ceiling.
- **Smaller models scale linearly** with size: 3B ≈ 107 tok/s, 7B ≈ 66 tok/s, 32B ≈ 17 tok/s.
- **MLX is ~13% faster** than Ollama/llama.cpp for the same 32B model.
- **MoE is the sweet spot** — `qwen3:30b-a3b` hits 80 tok/s (5x faster than dense 32B) with quality trained on 30B params.
- **Ollama tuning knobs** (flash attention, context size, batch size) have zero impact on generation speed at 32B — the bottleneck is memory bandwidth, not software.
- **qwen3.5 trades speed for quality** — 21% slower than qwen3:32b but meaningfully better reasoning.

## Contributing Results

1. Fork this repo
2. Run the benchmark on your hardware:
   ```bash
   uv run bench.py --model <your-model> --tag <your-hardware>
   ```
3. Copy your result JSON to `results/community/`
4. Add a row to the table above
5. Open a PR

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (or pip)
- A running LLM server (Ollama, LM Studio, etc.)

## License

MIT
