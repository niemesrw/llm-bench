# Copilot Instructions

## What This Is

A single-file Python benchmark tool (`bench.py`) for measuring local LLM inference performance (tokens/sec, TTFT, memory) on Apple Silicon. Targets models served via Ollama or any OpenAI-compatible API (LM Studio, mlx-lm, etc.).

## Commands

```bash
# Install dependencies
uv sync

# Run benchmark
uv run bench.py --model qwen3:32b

# Multiple models
uv run bench.py --model qwen3:32b --model qwen3:8b

# List available models on server
uv run bench.py --list

# Compare saved results
uv run bench.py --compare results/baseline.json results/optimized.json

# Non-Ollama server (LM Studio, mlx-lm, etc.)
uv run bench.py --base-url http://localhost:1234/v1 --model some-model

# Ollama-specific tuning flags
uv run bench.py --model qwen3:32b --num-ctx 8192 --num-batch 512 --think
```

No tests or linter configured. Single dependency: `httpx`.

## Architecture

Everything lives in `bench.py`. Two benchmark backends, routed by `is_ollama()` (detects port 11434 in the URL):

- **`bench_ollama_native`** — Streaming via `/api/chat`. Records accurate TTFT from first content chunk. Gets token counts from Ollama's `eval_count`/`prompt_eval_count` in the final `done` chunk. Supports `think: true/false` and Ollama options (`num_ctx`, `num_batch`).
- **`bench_openai_compat`** — Non-streaming via `/v1/chat/completions`. No TTFT (returns 0). Falls back to `len(content) // 4` if the server doesn't return `usage`.

`BenchResult` is a dataclass serialized with `dataclasses.asdict()` to JSON. Results are saved to `results/<tag>_<timestamp>.json`. Community results go in `results/community/`.

## Key Conventions

- **Backend detection is port-based**: `is_ollama()` checks for `"11434"` in the URL string. Non-Ollama servers always use `bench_openai_compat` regardless of their actual API.
- **TTFT is Ollama-only**: `bench_openai_compat` always sets `time_to_first_token_ms=0`. Don't add TTFT logic there without switching to streaming.
- **Warmup is on by default**: A short dummy request loads the model before timing starts. Disable with `--no-warmup`.
- **Token count fallback**: If the OpenAI endpoint doesn't return `usage`, token counts are estimated as `len(text) // 4`.
- **`--tag` controls result filenames**: Saves as `results/<tag>_<timestamp>.json`. Without a tag, just `results/<timestamp>.json`.
- **`--model` is repeatable**: `argparse` uses `action="append"` — pass it multiple times to benchmark multiple models in one run.
- **Adding prompts**: The `PROMPTS` dict at the top of `bench.py` defines all benchmark prompts. The `--prompts` flag accepts prompt names or `all`.
