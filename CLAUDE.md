# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A single-file Python benchmark tool (`bench.py`) for measuring local LLM inference performance (tokens/sec, TTFT, memory) on Apple Silicon. Targets models served via Ollama or any OpenAI-compatible API (LM Studio, mlx-lm, etc.).

## Commands

```bash
# Run benchmark
uv run bench.py --model qwen3:32b

# Multiple models
uv run bench.py --model qwen3:32b --model qwen3:8b

# List available models
uv run bench.py --list

# Compare saved results
uv run bench.py --compare results/baseline.json results/optimized.json

# Non-Ollama server
uv run bench.py --base-url http://localhost:1234/v1 --model some-model
```

No tests or linter configured. Single dependency: `httpx`.

## Python Environment

Use `uv` for all Python package management (not pip/pipx/brew). For CLI tools like mlx-lm:

```bash
uv tool install mlx-lm
```

## Architecture

Single module `bench.py` with two benchmark backends:
- **`bench_ollama_native`** — Streaming via Ollama's `/api/chat` endpoint. Supports `think: true/false` and Ollama-specific options (`num_ctx`, `num_batch`). Gets accurate TTFT and token counts from Ollama's response metadata.
- **`bench_openai_compat`** — Non-streaming via `/v1/chat/completions`. Used for LM Studio, mlx-lm, etc. No TTFT measurement (non-streaming).

Routing: `is_ollama()` checks for port 11434 in the URL to pick the backend.

Results are saved as JSON arrays of `BenchResult` dataclasses in `results/`. `RESULTS.md` contains analysis from M3 Max 48GB benchmarks.
