#!/usr/bin/env python3
"""Lightweight LLM benchmark for local inference on Apple Silicon.

Measures tokens/sec, time-to-first-token, and peak memory for models
served via any OpenAI-compatible API (Ollama, LM Studio, mlx-lm, etc).

Usage:
    # Benchmark a single model on Ollama (uses native API, disables thinking)
    uv run bench.py --model qwen3:32b

    # Compare multiple models
    uv run bench.py --model qwen3:32b --model qwen3:8b --model deepseek-r1:14b

    # Use LM Studio instead (default port 1234)
    uv run bench.py --base-url http://localhost:1234/v1 --model qwen3-8b-4bit

    # Custom prompts and iterations
    uv run bench.py --model qwen3:32b --iterations 5 --max-tokens 500

    # List available models on the server
    uv run bench.py --list
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Missing dependency: httpx")
    print("Run: uv pip install httpx")
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent / "results"

PROMPTS = {
    "short": "What is the capital of France?",
    "medium": (
        "Explain the difference between TCP and UDP protocols. "
        "Cover reliability, ordering, use cases, and performance trade-offs."
    ),
    "long": (
        "Write a Python function that implements a least-recently-used (LRU) cache "
        "with O(1) get and put operations. Include type hints, docstrings, and a "
        "brief explanation of the data structures used. Then write 3 unit tests for it."
    ),
    "reasoning": (
        "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer "
        "have left? Think step by step."
    ),
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass
class BenchResult:
    model: str
    prompt_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_to_first_token_ms: float
    total_time_s: float
    tokens_per_second: float
    base_url: str
    timestamp: str
    max_tokens: int
    temperature: float


def is_ollama(base_url: str) -> bool:
    """Check if the base URL points to an Ollama server."""
    return "11434" in base_url


def get_models(base_url: str) -> list[str]:
    """List models available on the server."""
    url = f"{base_url}/models" if "/v1" in base_url else f"{base_url}/v1/models"
    try:
        r = httpx.get(url, timeout=5)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def get_ollama_ps() -> str | None:
    """Get memory usage from ollama ps (if available)."""
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def bench_ollama_native(
    base_url: str,
    model: str,
    prompt_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    think: bool = False,
    ollama_options: dict | None = None,
) -> BenchResult | None:
    """Benchmark using Ollama's native /api/chat (supports think:false)."""
    # Strip /v1 suffix if present
    api_base = base_url.replace("/v1", "")
    url = f"{api_base}/api/chat"
    options = {
        "num_predict": max_tokens,
        "temperature": temperature,
    }
    if ollama_options:
        options.update(ollama_options)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": think,
        "options": options,
    }

    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    start = time.perf_counter()

    try:
        with httpx.stream(
            "POST", url, json=payload, timeout=httpx.Timeout(120.0)
        ) as response:
            if response.status_code != 200:
                response.read()
                print(f"  HTTP error: {response.status_code} — {response.text[:200]}")
                return None
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = chunk.get("message", {}).get("content", "")
                if content and ttft is None:
                    ttft = (time.perf_counter() - start) * 1000

                # Final chunk has eval_count
                if chunk.get("done"):
                    completion_tokens = chunk.get("eval_count", 0)
                    prompt_tokens = chunk.get("prompt_eval_count", 0)

    except httpx.ConnectError:
        print(f"  Connection refused at {api_base} — is Ollama running?")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

    total_time = time.perf_counter() - start
    if completion_tokens == 0:
        completion_tokens = 1

    return BenchResult(
        model=model,
        prompt_name=prompt_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        time_to_first_token_ms=round(ttft, 1) if ttft else 0,
        total_time_s=round(total_time, 2),
        tokens_per_second=round(completion_tokens / total_time, 1) if total_time > 0 else 0,
        base_url=base_url,
        timestamp=datetime.now().isoformat(),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def bench_openai_compat(
    base_url: str,
    model: str,
    prompt_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> BenchResult | None:
    """Benchmark using OpenAI-compatible /v1/chat/completions (LM Studio, mlx-lm, etc).

    Uses non-streaming to get accurate token counts from the server.
    Measures wall-clock time which includes TTFT + generation.
    """
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start = time.perf_counter()

    try:
        response = httpx.post(url, json=payload, timeout=httpx.Timeout(120.0))
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as e:
        print(f"  HTTP error: {e.response.status_code} — {e.response.text[:200]}")
        return None
    except httpx.ConnectError:
        print(f"  Connection refused at {base_url} — is the server running?")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

    total_time = time.perf_counter() - start

    # Use server-reported token counts if available
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    if completion_tokens == 0:
        # Fallback: estimate from content
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        completion_tokens = max(1, len(content) // 4)
        prompt_tokens = max(1, len(prompt) // 4)

    return BenchResult(
        model=model,
        prompt_name=prompt_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        time_to_first_token_ms=0,  # not measurable without streaming
        total_time_s=round(total_time, 2),
        tokens_per_second=round(completion_tokens / total_time, 1) if total_time > 0 else 0,
        base_url=base_url,
        timestamp=datetime.now().isoformat(),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def bench_single(
    base_url: str,
    model: str,
    prompt_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    think: bool = False,
    ollama_options: dict | None = None,
) -> BenchResult | None:
    """Route to the appropriate backend."""
    if is_ollama(base_url):
        return bench_ollama_native(base_url, model, prompt_name, prompt, max_tokens, temperature, think=think, ollama_options=ollama_options)
    return bench_openai_compat(base_url, model, prompt_name, prompt, max_tokens, temperature)


def ensure_ollama_model(base_url: str, model: str) -> bool:
    """Check if an Ollama model is available locally; offer to pull if not."""
    api_base = base_url.replace("/v1", "")
    try:
        r = httpx.post(f"{api_base}/api/show", json={"name": model}, timeout=5)
        if r.status_code == 200:
            return True
    except httpx.ConnectError:
        print(f"  Connection refused at {api_base} — is Ollama running?")
        return False
    except Exception:
        pass

    # Model not found — offer to pull
    print(f"\n  Model '{model}' not found locally.")
    try:
        answer = input(f"  Pull '{model}' from Ollama? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if answer and answer not in ("y", "yes"):
        print("  Skipping model.")
        return False

    print(f"  Pulling '{model}' (this may take a while)...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            timeout=3600,
        )
        if result.returncode == 0:
            print(f"  Pull complete.")
            return True
        else:
            print(f"  Pull failed (exit code {result.returncode}).")
            return False
    except FileNotFoundError:
        print("  'ollama' command not found. Install Ollama from https://ollama.com")
        return False
    except subprocess.TimeoutExpired:
        print("  Pull timed out.")
        return False


def run_benchmark(
    base_url: str,
    models: list[str],
    prompt_names: list[str],
    iterations: int,
    max_tokens: int,
    temperature: float,
    warmup: bool,
    think: bool = False,
    ollama_options: dict | None = None,
) -> list[BenchResult]:
    """Run the full benchmark suite."""
    all_results = []

    for model in models:
        # For Ollama, check model availability and offer to pull
        if is_ollama(base_url) and not ensure_ollama_model(base_url, model):
            print(f"\n  Skipping '{model}' — not available.")
            continue
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        if warmup:
            print("  Warming up (loading model into memory)...")
            bench_single(base_url, model, "warmup", "Hi", 10, 0, think=think, ollama_options=ollama_options)
            print("  Warm.")

        for pname in prompt_names:
            prompt = PROMPTS[pname]
            iter_results = []

            for i in range(iterations):
                label = f"  [{pname}] iteration {i+1}/{iterations}"
                print(f"{label}...", end="", flush=True)
                result = bench_single(base_url, model, pname, prompt, max_tokens, temperature, think=think, ollama_options=ollama_options)
                if result:
                    iter_results.append(result)
                    print(
                        f" {result.tokens_per_second} tok/s, "
                        f"TTFT {result.time_to_first_token_ms}ms, "
                        f"{result.completion_tokens} tokens in {result.total_time_s}s"
                    )
                else:
                    print(" FAILED")

            all_results.extend(iter_results)

        # Show memory info if ollama
        ps = get_ollama_ps()
        if ps:
            print(f"\n  ollama ps:\n  {ps}")

    return all_results


def print_summary(results: list[BenchResult]):
    """Print a summary table of results."""
    if not results:
        print("\nNo results to summarize.")
        return

    by_model: dict[str, list[BenchResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Avg tok/s':>10} {'Avg TTFT':>10} {'Runs':>6}")
    print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*6}")

    for model, runs in by_model.items():
        avg_tps = sum(r.tokens_per_second for r in runs) / len(runs)
        avg_ttft = sum(r.time_to_first_token_ms for r in runs) / len(runs)
        print(f"{model:<30} {avg_tps:>10.1f} {avg_ttft:>8.1f}ms {len(runs):>6}")

    print(f"\n{'Model':<25} {'Prompt':<12} {'tok/s':>8} {'TTFT':>8} {'Time':>8}")
    print(f"{'-'*25} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

    for model, runs in by_model.items():
        by_prompt: dict[str, list[BenchResult]] = {}
        for r in runs:
            by_prompt.setdefault(r.prompt_name, []).append(r)
        for pname, pruns in by_prompt.items():
            avg_tps = sum(r.tokens_per_second for r in pruns) / len(pruns)
            avg_ttft = sum(r.time_to_first_token_ms for r in pruns) / len(pruns)
            avg_time = sum(r.total_time_s for r in pruns) / len(pruns)
            print(
                f"{model:<25} {pname:<12} {avg_tps:>7.1f} {avg_ttft:>6.0f}ms {avg_time:>6.1f}s"
            )


def save_results(results: list[BenchResult], tag: str | None):
    """Save results to a JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{tag}_{ts}" if tag else ts
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {path}")


def compare_results(files: list[Path]):
    """Compare results from multiple saved benchmark runs."""
    all_runs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            tag = f.stem
            for d in data:
                d["_file"] = tag
            all_runs.extend(data)

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"{'Run':<25} {'Model':<25} {'Avg tok/s':>10} {'Avg TTFT':>10}")
    print(f"{'-'*25} {'-'*25} {'-'*10} {'-'*10}")

    by_file: dict[str, list] = {}
    for r in all_runs:
        by_file.setdefault(r["_file"], []).append(r)

    for fname, runs in by_file.items():
        by_model: dict[str, list] = {}
        for r in runs:
            by_model.setdefault(r["model"], []).append(r)
        for model, mruns in by_model.items():
            avg_tps = sum(r["tokens_per_second"] for r in mruns) / len(mruns)
            avg_ttft = sum(r["time_to_first_token_ms"] for r in mruns) / len(mruns)
            print(f"{fname:<25} {model:<25} {avg_tps:>10.1f} {avg_ttft:>8.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Lightweight LLM benchmark for Apple Silicon")
    parser.add_argument(
        "--base-url", default=f"{DEFAULT_OLLAMA_URL}/v1",
        help="OpenAI-compatible API base URL (default: Ollama)"
    )
    parser.add_argument("--model", action="append", dest="models", help="Model(s) to benchmark")
    parser.add_argument(
        "--prompts", nargs="+", choices=list(PROMPTS.keys()) + ["all"],
        default=["all"], help="Which prompts to run"
    )
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per prompt (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens to generate (default: 300)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0)")
    parser.add_argument("--tag", help="Tag for saving results (e.g. 'baseline', 'optimized')")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode (Ollama only)")
    parser.add_argument("--num-ctx", type=int, help="Ollama context size (default: 4096)")
    parser.add_argument("--num-batch", type=int, help="Ollama batch size for prompt eval")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup request")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument(
        "--compare", nargs="+", type=Path,
        help="Compare saved result files (e.g. results/baseline.json results/optimized.json)"
    )
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare)
        return

    if args.list:
        models = get_models(args.base_url)
        if models:
            print("Available models:")
            for m in models:
                print(f"  {m}")
        else:
            print("No models found (is the server running?)")
        return

    if not args.models:
        parser.error("Specify at least one --model (or use --list to see available models)")

    prompt_names = list(PROMPTS.keys()) if "all" in args.prompts else args.prompts
    backend = "Ollama native API" if is_ollama(args.base_url) else "OpenAI-compat API"

    print(f"LLM Bench — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Server:     {args.base_url}")
    print(f"Backend:    {backend}")
    print(f"Models:     {', '.join(args.models)}")
    print(f"Prompts:    {', '.join(prompt_names)}")
    print(f"Iterations: {args.iterations}")
    print(f"Max tokens: {args.max_tokens}")

    ollama_opts = {}
    if args.num_ctx:
        ollama_opts["num_ctx"] = args.num_ctx
    if args.num_batch:
        ollama_opts["num_batch"] = args.num_batch

    results = run_benchmark(
        base_url=args.base_url,
        models=args.models,
        prompt_names=prompt_names,
        iterations=args.iterations,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup=not args.no_warmup,
        think=args.think,
        ollama_options=ollama_opts or None,
    )

    print_summary(results)
    save_results(results, args.tag)


if __name__ == "__main__":
    main()
