"""CUDA inference benchmarks for comparing small language models."""

import argparse
import statistics
import time

import torch

from lab_01.inference import generate_once
from lab_01.models import load_model


def run_generation_benchmark(args: argparse.Namespace) -> None:
    """Run the original chat-generation benchmark."""
    started = time.perf_counter()
    model, tokenizer = load_model(args.model)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    generate_once(model, tokenizer, args.model, args.prompt, args.max_new_tokens)
    results = [
        generate_once(model, tokenizer, args.model, args.prompt, args.max_new_tokens)
        for _ in range(args.runs)
    ]
    tps = [result["tps"] for result in results]

    print("\n--- RESULT ---")
    print(results[-1]["text"])
    print("\n--- METRICS ---")
    print(f"model: {args.model}")
    print(f"load_time_s: {load_seconds:.2f}")
    print(f"runs: {args.runs}")
    print(f"generated_tokens: {results[-1]['tokens']}")
    print(f"tokens_per_second_mean: {statistics.mean(tps):.2f}")
    print(f"tokens_per_second_median: {statistics.median(tps):.2f}")
    print(f"tokens_per_second_min: {min(tps):.2f}")
    print(f"tokens_per_second_max: {max(tps):.2f}")
    print(f"peak_vram_gb: {max(result['vram'] for result in results):.2f}")


def generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    return parser


def decode_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=5)
    return parser


def run_decode_benchmark(args: argparse.Namespace) -> None:
    """Measure eager fixed-length decode throughput.

    torch.compile is intentionally omitted: it did not provide a meaningful
    benefit in the lab's measurements.
    """
    model, tokenizer = load_model(args.model)
    prompt = "Write a detailed explanation of how computers execute programs."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def run():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                min_new_tokens=args.tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output.shape[1] - inputs.input_ids.shape[1]
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        return generated / elapsed, elapsed, peak_vram

    print("warming up...")
    run()
    results = [run() for _ in range(args.runs)]
    tps = [result[0] for result in results]
    print(f"\nmodel: {args.model}")
    print(f"prompt_tokens: {inputs.input_ids.shape[1]}")
    print(f"generated_tokens: {args.tokens}")
    print(f"runs: {args.runs}")
    print(f"tok/s mean: {statistics.mean(tps):.2f}")
    print(f"tok/s median: {statistics.median(tps):.2f}")
    print(f"tok/s min: {min(tps):.2f}")
    print(f"tok/s max: {max(tps):.2f}")
    print(f"peak_vram_gb: {max(result[2] for result in results):.2f}")


def main_generation() -> None:
    run_generation_benchmark(generation_parser().parse_args())


def main_decode() -> None:
    run_decode_benchmark(decode_parser().parse_args())
