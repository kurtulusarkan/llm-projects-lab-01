import argparse
from lab_01.evaluate import evaluate_sst2_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on held-out GLUE SST-2.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="Path to a saved LoRA adapter checkpoint.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    metrics, elapsed = evaluate_sst2_adapter(
        model_name=args.model,
        adapter_path=args.adapter,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        show_progress=True,
    )
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"total_evaluation_time_s: {elapsed:.2f}")
    total = metrics["total_examples"]
    print(f"examples_per_second: {total / elapsed:.2f}" if elapsed else "examples_per_second: inf")


if __name__ == "__main__":
    main()
