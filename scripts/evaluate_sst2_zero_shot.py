import argparse
import time

from lab_01.data import load_sst2_experiment
from lab_01.evaluate import evaluate_sst2_zero_shot
from lab_01.models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on held-out GLUE SST-2.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="Path to a saved LoRA adapter checkpoint.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset = load_sst2_experiment()["test"]
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    model, tokenizer = load_model(args.model)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    started = time.perf_counter()
    metrics = evaluate_sst2_zero_shot(
        model,
        tokenizer,
        args.model,
        dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        show_progress=True,
    )
    elapsed = time.perf_counter() - started
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"total_evaluation_time_s: {elapsed:.2f}")
    print(f"examples_per_second: {len(dataset) / elapsed:.2f}" if elapsed else "examples_per_second: inf")


if __name__ == "__main__":
    main()
