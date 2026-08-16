import argparse

from lab_01.general_eval import CATEGORIES
from lab_01.general_eval import DEFAULT_BATCH_SIZE
from lab_01.general_eval import DEFAULT_DATASET
from lab_01.general_eval import DEFAULT_MAX_NEW_TOKENS
from lab_01.general_eval import DEFAULT_MODEL
from lab_01.general_eval import evaluate_general


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen general-capability regression suite.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", help="Path to a saved LoRA adapter.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", choices=sorted(CATEGORIES), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    output_directory, metrics = evaluate_general(
        model_name=args.model,
        adapter_path=args.adapter,
        dataset_path=args.dataset,
        limit=args.limit,
        category=args.category,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        show_progress=True,
    )
    print("=== General Evaluation Complete ===")
    print(f"overall_accuracy: {metrics['overall_accuracy']}")
    print(f"invalid_output_count: {metrics['invalid_output_count']}")
    print(f"total_examples: {metrics['total_examples']}")
    print(f"output_directory: {output_directory}")


if __name__ == "__main__":
    main()
