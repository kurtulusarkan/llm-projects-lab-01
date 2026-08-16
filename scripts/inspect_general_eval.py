import argparse

from lab_01.general_eval_inspection import format_comparison
from lab_01.general_eval_inspection import format_prediction
from lab_01.general_eval_inspection import load_evaluation_artifacts
from lab_01.general_eval_inspection import select_predictions
from lab_01.general_eval_inspection import summary


def print_summary(name: str, values: dict) -> None:
    print(f"=== {name} Summary ===")
    print(f"total examples: {values['total_examples']}")
    print(f"passed: {values['passed']}")
    print(f"failed: {values['failed']}")
    if values["invalid_count_applies_to_selection"]:
        print(f"invalid outputs: {values['invalid_outputs']}")
    else:
        print(f"invalid outputs: unavailable for selection (run total: {values['invalid_outputs']})")
    print(f"selected examples: {values['selected_examples']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved general-evaluation artifacts without rescoring.")
    parser.add_argument("--eval-dir", required=True, help="Evaluation artifact directory to inspect.")
    parser.add_argument("--compare", help="Second evaluation directory; the first is shown as BASE.")
    parser.add_argument("--category", default=None)
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--id", dest="item_id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base = load_evaluation_artifacts(args.eval_dir)
    selected = select_predictions(
        base,
        category=args.category,
        failed_only=args.failed_only,
        item_id=args.item_id,
        limit=args.limit,
    )
    print_summary("Evaluation", summary(base, selected))

    if args.compare:
        adapter = load_evaluation_artifacts(args.compare)
        print()
        print_summary("Comparison run", summary(adapter, adapter.predictions))
        pairs = format_comparison(base, adapter, selected)
        for pair in pairs:
            print()
            print(pair)
        if not pairs:
            print("\nNo matching IDs selected.")
        return

    for prediction in selected:
        print()
        print(format_prediction(base, prediction))


if __name__ == "__main__":
    main()
