import argparse

from lab_01.data import load_sst2_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic GLUE SST-2 experiment splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=200)
    args = parser.parse_args()

    splits = load_sst2_experiment(args.train_size, args.validation_size, args.seed)
    for name, split in splits.items():
        print(f"{name}: {len(split)} examples")


if __name__ == "__main__":
    main()
