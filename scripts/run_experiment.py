import argparse

from lab_01.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one YAML-defined SST-2 experiment.")
    parser.add_argument("config")
    parser.add_argument("--model")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir", help="Root directory for generated experiment outputs.")
    args = parser.parse_args()
    output_dir = run_experiment(
        args.config,
        {
            "model": args.model,
            "train_size": args.train_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "output_dir": args.output_dir,
        },
    )
    print(f"experiment_output: {output_dir}")


if __name__ == "__main__":
    main()
