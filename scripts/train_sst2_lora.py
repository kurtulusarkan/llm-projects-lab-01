import argparse

from lab_01.train import train_sst2_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM on an SST-2 subset with LoRA.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/checkpoints/qwen3-0.6b-sst2-lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=500)
    args = parser.parse_args()
    train_sst2_lora(args.model, args.output_dir, args.seed, args.train_size)


if __name__ == "__main__":
    main()
