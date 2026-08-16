import argparse

from lab_01.train import train_sst2_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM on 500 SST-2 examples with LoRA.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/checkpoints/qwen3-0.6b-sst2-lora")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_sst2_lora(args.model, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
