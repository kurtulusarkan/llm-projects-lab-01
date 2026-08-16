# Small-LLM experimentation lab

This repository is a compact workspace for comparing and fine-tuning small
language models, starting with Qwen, SmolLM, and Liquid AI models. It uses
`uv`, PyTorch CUDA, and Hugging Face Transformers. Fine-tuning support will
later add PEFT/TRL without changing the inference workflow.

## Setup

Install the locked environment:

```bash
uv sync
```

If uv's shared cache is not writable in a restricted environment, set a
temporary cache for the command:

```bash
UV_CACHE_DIR=/tmp/lab01-uv-cache uv sync
```

Verify CUDA from the target machine before running a benchmark:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Benchmarks

The generation benchmark measures end-to-end chat generation. It honors each
model's chat template and disables Qwen thinking mode to keep results
comparable with the original lab benchmark.

```bash
uv run python scripts/benchmark_model.py --model Qwen/Qwen3-0.6B
uv run python scripts/benchmark_model.py --model HuggingFaceTB/SmolLM2-1.7B
uv run python scripts/benchmark_model.py --model LiquidAI/LFM2-1.2B
```

The decode benchmark uses eager execution and fixed-length generation.
`torch.compile` is intentionally not used because it showed no meaningful
benefit in this lab.

```bash
uv run python scripts/benchmark_decode.py --model Qwen/Qwen3-0.6B --tokens 256 --runs 5
```

## Layout

- `src/lab_01/models.py`: model loading and model-family chat prompting.
- `src/lab_01/inference.py`: one-pass generation and metrics.
- `src/lab_01/benchmark.py`: generation and decode benchmark logic.
- `src/lab_01/data.py`: Hugging Face dataset loading and chat-schema checks.
- `src/lab_01/evaluate.py`: simple repeatable evaluation metrics.
- `src/lab_01/train.py`: placeholders for future SFT, LoRA, and QLoRA work.
- `scripts/`: thin executable benchmark wrappers.
- `tests/`: small, CPU-only unit tests.

Run the non-GPU test suite with:

```bash
uv run python -m unittest discover -s tests -v
```
