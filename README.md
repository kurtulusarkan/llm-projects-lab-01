# Small-LLM experimentation lab

This repository is a compact workspace for comparing and fine-tuning small
language models, starting with Qwen, SmolLM, and Liquid AI models. It uses
`uv`, PyTorch CUDA, Hugging Face Transformers, PEFT, and TRL.

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

## SST-2 experiment

The SST-2 experiment takes 500 deterministic examples for training and 200
for validation from GLUE's official training split (seed `42`). The official
GLUE SST-2 validation split remains untouched and is used as the final held-out
`test` split. Labels are JSON completions only: `{"sentiment":"positive"}` or
`{"sentiment":"negative"}`.

```bash
uv run python scripts/prepare_sst2.py
uv run python scripts/evaluate_sst2_zero_shot.py --model Qwen/Qwen3-0.6B
```

To smoke-test the zero-shot evaluator on part of the held-out set:

```bash
uv run python scripts/evaluate_sst2_zero_shot.py --max-examples 20
```

Evaluation uses batched GPU generation by default (`--batch-size 32`) while
preserving the same prompt, JSON parsing, and metrics as single-example
generation. It displays tqdm progress with examples and batches completed, then
reports accuracy, invalid JSON and label counts, total examples, total
evaluation time, and examples per second.

Use batch size 1 for a direct single-example throughput comparison, or reduce
the batch size if evaluation memory is insufficient:

```bash
uv run python scripts/evaluate_sst2_zero_shot.py --batch-size 1
```

## SST-2 LoRA fine-tuning

Train LoRA adapters for Qwen3-0.6B on a deterministic SST-2 `train` subset
(500 examples by default). The official SST-2 validation split remains held out. This uses
BF16 CUDA with batch size 8 and no gradient accumulation; adapters and
adapter-only checkpoints are written below `outputs/checkpoints/`. This is
tuned for GPUs with enough VRAM (such as 16 GB). If memory is insufficient,
reduce the batch size and increase gradient accumulation to keep the effective
batch size at 8.

Measured baselines, optimization results, and the preferred configuration are
recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

```bash
uv run python scripts/train_sst2_lora.py
```

Change only the deterministic training-subset size for a controlled data-scale
experiment:

```bash
uv run python scripts/train_sst2_lora.py --train-size 5000
```

## YAML experiments

Run a reproducible train-and-evaluate experiment from YAML. The runner writes
the original configuration, resolved overrides, metadata, training log, adapter,
and evaluation results to a unique directory below `outputs/experiments/`.
Directory names include the experiment name, model, training-set size, and a
stable configuration fingerprint; repeated identical runs receive a numeric
suffix rather than overwriting an existing adapter.
Metadata includes derived optimizer-step counts and effective batch size so
batch-size experiments can be compared by updates rather than epochs alone.

```bash
uv run python scripts/run_experiment.py experiments/sst2/baseline_500.yaml
uv run python scripts/run_experiment.py experiments/sst2/baseline_500.yaml --train-size 5000
```

The largest deterministic training subset is 67,149 examples: the remaining
200 examples from GLUE's official training split are reserved for the lab
validation subset. The completed one-epoch full-data run used batch size 32:

```bash
uv run python scripts/run_experiment.py \
  experiments/sst2/baseline_500.yaml \
  --train-size 67149 \
  --epochs 1 \
  --batch-size 32
```

Measured results and general-capability regression observations are recorded
in [EXPERIMENTS.md](EXPERIMENTS.md).

Compare completed experiments by held-out accuracy, update counts, and training
throughput. The comparison is read-only: it includes completed, incomplete, and
failed artifact directories without changing them. Optionally pass an experiment
directory or name prefix, or write a CSV file.

```bash
uv run python scripts/compare_experiments.py
uv run python scripts/compare_experiments.py sst2-qwen3-500 --csv outputs/comparison.csv
```

Evaluate a saved adapter on the held-out test split without merging it:

```bash
uv run python scripts/evaluate_sst2_zero_shot.py \
  --adapter outputs/checkpoints/qwen3-0.6b-sst2-lora
```

## General capability regression evaluation

The frozen `evals/general/v1/dataset.jsonl` suite contains 115 deterministic,
evaluation-only examples. It is designed for paired base-versus-LoRA regression
testing, not model ranking. Never use these examples for training or adapter
selection.

Evaluate the base model:

```bash
uv run python scripts/evaluate_general.py \
  --model Qwen/Qwen3-0.6B \
  --dataset evals/general/v1/dataset.jsonl
```

Evaluate the same model with an adapter:

```bash
uv run python scripts/evaluate_general.py \
  --model Qwen/Qwen3-0.6B \
  --adapter outputs/experiments/example/adapter \
  --dataset evals/general/v1/dataset.jsonl
```

Use `--limit` for a smoke test or `--category` for one capability area:

```bash
uv run python scripts/evaluate_general.py --category reasoning --limit 10
```

Each run creates a non-overwriting directory under
`outputs/evals/general/v1/` containing `metadata.json`, `predictions.jsonl`,
and `metrics.json`. Generation is greedy and records the dataset checksum,
system prompt, chat template, and generation settings. Code responses are
checked with narrow per-task validation in an isolated, resource-limited
Python subprocess; generated code is never matched against canonical source.

Inspect saved results without rescoring or generating new outputs:

```bash
uv run python scripts/inspect_general_eval.py \
  --eval-dir outputs/evals/general/v1/qwen3-0.6b-base \
  --failed-only \
  --category reasoning
```

Compare matching saved IDs from a base and adapter run:

```bash
uv run python scripts/inspect_general_eval.py \
  --eval-dir outputs/evals/general/v1/qwen3-0.6b-base \
  --compare outputs/evals/general/v1/qwen3-0.6b-sst2-lora \
  --failed-only
```

The inspector displays per-item failure reasons only when an artifact already
stores one. Existing artifacts record aggregate invalid-output counts but do
not provide per-item invalid reasons, so the tool reports `not recorded`
rather than reinterpreting a model response.

## Layout

- `src/lab_01/models.py`: model loading and model-family chat prompting.
- `src/lab_01/inference.py`: single-prompt benchmark and batched generation helpers.
- `src/lab_01/benchmark.py`: generation and decode benchmark logic.
- `src/lab_01/data.py`: SST-2 split creation, JSON targets, and dataset checks.
- `src/lab_01/evaluate.py`: zero-shot SST-2 prompting, JSON parsing, and metrics.
- `src/lab_01/general_eval.py`: frozen general-suite validation, scoring, safe code checks, and artifacts.
- `src/lab_01/general_eval_inspection.py`: read-only saved-artifact inspection and comparison.
- `src/lab_01/train.py`: LoRA SFT training for configurable SST-2 subsets.
- `src/lab_01/experiments.py`: YAML experiment resolution and orchestration.
- `experiments/`: versioned YAML experiment configurations.
- `evals/`: draft and frozen general-capability regression datasets.
- `scripts/`: thin executable benchmark wrappers.
- `tests/`: small, CPU-only unit tests.

Run the non-GPU test suite with:

```bash
uv run python -m unittest discover -s tests -v
```
