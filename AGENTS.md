# AGENTS.md

## Project purpose

This repository is a small-LLM fine-tuning laboratory. Its purpose is to learn
and compare dataset preparation, evaluation, LoRA fine-tuning, and disciplined
experiment design for causal language models.

## Development workflow

- Read `README.md` and `EXPERIMENTS.md` before making LLM-related changes.
- Keep the project simple and educational; do not introduce framework-like
  abstractions, distributed training, or tracking infrastructure without a
  demonstrated need.
- Keep reusable logic in `src/lab_01/` and command-line entry points thin under
  `scripts/`.
- Preserve existing benchmark behavior unless a requested experiment explicitly
  changes it.
- Add deterministic, CPU-only tests for new deterministic behavior. Do not
  require a GPU, model download, or network access for unit tests.
- Use `uv` for dependencies. Keep the existing PyTorch CUDA package source
  unchanged unless the task explicitly requires a CUDA change.
- Update `README.md` when commands, workflow, dependencies, or operational
  guidance change.

## Experiment discipline

- Prefer controlled experiments: change one meaningful variable at a time.
- Preserve dataset splits, random seeds, evaluation prompts, output parsing,
  and metrics unless they are the variable being tested.
- Do not optimize based only on assumptions; measure baseline and changed runs
  under comparable conditions.
- Record important measured findings, configuration changes, and conclusions in
  `EXPERIMENTS.md`. Do not store experiment results in this file.
- State expected impact and trade-offs before making a material experiment
  change.
- Avoid unrelated refactors while implementing an experiment.

## Current environment

Hardware:

- NVIDIA RTX 5070 Ti with 16 GB VRAM
- CUDA is available for model inference and training

Software:

- Python 3.12
- `uv` package manager and lockfile
- PyTorch CUDA 12.8 build
- Hugging Face Transformers, Datasets, Accelerate, PEFT, TRL, and bitsandbytes

## Current model workflow

Primary workflow:

- Qwen/Qwen3-0.6B causal language model
- GLUE SST-2 sentiment classification
- Strict JSON classification outputs
- Hugging Face Transformers for loading/generation
- PEFT LoRA and TRL SFT for fine-tuning

Other model families in scope are SmolLM and Liquid AI models. Maintain their
benchmark compatibility when changing shared loading or chat-template code.

## Before making changes

Read, in order:

1. `README.md`
2. `EXPERIMENTS.md`
3. Relevant source files and tests

When proposing or implementing a change:

- explain the expected impact;
- identify what remains controlled;
- preserve reproducibility; and
- report validation performed without running expensive GPU work unless asked.
