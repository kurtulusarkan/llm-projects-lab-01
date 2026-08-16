# Experimental findings

This file records measured results for the SST-2 LoRA lab. Future pipeline
changes should be based on controlled comparisons rather than assumptions.

## Hardware

- GPU: NVIDIA RTX 5070 Ti
- VRAM: 16 GB
- CUDA: working correctly

## Project and dataset

Qwen3-0.6B performs SST-2 sentiment classification through LoRA SFT.

- Dataset: GLUE SST-2
- Baseline training subset: 500 examples
- Held-out evaluation: official SST-2 validation split, 872 examples

## Initial baseline

Zero-shot Qwen3-0.6B measured:

- Accuracy: 0.3509
- Invalid JSON count: 517 of 872

The main failure was compliance with the strict JSON output contract, rather
than sentiment understanding.

## LoRA experiment

Configuration:

- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- Epochs: 3
- Learning rate: 2e-4

With 500 examples, evaluation accuracy was approximately 90% and the invalid
JSON count was zero. The main benefit was teaching the model to follow the JSON
output contract.

## Evaluation optimization

Changing evaluation from single-example generation to batched generation kept
metrics identical while increasing throughput:

| Batch size | Throughput |
| --- | --- |
| 1 | ~1.7 examples/sec |
| 32 | ~35 examples/sec |

This is approximately a 20x throughput improvement.

## Training optimization experiments

| Configuration | Runtime | Result |
| --- | ---: | --- |
| Batch size 1, accumulation 8, gradient checkpointing enabled | 761 s | Baseline |
| Batch size 8, accumulation 1, gradient checkpointing enabled | 112.8 s | ~6.7x faster; same evaluation quality |
| Batch size 8, accumulation 1, gradient checkpointing disabled | 71.6 s | ~10.6x faster; ~5.7 GB VRAM; 89–90% accuracy |

## Preferred configuration

For Qwen3-0.6B LoRA training on this RTX 5070 Ti, use:

```text
per_device_train_batch_size=8
gradient_accumulation_steps=1
gradient_checkpointing=False
```

The GPU has sufficient headroom for this model. The prior bottleneck was
inefficient batching and gradient checkpointing, not VRAM capacity.

## Full-data SST-2 scale experiment

A completed full-data LoRA run used the maximum deterministic training subset:

- Base model: `Qwen/Qwen3-0.6B`
- Training examples: 67,149
- Reserved lab validation examples: 200 from the official SST-2 training split
- Seed: 42
- Epochs: 1
- Batch size: 32
- Gradient accumulation: 1
- Gradient checkpointing: disabled
- LoRA: rank 8, alpha 16, dropout 0.05, with the standard Qwen target modules
- Learning rate: 2e-4
- Optimizer updates: 2,099

Measured training runtime was 941.1 seconds (71.35 examples/second). Held-out
official SST-2 validation performance was 93.81% accuracy, with 0 invalid JSON
outputs out of 872 examples.

## General capability regression check

The frozen `evals/general/v1` suite was run as a paired base-versus-adapter
check for the completed 67,149-example adapter above. Exact deterministic
accuracy increased from 15/115 (13.0%) for the base model to 24/115 (20.9%)
for the adapter. Most exact gains were short-answer and output-format
compliance improvements.

A manual review of the paired stored outputs found likely regressions in
reasoning and language behavior despite the exact-score increase. This is an
important qualitative regression signal, not a new reproducible semantic
metric; retain the frozen exact evaluation and inspect paired outputs before
claiming a general-capability improvement.

## Next experiments

1. Compare the completed 67,149-example one-epoch run with controlled
   full-data variants; measure training time, held-out accuracy, and paired
   general-capability regression behavior.
2. Compare LoRA capacity at ranks 4, 8, and 16.
3. Compare Qwen3-0.6B, SmolLM2-360M, and LiquidAI LFM2.5-1.2B.

Preserve controlled experiments whenever possible. Prioritize data scaling and
model behavior before further performance optimization.
