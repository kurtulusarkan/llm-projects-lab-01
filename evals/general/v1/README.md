# General capability regression suite v1

This frozen dataset is a small paired regression suite for comparing a base
instruction model with the same model plus a LoRA adapter. It is intended to
detect catastrophic forgetting, capability degradation, unintended
specialization, and output-format regressions. It is not a leaderboard
benchmark and must never be used for training or model selection.

## Version metadata

- Version: `v1`
- Created: `2026-08-16`
- Items: `115`
- SHA256: `dfebee2df92e034311b63f058248d08cdd3739b3fa08808e91f6cb1bc951a0fe`

The checksum is calculated from the raw bytes of `dataset.jsonl`. The frozen
file must not be edited. Any content change requires a new version directory
and checksum.

## Balance

| Category | Items |
| --- | ---: |
| instruction_following | 25 |
| reasoning | 32 |
| knowledge | 22 |
| coding | 21 |
| language | 15 |

Evaluation types are `strict_exact`, `short_answer`, `multiple_choice`, and
`code`. For `short_answer`, `expected` is either a canonical string or an
object containing an `accepted` list. For `code`, `expected` contains a
`validation` list describing safe semantic checks; generated source must not
be compared to one canonical implementation.

Difficulty labels are intended for 0.6B–1.7B instruction models and were not
empirically calibrated before this release.

## Evaluation discipline

Use identical prompts, system prompt, chat-template behavior, and deterministic
generation settings for paired base and adapter runs. Preserve per-item outputs
so regressions can be inspected rather than treating the aggregate score as a
general intelligence measure.
