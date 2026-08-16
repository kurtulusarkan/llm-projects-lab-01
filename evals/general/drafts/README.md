# General capability evaluation drafts

`dataset.jsonl` contains **draft candidates**, not a frozen evaluation suite.
It is a review pool for a future `evals/general/v1/` dataset and may be edited,
removed, or replaced before that release.

## Important boundaries

- These examples are evaluation-only and must never be used for training,
  validation selection, prompt tuning, or synthetic-data generation.
- Do not treat scores on this file as stable regression results.
- A reviewed, immutable copy must be created under `evals/general/v1/` before
  any adapter-versus-base comparison is reported.

## Schema

Each JSONL record has:

- `id`: stable candidate identifier.
- `category`: `instruction_following`, `reasoning`, `knowledge`,
  `coding`, or `language`.
- `prompt`: user prompt for the model.
- `expected`: a canonical answer, an `accepted` answer list, or code
  `validation` criteria.
- `evaluation_type`: `strict_exact`, `short_answer`,
  `multiple_choice`, or `code`.
- `difficulty`: intended `easy`, `medium`, or `hard` difficulty for
  0.6B–1.7B models. These labels are provisional, not empirically calibrated.

## Scoring intent

- `strict_exact`: formatting is part of the behavior under test. The output
  should match after only trivial transport normalization, such as line endings.
- `short_answer`: a short semantic answer. Its `expected` value is either one
  canonical string or `{\"accepted\": [...]}` when multiple reviewed forms are
  valid. A future evaluator may apply minimal normalization.
- `multiple_choice`: one explicitly offered answer token, such as `A`, `B`,
  `C`, `yes`, or `no`.
- `code`: `expected` is `{\"validation\": [...]}`. A future safe validator must
  check those criteria; canonical source-text equality is not sufficient.

The current candidate mix is 115 examples: 25 instruction-following, 32
reasoning, 22 knowledge, 21 coding, and 15 language. Ten reasoning examples
explicitly probe uncertainty, entailment, possible-versus-guaranteed
conclusions, and unsupported assumptions. The instruction category retains
some strict-format probes while covering extraction, filtering, ordering,
exclusion, preservation of input order, and explicit constraint priority.

## Review before freezing

Review every candidate for an unambiguous answer, stable facts, duplicate or
near-duplicate prompts, model-appropriate difficulty, and leakage into any
training material. Review accepted answer variants and implement safe code
validation before promotion. A future frozen version should retain its JSONL
contents and documentation unchanged; corrections belong in a new version.
