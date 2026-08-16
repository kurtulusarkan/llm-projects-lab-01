"""Evaluation helpers for strict JSON SST-2 classification."""

import json

import torch
from tqdm.auto import tqdm

from lab_01.data import SST2_LABELS
from lab_01.inference import generate_batch
from lab_01.models import format_chat_prompt


def exact_match(predictions: list[str], references: list[str]) -> float:
    """Return normalized exact-match accuracy."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not references:
        return 0.0
    matches = sum(
        prediction.strip() == reference.strip()
        for prediction, reference in zip(predictions, references, strict=True)
    )
    return matches / len(references)


def sst2_prompt(sentence: str) -> str:
    """Build the single-turn prompt used for zero-shot SST-2 evaluation."""
    return (
        "Classify the sentiment of the following sentence. "
        'Return only one JSON object in exactly this form: {"sentiment":"positive"} '
        'or {"sentiment":"negative"}.\n\n'
        f"Sentence: {json.dumps(sentence)}"
    )


def parse_sst2_prediction(text: str) -> tuple[str | None, str | None]:
    """Parse one strict SST-2 response as ``(sentiment, error_kind)``.

    ``invalid_json`` denotes a response that is not one complete JSON value.
    ``invalid_label`` denotes valid JSON that is not exactly one supported
    ``sentiment`` label.
    """
    try:
        result = json.loads(text.strip())
    except json.JSONDecodeError:
        return None, "invalid_json"

    if (
        not isinstance(result, dict)
        or set(result) != {"sentiment"}
        or not isinstance(result["sentiment"], str)
        or result["sentiment"] not in SST2_LABELS.values()
    ):
        return None, "invalid_label"
    return result["sentiment"], None


def summarize_sst2_predictions(predictions: list[str], labels: list[int]) -> dict[str, int | float]:
    """Report accuracy and invalid-response counts over all examples."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    correct = 0
    invalid_json = 0
    invalid_label = 0
    for prediction, label in zip(predictions, labels, strict=True):
        sentiment, error_kind = parse_sst2_prediction(prediction)
        if error_kind == "invalid_json":
            invalid_json += 1
        elif error_kind == "invalid_label":
            invalid_label += 1
        elif sentiment == SST2_LABELS[label]:
            correct += 1

    total = len(labels)
    return {
        "accuracy": correct / total if total else 0.0,
        "invalid_json_count": invalid_json,
        "invalid_label_count": invalid_label,
        "total_examples": total,
    }


def evaluate_sst2_zero_shot(
    model,
    tokenizer,
    model_name: str,
    dataset,
    max_new_tokens: int = 32,
    batch_size: int = 32,
    show_progress: bool = False,
):
    """Generate and score batched zero-shot SST-2 predictions."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    predictions = []
    labels = []
    total = len(dataset)
    progress = tqdm(total=total, desc="Evaluating SST-2", unit="examples") if show_progress else None
    try:
        for start in range(0, total, batch_size):
            examples = [dataset[index] for index in range(start, min(start + batch_size, total))]
            prompts = [
                format_chat_prompt(tokenizer, model_name, sst2_prompt(example["sentence"]))
                for example in examples
            ]
            predictions.extend(generate_batch(model, tokenizer, prompts, max_new_tokens))
            labels.extend(example["label"] for example in examples)
            if progress:
                progress.update(len(examples))
                progress.set_postfix(batches=(start // batch_size) + 1)
    finally:
        if progress:
            progress.close()
    return summarize_sst2_predictions(predictions, labels)
