"""Dataset helpers for the SST-2 fine-tuning experiment."""

import json

from datasets import DatasetDict, load_dataset


SST2_TRAIN_SIZE = 500
SST2_VALIDATION_SIZE = 200
SST2_LABELS = {0: "negative", 1: "positive"}


def load_split(dataset_name: str, split: str = "train", **kwargs):
    """Load one dataset split through Hugging Face Datasets."""
    return load_dataset(dataset_name, split=split, **kwargs)


def sst2_target(label: int) -> str:
    """Return the exact JSON target used for SST-2 classification."""
    try:
        sentiment = SST2_LABELS[label]
    except KeyError as error:
        raise ValueError(f"Unknown SST-2 label: {label}") from error
    return json.dumps({"sentiment": sentiment}, separators=(",", ":"))


def format_sst2_example(example: dict) -> dict:
    """Add the strict JSON completion target while retaining the original fields."""
    return {**example, "target": sst2_target(example["label"])}


def make_sst2_experiment_splits(
    train_dataset,
    held_out_dataset,
    train_size: int = SST2_TRAIN_SIZE,
    validation_size: int = SST2_VALIDATION_SIZE,
    seed: int = 42,
) -> DatasetDict:
    """Create reproducible train/validation splits without touching held-out data.

    The 500-example train and 200-example validation subsets both come from the
    official GLUE training split. The official SST-2 validation split is exposed
    as the ``test`` split for final held-out evaluation and is never sampled for
    training or validation.
    """
    requested = train_size + validation_size
    if requested > len(train_dataset):
        raise ValueError(
            f"Requested {requested} examples, but SST-2 train has only {len(train_dataset)}."
        )

    shuffled = train_dataset.shuffle(seed=seed)
    train = shuffled.select(range(train_size))
    validation = shuffled.select(range(train_size, requested))
    return DatasetDict(
        {
            "train": train.map(format_sst2_example),
            "validation": validation.map(format_sst2_example),
            "test": held_out_dataset.map(format_sst2_example),
        }
    )


def load_sst2_experiment(
    train_size: int = SST2_TRAIN_SIZE,
    validation_size: int = SST2_VALIDATION_SIZE,
    seed: int = 42,
) -> DatasetDict:
    """Load GLUE SST-2 and create the lab's train/validation/test-style splits."""
    dataset = load_dataset("glue", "sst2")
    return make_sst2_experiment_splits(
        dataset["train"],
        dataset["validation"],
        train_size=train_size,
        validation_size=validation_size,
        seed=seed,
    )


def validate_messages(example: dict) -> dict:
    """Check the simple chat schema used by the lab's SFT experiments."""
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each example must contain a non-empty 'messages' list.")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dictionary.")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Message roles must be system, user, or assistant.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Each message needs non-empty string content.")
    return example


def prepare_chat_dataset(dataset):
    """Validate a Dataset with a `messages` column before tokenization."""
    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must have a 'messages' column.")
    return dataset.map(validate_messages)
