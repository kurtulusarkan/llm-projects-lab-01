"""Small helpers for loading and checking chat fine-tuning datasets."""

from datasets import load_dataset


def load_split(dataset_name: str, split: str = "train", **kwargs):
    """Load one dataset split through Hugging Face Datasets."""
    return load_dataset(dataset_name, split=split, **kwargs)


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
