"""Shared model loading and prompt formatting for the lab."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_family(model_name: str) -> str:
    """Return the supported model family inferred from a model identifier."""
    name = model_name.lower()
    if "qwen" in name:
        return "qwen"
    if "smollm" in name:
        return "smollm"
    if "liquid" in name or "lfm" in name:
        return "liquid"
    return "other"


def format_messages(tokenizer, model_name: str, messages: list[dict], add_generation_prompt: bool) -> str:
    """Format chat messages using the model's bundled template."""
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if model_family(model_name) == "qwen":
        # Preserve the existing benchmark behavior for Qwen 3 chat templates.
        kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(messages, **kwargs)


def format_chat_prompt(tokenizer, model_name: str, prompt: str) -> str:
    """Format a single user turn using the model's bundled chat template."""
    messages = [{"role": "user", "content": prompt}]
    return format_messages(tokenizer, model_name, messages, add_generation_prompt=True)


def load_model(model_name: str):
    """Load a model exactly as the original CUDA benchmarks did."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    return model, tokenizer
