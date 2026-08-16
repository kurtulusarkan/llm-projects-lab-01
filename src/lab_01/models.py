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


def format_chat_prompt(tokenizer, model_name: str, prompt: str) -> str:
    """Format a single user turn using the model's bundled chat template."""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_family(model_name) == "qwen":
        # Preserve the existing benchmark behavior for Qwen 3 chat templates.
        kwargs["enable_thinking"] = False

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(model_name: str):
    """Load a model exactly as the original CUDA benchmarks did."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    return model, tokenizer
