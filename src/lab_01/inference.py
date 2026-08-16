"""Generation helpers used by benchmarks and interactive experiments."""

import time

import torch

from lab_01.models import format_chat_prompt


def generate_once(model, tokenizer, model_name: str, prompt: str, max_new_tokens: int):
    """Run one greedy chat generation and return text plus performance metrics."""
    text = format_chat_prompt(tokenizer, model_name, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    generated = output[0][inputs.input_ids.shape[1] :]
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "tokens": generated.numel(),
        "seconds": elapsed,
        "tps": generated.numel() / elapsed,
        "vram": torch.cuda.max_memory_allocated() / 1024**3,
    }


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Greedily generate one continuation for every already-formatted prompt."""
    if not prompts:
        return []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    finally:
        tokenizer.padding_side = original_padding_side

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[:, inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)
