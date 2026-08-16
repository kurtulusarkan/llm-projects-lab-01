import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--runs", type=int, default=5)
args = p.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    device_map="cuda",
)

model = torch.compile(model, mode="reduce-overhead")

prompt = "Write a detailed explanation of how computers execute programs."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

def run():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            min_new_tokens=args.tokens,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated = out.shape[1] - inputs.input_ids.shape[1]
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    return generated / elapsed, elapsed, peak_vram

print("warming up / compiling...")
run()

results = [run() for _ in range(args.runs)]
tps = [r[0] for r in results]

print(f"\nmodel: {args.model}")
print(f"prompt_tokens: {inputs.input_ids.shape[1]}")
print(f"generated_tokens: {args.tokens}")
print(f"runs: {args.runs}")
print(f"tok/s mean: {statistics.mean(tps):.2f}")
print(f"tok/s median: {statistics.median(tps):.2f}")
print(f"tok/s min: {min(tps):.2f}")
print(f"tok/s max: {max(tps):.2f}")
print(f"peak_vram_gb: {max(r[2] for r in results):.2f}")
