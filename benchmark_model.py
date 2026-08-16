import argparse, statistics, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_once(model, tokenizer, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)

    if "qwen" in model.name_or_path.lower():
        kwargs["enable_thinking"] = False

    text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    generated = output[0][inputs.input_ids.shape[1]:]

    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "tokens": generated.numel(),
        "seconds": elapsed,
        "tps": generated.numel() / elapsed,
        "vram": torch.cuda.max_memory_allocated() / 1024**3,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="Explain in two sentences why the sky is blue.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    run_once(model, tokenizer, args.prompt, args.max_new_tokens)

    results = [
        run_once(model, tokenizer, args.prompt, args.max_new_tokens)
        for _ in range(args.runs)
    ]

    tps = [r["tps"] for r in results]

    print("\n--- RESULT ---")
    print(results[-1]["text"])

    print("\n--- METRICS ---")
    print(f"model: {args.model}")
    print(f"load_time_s: {load_s:.2f}")
    print(f"runs: {args.runs}")
    print(f"generated_tokens: {results[-1]['tokens']}")
    print(f"tokens_per_second_mean: {statistics.mean(tps):.2f}")
    print(f"tokens_per_second_median: {statistics.median(tps):.2f}")
    print(f"tokens_per_second_min: {min(tps):.2f}")
    print(f"tokens_per_second_max: {max(tps):.2f}")
    print(f"peak_vram_gb: {max(r['vram'] for r in results):.2f}")

if __name__ == "__main__":
    main()
