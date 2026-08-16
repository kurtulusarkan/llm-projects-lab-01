import unittest

import torch

from lab_01.inference import generate_batch


class FakeBatch(dict):
    def __init__(self, batch_size: int):
        input_ids = torch.zeros((batch_size, 3), dtype=torch.long)
        super().__init__(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        self.input_ids = input_ids
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = 151643
        self.calls = []
        self.last_batch = None

    def __call__(self, prompts, *, padding, return_tensors):
        self.calls.append((prompts, padding, return_tensors, self.padding_side))
        self.last_batch = FakeBatch(len(prompts))
        return self.last_batch

    def batch_decode(self, generated, *, skip_special_tokens):
        self.generated = generated.tolist()
        self.skip_special_tokens = skip_special_tokens
        return [f"response-{row[0]}" for row in self.generated]


class FakeModel:
    device = "cuda"

    def __init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        continuations = torch.arange(10, 10 + batch_size).unsqueeze(1)
        return torch.cat((kwargs["input_ids"], continuations), dim=1)


class BatchInferenceTest(unittest.TestCase):
    def test_batches_prompts_and_decodes_each_continuation(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()

        generated = generate_batch(model, tokenizer, ["first", "second"], max_new_tokens=12)

        self.assertEqual(generated, ["response-10", "response-11"])
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 12)
        self.assertFalse(model.generate_calls[0]["do_sample"])
        self.assertEqual(model.generate_calls[0]["pad_token_id"], 151643)
        self.assertEqual(tokenizer.calls, [(["first", "second"], True, "pt", "left")])
        self.assertEqual(tokenizer.last_batch.moved_to, "cuda")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_empty_prompt_batch_skips_model_generation(self):
        model = FakeModel()
        self.assertEqual(generate_batch(model, FakeTokenizer(), [], max_new_tokens=12), [])
        self.assertEqual(model.generate_calls, [])

    def test_single_prompt_batch_uses_one_greedy_generation(self):
        model = FakeModel()

        generated = generate_batch(model, FakeTokenizer(), ["only prompt"], max_new_tokens=12)

        self.assertEqual(generated, ["response-10"])
        self.assertEqual(len(model.generate_calls), 1)
