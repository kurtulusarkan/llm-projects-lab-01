import unittest
from tempfile import TemporaryDirectory

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from lab_01.train import QWEN3_LORA_TARGET_MODULES, format_sst2_training_example, lora_config


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return repr((messages, kwargs))


class TrainTest(unittest.TestCase):
    def test_lora_configuration_matches_experiment_defaults(self):
        config = lora_config()
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertEqual(config.target_modules, set(QWEN3_LORA_TARGET_MODULES))

    def test_training_example_uses_json_assistant_completion(self):
        text = format_sst2_training_example(
            FakeTokenizer(),
            "Qwen/Qwen3-0.6B",
            {"sentence": "A delightful film.", "target": '{"sentiment":"positive"}'},
        )
        self.assertIn('{"sentiment":"positive"}', text)
        self.assertIn("Classify the sentiment", text)
        self.assertIn("'enable_thinking': False", text)

    def test_peft_adapter_save_and_load_round_trip(self):
        config = GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=32)
        base_model = GPT2LMHeadModel(config)
        adapter_model = get_peft_model(
            base_model,
            LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, target_modules=["c_attn"]),
        )
        with TemporaryDirectory() as directory:
            adapter_model.save_pretrained(directory)
            loaded = PeftModel.from_pretrained(GPT2LMHeadModel(config), directory)
        self.assertIn("default", loaded.peft_config)
