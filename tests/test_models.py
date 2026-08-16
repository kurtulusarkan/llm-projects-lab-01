import unittest

from lab_01.models import format_chat_prompt, format_messages, model_family


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return repr((messages, kwargs))


class ModelsTest(unittest.TestCase):
    def test_detects_supported_families(self):
        self.assertEqual(model_family("Qwen/Qwen3-0.6B"), "qwen")
        self.assertEqual(model_family("HuggingFaceTB/SmolLM2-1.7B"), "smollm")
        self.assertEqual(model_family("LiquidAI/LFM2-1.2B"), "liquid")

    def test_qwen_disables_thinking(self):
        prompt = format_chat_prompt(FakeTokenizer(), "Qwen/Qwen3-0.6B", "hello")
        self.assertIn("'enable_thinking': False", prompt)

    def test_other_models_use_their_template_without_qwen_option(self):
        prompt = format_chat_prompt(FakeTokenizer(), "HuggingFaceTB/SmolLM2-1.7B", "hello")
        self.assertNotIn("enable_thinking", prompt)

    def test_training_messages_disable_qwen_thinking(self):
        prompt = format_messages(
            FakeTokenizer(),
            "Qwen/Qwen3-0.6B",
            [{"role": "assistant", "content": "answer"}],
            add_generation_prompt=False,
        )
        self.assertIn("'enable_thinking': False", prompt)
        self.assertIn("'add_generation_prompt': False", prompt)
