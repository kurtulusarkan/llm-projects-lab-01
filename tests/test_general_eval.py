import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lab_01.general_eval import build_metadata
from lab_01.general_eval import dataset_checksum
from lab_01.general_eval import evaluate_general
from lab_01.general_eval import load_eval_dataset
from lab_01.general_eval import score_prediction
from lab_01.general_eval import ScoreResult
from lab_01.general_eval import summarize_scores
from lab_01.general_eval import validate_python_code


def example(evaluation_type, expected, prompt="Reply with the requested answer.", example_id="test-001"):
    return {
        "id": example_id,
        "category": "reasoning",
        "prompt": prompt,
        "expected": expected,
        "evaluation_type": evaluation_type,
        "difficulty": "easy",
    }


class GeneralScoringTest(unittest.TestCase):
    def test_strict_exact_normalizes_only_line_endings_and_trailing_whitespace(self):
        item = example("strict_exact", "first\nsecond")

        self.assertTrue(score_prediction(item, "first  \r\nsecond\t\r\n").score)
        self.assertFalse(score_prediction(item, " first\nsecond").score)
        self.assertFalse(score_prediction(item, "first\nsecond.").score)

    def test_short_answer_accepts_reviewed_variants(self):
        item = example("short_answer", {"accepted": ["The dog runs", "The dog is running"]})

        self.assertTrue(score_prediction(item, "  The dog is   running  ").score)
        self.assertFalse(score_prediction(item, "The dog ran").score)

    def test_multiple_choice_rejects_explanatory_text(self):
        item = example(
            "multiple_choice",
            "A",
            prompt="Choose one. Reply A, B, or C only. A: red B: blue C: green",
        )

        self.assertTrue(score_prediction(item, "A\n").score)
        wrong_option = score_prediction(item, "B")
        self.assertFalse(wrong_option.score)
        self.assertFalse(wrong_option.invalid)
        explanatory = score_prediction(item, "The answer is A")
        self.assertFalse(explanatory.score)
        self.assertTrue(explanatory.invalid)

    def test_code_validator_accepts_semantically_correct_function(self):
        result = validate_python_code("coding-008", "def add(a, b):\n    return a + b")

        self.assertTrue(result.score)
        self.assertFalse(result.invalid)

    def test_code_validator_accepts_a_fenced_alternative_implementation(self):
        result = validate_python_code(
            "coding-022",
            "```python\ndef square(n):\n    return n ** 2\n```",
        )

        self.assertTrue(result.score)

    def test_code_validator_distinguishes_wrong_behavior_from_invalid_code(self):
        wrong = validate_python_code("coding-008", "def add(a, b):\n    return a - b")
        unsafe = validate_python_code("coding-008", "import os\ndef add(a, b):\n    return a + b")

        self.assertFalse(wrong.score)
        self.assertFalse(wrong.invalid)
        self.assertFalse(unsafe.score)
        self.assertTrue(unsafe.invalid)

    def test_every_frozen_code_validator_accepts_a_valid_solution(self):
        solutions = {
            "coding-008": "def add(a, b):\n    return a + b",
            "coding-011": "def larger(a, b):\n    return max(a, b)",
            "coding-013": '"yes" if x > 0 else "no"',
            "coding-014": "for item in items:\n    print(item)",
            "coding-016": "def is_even(n):\n    return n % 2 == 0",
            "coding-020": "def count_values(values):\n    return len(values)",
            "coding-021": "def is_five(x):\n    if x == 5:\n        return True\n    return False",
            "coding-022": "def square(n):\n    return n * n",
            "coding-023": "items.append(item)",
            "coding-025": '{"name": "Ada"}',
        }

        for example_id, source in solutions.items():
            with self.subTest(example_id=example_id):
                self.assertEqual(validate_python_code(example_id, source), ScoreResult(True))

    def test_metrics_are_grouped_by_regression_dimensions(self):
        examples = [
            example("short_answer", "yes", example_id="reasoning-001"),
            {
                **example("multiple_choice", "A", prompt="A: red B: blue", example_id="knowledge-001"),
                "category": "knowledge",
                "difficulty": "medium",
            },
        ]

        metrics = summarize_scores(examples, [ScoreResult(True), ScoreResult(False, invalid=True)])

        self.assertEqual(metrics["overall_accuracy"], 0.5)
        self.assertEqual(metrics["category_accuracy"], {"knowledge": 0.0, "reasoning": 1.0})
        self.assertEqual(metrics["evaluation_type_accuracy"], {"multiple_choice": 0.0, "short_answer": 1.0})
        self.assertEqual(metrics["difficulty_accuracy"], {"easy": 1.0, "medium": 0.0})
        self.assertEqual(metrics["invalid_output_count"], 1)


class GeneralMetadataTest(unittest.TestCase):
    def test_frozen_v1_dataset_matches_recorded_checksum(self):
        path = Path(__file__).resolve().parents[1] / "evals/general/v1/dataset.jsonl"

        examples = load_eval_dataset(path)

        self.assertEqual(len(examples), 115)
        self.assertEqual(
            dataset_checksum(path),
            "dfebee2df92e034311b63f058248d08cdd3739b3fa08808e91f6cb1bc951a0fe",
        )

    def test_dataset_checksum_uses_raw_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.jsonl"
            path.write_bytes(b"abc\n")

            self.assertEqual(dataset_checksum(path), hashlib.sha256(b"abc\n").hexdigest())

    def test_metadata_contains_reproducibility_fields(self):
        metadata = build_metadata(
            model_name="Qwen/Qwen3-0.6B",
            adapter_path="outputs/example/adapter",
            dataset_path="evals/general/v1/dataset.jsonl",
            dataset_version="v1",
            checksum="abc123",
            generation_parameters={"do_sample": False, "max_new_tokens": 128, "batch_size": 16},
            system_prompt="Be concise.",
            chat_template={"model_family": "qwen", "qwen_thinking_disabled": True},
            output_directory="outputs/evals/general/v1/run",
            selected_examples=10,
            category="reasoning",
            limit=10,
            timestamp="2026-08-16T12:00:00+00:00",
            git_commit="deadbeef",
        )

        self.assertEqual(metadata["dataset_version"], "v1")
        self.assertEqual(metadata["dataset_checksum"], "abc123")
        self.assertEqual(metadata["adapter_path"], "outputs/example/adapter")
        self.assertFalse(metadata["generation_parameters"]["do_sample"])
        self.assertTrue(metadata["chat_template"]["qwen_thinking_disabled"])
        self.assertEqual(metadata["timestamp"], "2026-08-16T12:00:00+00:00")

    def test_mocked_evaluation_writes_required_artifacts(self):
        class FakeModel:
            def eval(self):
                return self

        class FakeTokenizer:
            chat_template = "fake-template"

            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]["content"]

        record = {
            "id": "instruction-following-test",
            "category": "instruction_following",
            "prompt": "Reply OK.",
            "expected": "OK",
            "evaluation_type": "strict_exact",
            "difficulty": "easy",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "v1" / "dataset.jsonl"
            dataset.parent.mkdir()
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            output_directory = root / "artifacts"

            with (
                patch("lab_01.general_eval.load_model", return_value=(FakeModel(), FakeTokenizer())),
                patch("lab_01.general_eval.generate_batch", return_value=["OK"]),
                patch("lab_01.general_eval.git_commit_hash", return_value="deadbeef"),
            ):
                result_directory, metrics = evaluate_general(
                    model_name="Qwen/Qwen3-0.6B",
                    dataset_path=dataset,
                    output_dir=output_directory,
                    batch_size=1,
                    show_progress=False,
                )

            self.assertEqual(result_directory, output_directory)
            self.assertEqual(metrics["overall_accuracy"], 1.0)
            metadata = json.loads((output_directory / "metadata.json").read_text())
            prediction = json.loads((output_directory / "predictions.jsonl").read_text())
            saved_metrics = json.loads((output_directory / "metrics.json").read_text())
            self.assertEqual(metadata["dataset_version"], "v1")
            self.assertEqual(metadata["dataset_checksum"], dataset_checksum(dataset))
            self.assertEqual(metadata["git_commit"], "deadbeef")
            self.assertEqual(
                set(prediction),
                {"id", "category", "prompt", "output", "score"},
            )
            self.assertTrue(prediction["score"])
            self.assertEqual(saved_metrics["invalid_output_count"], 0)


if __name__ == "__main__":
    unittest.main()
