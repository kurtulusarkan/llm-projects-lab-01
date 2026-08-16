import json
import tempfile
import unittest
from pathlib import Path

from lab_01.general_eval_inspection import format_comparison
from lab_01.general_eval_inspection import format_prediction
from lab_01.general_eval_inspection import load_evaluation_artifacts
from lab_01.general_eval_inspection import select_predictions
from lab_01.general_eval_inspection import summary


DATASET_ROWS = [
    {
        "id": "instruction-001",
        "category": "instruction_following",
        "prompt": "Reply exactly OK.",
        "expected": "OK",
        "evaluation_type": "strict_exact",
        "difficulty": "easy",
    },
    {
        "id": "reasoning-001",
        "category": "reasoning",
        "prompt": "What is 1 + 1?",
        "expected": "2",
        "evaluation_type": "short_answer",
        "difficulty": "easy",
    },
]


def write_artifacts(root: Path, name: str, predictions: list[dict]) -> Path:
    dataset_path = root / "dataset.jsonl"
    if not dataset_path.exists():
        dataset_path.write_text(
            "".join(json.dumps(row) + "\n" for row in DATASET_ROWS),
            encoding="utf-8",
        )
    directory = root / name
    directory.mkdir()
    (directory / "metadata.json").write_text(
        json.dumps({"dataset_path": str(dataset_path)}),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps({"invalid_output_count": 1, "total_examples": len(predictions)}),
        encoding="utf-8",
    )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(prediction) + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    return directory


class GeneralEvaluationInspectionTest(unittest.TestCase):
    def test_filters_and_formats_saved_predictions_without_rescoring(self):
        predictions = [
            {
                "id": "instruction-001",
                "category": "instruction_following",
                "prompt": "Reply exactly OK.",
                "output": "OK",
                "score": False,
                "failure_reason": "stored diagnostic",
            },
            {
                "id": "reasoning-001",
                "category": "reasoning",
                "prompt": "What is 1 + 1?",
                "output": "2",
                "score": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = write_artifacts(root, "base", predictions)
            before = (directory / "predictions.jsonl").read_bytes()

            artifacts = load_evaluation_artifacts(directory)
            failed = select_predictions(artifacts, failed_only=True)
            selected_by_id = select_predictions(artifacts, item_id="reasoning-001")
            values = summary(artifacts, failed)
            rendered = format_prediction(artifacts, failed[0])

            self.assertEqual([prediction["id"] for prediction in failed], ["instruction-001"])
            self.assertEqual([prediction["id"] for prediction in selected_by_id], ["reasoning-001"])
            self.assertEqual(values["passed"], 1)
            self.assertEqual(values["failed"], 1)
            self.assertFalse(values["invalid_count_applies_to_selection"])
            self.assertIn("Evaluation type: strict_exact", rendered)
            self.assertIn("Expected:\nOK", rendered)
            self.assertIn("Passed: False", rendered)
            self.assertIn("Failure reason: stored diagnostic", rendered)
            self.assertEqual((directory / "predictions.jsonl").read_bytes(), before)

    def test_category_and_limit_filtering(self):
        predictions = [
            {
                "id": "instruction-001",
                "category": "instruction_following",
                "prompt": "Reply exactly OK.",
                "output": "wrong",
                "score": False,
            },
            {
                "id": "reasoning-001",
                "category": "reasoning",
                "prompt": "What is 1 + 1?",
                "output": "2",
                "score": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = load_evaluation_artifacts(
                write_artifacts(Path(temporary_directory), "base", predictions)
            )

            selected = select_predictions(artifacts, category="reasoning", limit=1)

            self.assertEqual([prediction["id"] for prediction in selected], ["reasoning-001"])

    def test_comparison_uses_matching_stored_ids(self):
        base_predictions = [
            {
                "id": "instruction-001",
                "category": "instruction_following",
                "prompt": "Reply exactly OK.",
                "output": "bad",
                "score": False,
            },
            {
                "id": "reasoning-001",
                "category": "reasoning",
                "prompt": "What is 1 + 1?",
                "output": "2",
                "score": True,
            },
        ]
        adapter_predictions = [
            {
                "id": "instruction-001",
                "category": "instruction_following",
                "prompt": "Reply exactly OK.",
                "output": "OK",
                "score": True,
            },
            {
                "id": "extra-id",
                "category": "reasoning",
                "prompt": "Unrelated.",
                "output": "x",
                "score": False,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = load_evaluation_artifacts(write_artifacts(root, "base", base_predictions))
            adapter = load_evaluation_artifacts(write_artifacts(root, "adapter", adapter_predictions))

            rendered = format_comparison(base, adapter, base.predictions)

            self.assertEqual(len(rendered), 1)
            self.assertIn("ID: instruction-001", rendered[0])
            self.assertIn("BASE:\noutput: bad\nscore: False", rendered[0])
            self.assertIn("ADAPTER:\noutput: OK\nscore: True", rendered[0])


if __name__ == "__main__":
    unittest.main()
