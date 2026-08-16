import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lab_01.compare import comparison_rows, load_experiment_row


def write_experiment(directory: Path, name: str, accuracy: float, **metadata_overrides) -> None:
    directory.mkdir()
    metadata = {
        "experiment_name": name,
        "model_name": "Qwen/Qwen3-0.6B",
        "dataset": "SST-2",
        "dataset_size": 5000,
        "training": {"epochs": 1, "batch_size": 8, "gradient_accumulation_steps": 1},
    }
    metadata.update(metadata_overrides)
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "metrics.json").write_text(
        json.dumps({"accuracy": accuracy, "invalid_json_count": 0})
    )


class ExperimentComparisonTest(unittest.TestCase):
    def test_loads_and_sorts_multiple_experiment_directories(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_experiment(root / "lower", "lower", 0.8)
            write_experiment(root / "higher", "higher", 0.9)

            rows = comparison_rows(root)

        self.assertEqual([row["experiment_name"] for row in rows], ["higher", "lower"])
        self.assertEqual(rows[0]["estimated_optimizer_steps"], 625)
        self.assertEqual(rows[0]["effective_batch_size"], 8)

    def test_missing_optional_fields_are_left_empty(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "minimal"
            write_experiment(directory, "minimal", 0.8)

            row = load_experiment_row(directory)

        self.assertIsNone(row["actual_optimizer_steps"])
        self.assertIsNone(row["train_runtime_seconds"])
        self.assertIsNone(row["examples_per_second"])
        self.assertEqual(row["status"], "completed")

    def test_incomplete_and_failed_metadata_are_included(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "metadata.json").write_text(json.dumps({"model_name": "Qwen/Qwen3-0.6B"}))
            failed = root / "failed"
            failed.mkdir()
            (failed / "metadata.json").write_text(
                json.dumps({"model_name": "Qwen/Qwen3-0.6B", "failure": "CUDA out of memory"})
            )

            rows = comparison_rows(root)

        statuses = {row["experiment_name"]: row["status"] for row in rows}
        self.assertEqual(statuses["incomplete"], "incomplete")
        self.assertEqual(statuses["failed"], "failed")
