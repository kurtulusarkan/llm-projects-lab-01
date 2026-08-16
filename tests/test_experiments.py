import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lab_01.experiments import (
    apply_overrides,
    completion_summary,
    load_experiment_config,
    output_directory_name,
    resolved_config_document,
)


class ExperimentConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "name": "sst2-qwen3-500",
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "dataset": {"name": "SST-2", "train_size": 500, "seed": 42},
            "training": {
                "epochs": 3,
                "learning_rate": 0.0002,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
            },
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
            "output": {"root": "outputs/experiments"},
        }

    def test_loads_yaml_mapping(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text("name: test\nmodel:\n  name: Qwen/Qwen3-0.6B\n")
            self.assertEqual(load_experiment_config(path)["name"], "test")

    def test_cli_overrides_take_precedence_without_mutating_original(self):
        resolved = apply_overrides(self.config, {"train_size": 5000, "batch_size": 4})
        self.assertEqual(self.config["dataset"]["train_size"], 500)
        self.assertEqual(resolved["dataset"]["train_size"], 5000)
        self.assertEqual(resolved["training"]["batch_size"], 4)

    def test_output_name_is_deterministic_and_includes_identity(self):
        name = output_directory_name(self.config)
        self.assertEqual(name, output_directory_name(self.config))
        self.assertIn("sst2-qwen3-500", name)
        self.assertIn("qwen3-0.6b", name)
        self.assertIn("train500", name)
        changed = apply_overrides(self.config, {"train_size": 5000})
        self.assertNotEqual(name, output_directory_name(changed))

    def test_resolved_config_records_original_overrides_and_final_values(self):
        overrides = {"train_size": 5000, "model": None}
        resolved = apply_overrides(self.config, overrides)
        document = resolved_config_document(self.config, overrides, resolved)
        self.assertEqual(document["original_config"]["dataset"]["train_size"], 500)
        self.assertEqual(document["cli_overrides"], {"train_size": 5000})
        self.assertEqual(document["resolved_config"]["dataset"]["train_size"], 5000)

    def test_completion_summary_uses_existing_result_values(self):
        summary = completion_summary(
            self.config,
            Path("outputs/experiments/example"),
            {"train_runtime": 77.89},
            {
                "accuracy": 0.8945,
                "invalid_json_count": 0,
                "total_examples": 872,
            },
        )
        self.assertIn("=== Experiment Complete ===", summary)
        self.assertIn("train_runtime_seconds: 77.89", summary)
        self.assertIn("accuracy: 0.8945", summary)
        self.assertIn("output_directory: outputs/experiments/example", summary)
