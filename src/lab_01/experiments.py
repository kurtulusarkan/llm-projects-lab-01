"""Small YAML-driven orchestration for repeatable SST-2 experiments."""

import copy
import hashlib
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import yaml

from lab_01.evaluate import evaluate_sst2_adapter
from lab_01.train import train_sst2_lora


def load_experiment_config(path: str | Path) -> dict:
    """Load one experiment YAML file."""
    with Path(path).open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")
    return config


def apply_overrides(config: dict, overrides: dict[str, object]) -> dict:
    """Apply supported CLI overrides to a copied experiment configuration."""
    resolved = copy.deepcopy(config)
    mapping = {
        "model": ("model", "name"),
        "train_size": ("dataset", "train_size"),
        "epochs": ("training", "epochs"),
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("training", "batch_size"),
        "output_dir": ("output", "root"),
    }
    for name, value in overrides.items():
        if value is not None:
            section, key = mapping[name]
            resolved.setdefault(section, {})[key] = value
    return resolved


def output_directory_name(config: dict) -> str:
    """Create a deterministic directory name from the experiment identity."""
    model = config["model"]["name"].rsplit("/", maxsplit=1)[-1].lower()
    dataset = str(config["dataset"]["name"]).lower().replace("_", "-")
    name = str(config["name"]).lower().replace("_", "-")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    return f"{name}-{model}-{dataset}-train{config['dataset']['train_size']}-{fingerprint}"


def resolved_config_document(original: dict, overrides: dict[str, object], resolved: dict) -> dict:
    """Build the self-contained record saved alongside an experiment."""
    return {
        "original_config": original,
        "cli_overrides": {name: value for name, value in overrides.items() if value is not None},
        "resolved_config": resolved,
    }


def unique_output_directory(root: str | Path, config: dict) -> Path:
    """Create a non-overwriting directory based on the deterministic identity."""
    root = Path(root)
    base = root / output_directory_name(config)
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base.name}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def git_commit() -> str | None:
    """Return the current commit when Git metadata is available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def completion_summary(
    config: dict, output_dir: Path, training_metrics: dict, evaluation_metrics: dict
) -> str:
    """Format the concise terminal summary for a completed experiment."""
    return "\n".join(
        [
            "=== Experiment Complete ===",
            f"experiment_name: {config['name']}",
            f"model: {config['model']['name']}",
            f"training_examples: {config['dataset']['train_size']}",
            f"train_runtime_seconds: {training_metrics['train_runtime']:.2f}",
            f"accuracy: {evaluation_metrics['accuracy']:.4f}",
            f"invalid_json_count: {evaluation_metrics['invalid_json_count']}",
            f"total_examples: {evaluation_metrics['total_examples']}",
            f"output_directory: {output_dir}",
        ]
    )


def run_experiment(config_path: str | Path, overrides: dict[str, object]) -> Path:
    """Train, evaluate, and save records for one resolved YAML experiment."""
    original = load_experiment_config(config_path)
    resolved = apply_overrides(original, overrides)
    output_dir = unique_output_directory(resolved["output"]["root"], resolved)
    resolved["output"]["directory"] = str(output_dir)
    adapter_dir = output_dir / "adapter"

    (output_dir / "config.yaml").write_text(yaml.safe_dump(original, sort_keys=False))
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config_document(original, overrides, resolved), sort_keys=False)
    )
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "model_name": resolved["model"]["name"],
        "dataset_size": resolved["dataset"]["train_size"],
        "lora": resolved["lora"],
        "training": resolved["training"],
        "output_directory": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    training = resolved["training"]
    with (output_dir / "train.log").open("w") as train_log:
        with redirect_stdout(_Tee(sys.stdout, train_log)):
            training_result = train_sst2_lora(
                model_name=resolved["model"]["name"],
                output_dir=str(adapter_dir),
                seed=resolved["dataset"]["seed"],
                train_size=resolved["dataset"]["train_size"],
                epochs=training["epochs"],
                learning_rate=training["learning_rate"],
                batch_size=training["batch_size"],
                gradient_accumulation_steps=training["gradient_accumulation_steps"],
                gradient_checkpointing=training["gradient_checkpointing"],
                lora_rank=resolved["lora"]["rank"],
                lora_alpha=resolved["lora"]["alpha"],
                lora_dropout=resolved["lora"]["dropout"],
            )

    metrics, elapsed = evaluate_sst2_adapter(
        model_name=resolved["model"]["name"],
        adapter_path=str(adapter_dir),
        batch_size=resolved["output"].get("evaluation_batch_size", 32),
    )
    evaluation = {**metrics, "total_evaluation_time_s": elapsed}
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(completion_summary(resolved, output_dir, training_result.metrics, metrics))
    return output_dir
