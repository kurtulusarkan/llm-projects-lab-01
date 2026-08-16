"""Read-only comparison helpers for completed YAML experiments."""

import csv
import json
import math
from pathlib import Path

COLUMNS = [
    "status",
    "experiment_name",
    "experiment_output",
    "model",
    "dataset",
    "training_examples",
    "epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "estimated_optimizer_steps",
    "actual_optimizer_steps",
    "train_runtime_seconds",
    "examples_per_second",
    "accuracy",
    "invalid_json_count",
]


def estimated_step_fields(training_examples, training: dict) -> dict:
    """Calculate display-only estimates without importing the training package."""
    required = {"batch_size", "gradient_accumulation_steps", "epochs"}
    if training_examples is None or not required <= training.keys():
        return {}
    batches_per_epoch = math.ceil(training_examples / training["batch_size"])
    steps_per_epoch = math.ceil(batches_per_epoch / training["gradient_accumulation_steps"])
    return {
        "effective_batch_size": training["batch_size"] * training["gradient_accumulation_steps"],
        "estimated_optimizer_steps": math.ceil(steps_per_epoch * training["epochs"]),
    }


def completion_status(metadata: dict, metrics_available: bool) -> str:
    """Classify the artifact without assuming one particular failure schema."""
    failure_keys = {"failure", "error", "failure_reason"}
    if metadata.get("status") == "failed" or metadata.get("failed") or failure_keys & metadata.keys():
        return "failed"
    return "completed" if metrics_available else "incomplete"


def load_experiment_row(directory: str | Path) -> dict:
    """Load one comparison row from an experiment's metadata and metrics files."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    training = metadata.get("training", {})
    training_examples = metadata.get("dataset_size")
    derived = estimated_step_fields(training_examples, training)
    return {
        "status": completion_status(metadata, metrics_path.is_file()),
        "experiment_name": metadata.get("experiment_name", directory.name),
        "experiment_output": str(directory),
        "model": metadata.get("model_name"),
        "dataset": metadata.get("dataset", "SST-2"),
        "training_examples": training_examples,
        "epochs": training.get("epochs"),
        "batch_size": training.get("batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "effective_batch_size": metadata.get("examples_per_optimizer_step", derived.get("effective_batch_size")),
        "estimated_optimizer_steps": metadata.get(
            "estimated_total_optimizer_steps", derived.get("estimated_optimizer_steps")
        ),
        "actual_optimizer_steps": metadata.get("actual_optimizer_steps"),
        "train_runtime_seconds": metadata.get("train_runtime_seconds"),
        "examples_per_second": metadata.get("examples_per_second"),
        "accuracy": metrics.get("accuracy"),
        "invalid_json_count": metrics.get("invalid_json_count"),
    }


def find_experiment_directories(root: str | Path, path_or_prefix: str | None = None) -> list[Path]:
    """Find complete experiment directories, optionally by path or name prefix."""
    root = Path(root)
    if path_or_prefix:
        requested = Path(path_or_prefix)
        if requested.is_dir():
            candidates = [requested] if (requested / "metadata.json").exists() else requested.glob("*")
        else:
            candidates = root.glob(f"{path_or_prefix}*")
    else:
        candidates = root.glob("*")
    return sorted(
        directory
        for directory in candidates
        if directory.is_dir() and (directory / "metadata.json").is_file()
    )


def comparison_rows(root: str | Path = "outputs/experiments", path_or_prefix: str | None = None) -> list[dict]:
    """Load completed experiment rows sorted by descending evaluation accuracy."""
    rows = [load_experiment_row(directory) for directory in find_experiment_directories(root, path_or_prefix)]
    return sorted(rows, key=lambda row: row["accuracy"] if row["accuracy"] is not None else -1, reverse=True)


def format_table(rows: list[dict]) -> str:
    """Render comparison rows as a compact plain-text table."""
    def value(row: dict, column: str) -> str:
        item = row.get(column)
        if item is None:
            return ""
        if column == "accuracy":
            return f"{item:.4f}"
        if column in {"train_runtime_seconds", "examples_per_second"}:
            return f"{item:.2f}"
        return str(item)

    values = [[value(row, column) for column in COLUMNS] for row in rows]
    widths = [max(len(column), *(len(row[index]) for row in values)) for index, column in enumerate(COLUMNS)]
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(COLUMNS))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(item.ljust(widths[index]) for index, item in enumerate(row)) for row in values]
    return "\n".join([header, divider, *body])


def write_csv(rows: list[dict], path: str | Path) -> None:
    """Write comparison rows with a stable column order."""
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
