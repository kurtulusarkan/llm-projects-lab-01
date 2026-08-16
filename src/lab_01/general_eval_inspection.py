"""Read-only helpers for inspecting saved general-evaluation artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab_01.general_eval import load_eval_dataset


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Stored outputs from one general-evaluation run."""

    directory: Path
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    examples_by_id: dict[str, dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing artifact: {path}") from error

    predictions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    required = {"id", "category", "prompt", "output", "score"}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            prediction = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid prediction JSON on line {line_number}: {error.msg}") from error
        if not isinstance(prediction, dict) or not required.issubset(prediction):
            raise ValueError(f"prediction line {line_number} is missing required fields")
        if not isinstance(prediction["id"], str) or prediction["id"] in seen_ids:
            raise ValueError(f"duplicate or invalid prediction id on line {line_number}")
        if not isinstance(prediction["score"], bool):
            raise ValueError(f"prediction {prediction['id']} has a non-boolean score")
        seen_ids.add(prediction["id"])
        predictions.append(prediction)
    return predictions


def _dataset_path(directory: Path, metadata: dict[str, Any]) -> Path:
    raw_path = metadata.get("dataset_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("metadata.json does not contain dataset_path")

    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, directory / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"dataset referenced by metadata is unavailable: {raw_path}")


def load_evaluation_artifacts(eval_dir: str | Path) -> EvaluationArtifacts:
    """Load saved artifacts and the referenced dataset without scoring outputs."""
    directory = Path(eval_dir)
    metadata = _load_json(directory / "metadata.json")
    metrics = _load_json(directory / "metrics.json")
    predictions = _load_predictions(directory / "predictions.jsonl")
    examples = load_eval_dataset(_dataset_path(directory, metadata))
    examples_by_id = {example["id"]: example for example in examples}
    return EvaluationArtifacts(directory, metadata, metrics, predictions, examples_by_id)


def select_predictions(
    artifacts: EvaluationArtifacts,
    *,
    category: str | None = None,
    failed_only: bool = False,
    item_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter stored predictions without recalculating any score."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    selected = artifacts.predictions
    if category is not None:
        selected = [prediction for prediction in selected if prediction["category"] == category]
    if failed_only:
        selected = [prediction for prediction in selected if not prediction["score"]]
    if item_id is not None:
        selected = [prediction for prediction in selected if prediction["id"] == item_id]
    return selected[:limit] if limit is not None else selected


def summary(artifacts: EvaluationArtifacts, selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize stored scores and explain invalid-count scope precisely."""
    passed = sum(prediction["score"] for prediction in artifacts.predictions)
    total = len(artifacts.predictions)
    selected_is_full_run = len(selected) == total and {
        prediction["id"] for prediction in selected
    } == {prediction["id"] for prediction in artifacts.predictions}
    return {
        "total_examples": total,
        "passed": passed,
        "failed": total - passed,
        "invalid_outputs": artifacts.metrics.get("invalid_output_count"),
        "selected_examples": len(selected),
        "invalid_count_applies_to_selection": selected_is_full_run,
    }


def format_expected(expected: Any) -> str:
    """Render scalar and structured frozen expectations for terminal inspection."""
    if isinstance(expected, str):
        return expected
    return json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True)


def failure_reason(prediction: dict[str, Any]) -> str:
    """Return a stored reason when newer artifacts provide one; never infer one."""
    if prediction["score"]:
        return "not applicable"
    for key in ("failure_reason", "reason"):
        value = prediction.get(key)
        if isinstance(value, str) and value:
            return value
    return "not recorded"


def format_prediction(artifacts: EvaluationArtifacts, prediction: dict[str, Any]) -> str:
    """Format one stored prediction with its frozen evaluation context."""
    example = artifacts.examples_by_id.get(prediction["id"])
    evaluation_type = example["evaluation_type"] if example else "unavailable"
    expected = format_expected(example["expected"]) if example else "unavailable"
    prompt = example["prompt"] if example else prediction["prompt"]
    return "\n".join(
        [
            f"ID: {prediction['id']}",
            f"Category: {prediction['category']}",
            f"Evaluation type: {evaluation_type}",
            "Prompt:",
            prompt,
            "Expected:",
            expected,
            "Model output:",
            prediction["output"],
            f"Passed: {prediction['score']}",
            f"Failure reason: {failure_reason(prediction)}",
        ]
    )


def format_comparison(
    base: EvaluationArtifacts,
    adapter: EvaluationArtifacts,
    selected_base_predictions: list[dict[str, Any]],
) -> list[str]:
    """Format paired, matching IDs using saved scores and outputs only."""
    adapter_by_id = {prediction["id"]: prediction for prediction in adapter.predictions}
    formatted: list[str] = []
    for base_prediction in selected_base_predictions:
        adapter_prediction = adapter_by_id.get(base_prediction["id"])
        if adapter_prediction is None:
            continue
        example = base.examples_by_id.get(base_prediction["id"]) or adapter.examples_by_id.get(
            base_prediction["id"]
        )
        prompt = example["prompt"] if example else base_prediction["prompt"]
        expected = format_expected(example["expected"]) if example else "unavailable"
        formatted.append(
            "\n".join(
                [
                    f"ID: {base_prediction['id']}",
                    "Prompt:",
                    prompt,
                    "Expected:",
                    expected,
                    "BASE:",
                    f"output: {base_prediction['output']}",
                    f"score: {base_prediction['score']}",
                    "ADAPTER:",
                    f"output: {adapter_prediction['output']}",
                    f"score: {adapter_prediction['score']}",
                ]
            )
        )
    return formatted
