"""Deterministic general-capability regression evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from lab_01.inference import generate_batch
from lab_01.models import format_messages, load_model, model_family


DEFAULT_DATASET = Path("evals/general/v1/dataset.jsonl")
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions and answer concisely."
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_BATCH_SIZE = 16

REQUIRED_FIELDS = {"id", "category", "prompt", "expected", "evaluation_type", "difficulty"}
CATEGORIES = {"instruction_following", "reasoning", "knowledge", "coding", "language"}
EVALUATION_TYPES = {"strict_exact", "short_answer", "multiple_choice", "code"}
DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring one generated response."""

    score: bool
    invalid: bool = False


def dataset_checksum(path: str | Path) -> str:
    """Return the SHA256 checksum of the dataset's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalized_prompt(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", prompt.casefold()).strip()


def validate_eval_dataset(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the frozen JSONL schema and return deterministic counts."""
    seen_ids: set[str] = set()
    seen_prompts: dict[str, str] = {}

    for index, example in enumerate(examples, start=1):
        if set(example) != REQUIRED_FIELDS:
            raise ValueError(f"example {index} must contain exactly {sorted(REQUIRED_FIELDS)}")
        if not isinstance(example["id"], str) or not example["id"].strip():
            raise ValueError(f"example {index} has an invalid id")
        if example["id"] in seen_ids:
            raise ValueError(f"duplicate id: {example['id']}")
        seen_ids.add(example["id"])

        if example["category"] not in CATEGORIES:
            raise ValueError(f"{example['id']} has an invalid category")
        if example["evaluation_type"] not in EVALUATION_TYPES:
            raise ValueError(f"{example['id']} has an invalid evaluation_type")
        if example["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"{example['id']} has an invalid difficulty")
        if not isinstance(example["prompt"], str) or not example["prompt"].strip():
            raise ValueError(f"{example['id']} has an invalid prompt")

        normalized = _normalized_prompt(example["prompt"])
        if normalized in seen_prompts:
            raise ValueError(f"duplicate prompt: {seen_prompts[normalized]} and {example['id']}")
        seen_prompts[normalized] = example["id"]
        _validate_expected(example)

    return {
        "total_examples": len(examples),
        "category_counts": dict(sorted(Counter(row["category"] for row in examples).items())),
        "evaluation_type_counts": dict(
            sorted(Counter(row["evaluation_type"] for row in examples).items())
        ),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in examples).items())),
    }


def _validate_expected(example: dict[str, Any]) -> None:
    expected = example["expected"]
    evaluation_type = example["evaluation_type"]
    example_id = example["id"]

    if evaluation_type in {"strict_exact", "multiple_choice"}:
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"{example_id} requires a non-empty string expected value")
        return
    if evaluation_type == "short_answer":
        if isinstance(expected, str) and expected:
            return
        if (
            isinstance(expected, dict)
            and set(expected) == {"accepted"}
            and isinstance(expected["accepted"], list)
            and expected["accepted"]
            and all(isinstance(value, str) and value for value in expected["accepted"])
        ):
            return
        raise ValueError(f"{example_id} has an invalid short_answer expectation")
    if not (
        isinstance(expected, dict)
        and set(expected) == {"validation"}
        and isinstance(expected["validation"], list)
        and expected["validation"]
        and all(isinstance(value, str) and value for value in expected["validation"])
    ):
        raise ValueError(f"{example_id} has an invalid code expectation")


def load_eval_dataset(
    path: str | Path,
    *,
    category: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load, validate, and optionally select frozen evaluation examples."""
    dataset_path = Path(path)
    examples: list[dict[str, Any]] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
    validate_eval_dataset(examples)

    if category is not None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        examples = [example for example in examples if example["category"] == category]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        examples = examples[:limit]
    return examples


def normalize_strict(text: str) -> str:
    """Normalize line endings and trailing whitespace only."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))
    return normalized.rstrip("\n")


def normalize_short_answer(text: str) -> str:
    """Apply minimal whitespace normalization without changing case or punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def _accepted_answers(expected: str | dict[str, list[str]]) -> list[str]:
    if isinstance(expected, str):
        return [expected]
    return expected["accepted"]


def _multiple_choice_options(prompt: str, expected: str) -> set[str]:
    options = set(re.findall(r"(?:^|\s)([A-Z]):", prompt))
    if re.search(r"\byes or no\b", prompt, flags=re.IGNORECASE):
        options.update({"yes", "no"})
    options.add(expected)
    return options


def score_prediction(example: dict[str, Any], output: str) -> ScoreResult:
    """Score one response according to its frozen evaluation type."""
    if not isinstance(output, str) or not normalize_short_answer(output):
        return ScoreResult(False, invalid=True)

    evaluation_type = example["evaluation_type"]
    if evaluation_type == "strict_exact":
        return ScoreResult(normalize_strict(output) == normalize_strict(example["expected"]))
    if evaluation_type == "short_answer":
        prediction = normalize_short_answer(output)
        accepted = {normalize_short_answer(value) for value in _accepted_answers(example["expected"])}
        return ScoreResult(prediction in accepted)
    if evaluation_type == "multiple_choice":
        prediction = normalize_short_answer(output)
        options = _multiple_choice_options(example["prompt"], example["expected"])
        if prediction not in options:
            return ScoreResult(False, invalid=True)
        return ScoreResult(prediction == example["expected"])
    if evaluation_type == "code":
        return validate_python_code(example["id"], output)
    raise ValueError(f"unsupported evaluation type: {evaluation_type}")


_ALLOWED_AST_NODES = (
    ast.Module,
    ast.Expression,
    ast.Expr,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.If,
    ast.For,
    ast.Assign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)
_ALLOWED_CALLS = {"len", "max", "print"}
_EXPRESSION_CODE_IDS = {"coding-013", "coding-025"}
_SUPPORTED_CODE_IDS = {
    "coding-008",
    "coding-011",
    "coding-013",
    "coding-014",
    "coding-016",
    "coding-020",
    "coding-021",
    "coding-022",
    "coding-023",
    "coding-025",
}


class _SafePythonVisitor(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"disallowed Python syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr != "append" or not isinstance(node.value, ast.Name) or node.value.id != "items":
            raise ValueError("attribute access is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_CALLS:
                raise ValueError(f"call is not allowed: {node.func.id}")
        elif not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "items"
        ):
            raise ValueError("call target is not allowed")
        self.generic_visit(node)


def _extract_code(output: str) -> str:
    text = output.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```" or lines[0].strip() not in {"```", "```python"}:
        return ""
    return "\n".join(lines[1:-1]).strip()


_CODE_WORKER = r"""
import json
import sys

payload = json.loads(sys.stdin.read())
source = payload["source"]
example_id = payload["example_id"]
mode = payload["mode"]
printed = []

def safe_print(*values, **kwargs):
    if kwargs:
        raise ValueError("print keyword arguments are not supported")
    printed.append(" ".join(str(value) for value in values))

safe_builtins = {"len": len, "max": max, "print": safe_print}

def namespace(**values):
    return {"__builtins__": safe_builtins, **values}

passed = False
if example_id == "coding-013":
    compiled = compile(source, "<model-output>", "eval")
    passed = eval(compiled, namespace(x=2)) == "yes" and eval(compiled, namespace(x=0)) == "no"
elif example_id == "coding-025":
    compiled = compile(source, "<model-output>", "eval")
    passed = eval(compiled, namespace()) == {"name": "Ada"}
else:
    compiled = compile(source, "<model-output>", mode)
    values = namespace()
    if example_id == "coding-014":
        values["items"] = [3, 1]
    elif example_id == "coding-023":
        values.update(items=[1], item=2)
    exec(compiled, values, values)
    if example_id == "coding-008":
        function = values.get("add")
        passed = callable(function) and function(2, 3) == 5 and function(-1, 1) == 0
    elif example_id == "coding-011":
        function = values.get("larger")
        passed = callable(function) and function(2, 7) == 7 and function(5, 5) == 5
    elif example_id == "coding-014":
        passed = printed == ["3", "1"]
    elif example_id == "coding-016":
        function = values.get("is_even")
        passed = callable(function) and function(4) is True and function(5) is False
    elif example_id == "coding-020":
        function = values.get("count_values")
        passed = callable(function) and function([]) == 0 and function([4, 5]) == 2
    elif example_id == "coding-021":
        function = values.get("is_five")
        passed = callable(function) and function(5) is True and function(4) is False
    elif example_id == "coding-022":
        function = values.get("square")
        passed = callable(function) and function(3) == 9 and function(-2) == 4
    elif example_id == "coding-023":
        passed = values["items"] == [1, 2]

sys.stdout.write(json.dumps({"passed": bool(passed)}))
"""


def _restrict_subprocess() -> None:
    """Apply conservative POSIX limits before running validated code."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    except (ImportError, OSError, ValueError):
        pass


def validate_python_code(example_id: str, output: str) -> ScoreResult:
    """Validate one supported v1 Python response in a restricted subprocess."""
    if example_id not in _SUPPORTED_CODE_IDS:
        return ScoreResult(False, invalid=True)
    source = _extract_code(output)
    if not source or len(source) > 4_000:
        return ScoreResult(False, invalid=True)

    mode = "eval" if example_id in _EXPRESSION_CODE_IDS else "exec"
    try:
        tree = ast.parse(source, mode=mode)
        _SafePythonVisitor().visit(tree)
    except (SyntaxError, ValueError):
        return ScoreResult(False, invalid=True)

    if example_id == "coding-011" and not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "max"
        for node in ast.walk(tree)
    ):
        return ScoreResult(False)

    payload = json.dumps({"source": source, "example_id": example_id, "mode": mode})
    try:
        with tempfile.TemporaryDirectory(prefix="lab01-general-eval-") as temporary_directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _CODE_WORKER],
                input=payload,
                text=True,
                capture_output=True,
                timeout=2,
                cwd=temporary_directory,
                env={"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"},
                preexec_fn=_restrict_subprocess if os.name == "posix" else None,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired):
        return ScoreResult(False, invalid=True)

    if result.returncode != 0:
        return ScoreResult(False, invalid=True)
    try:
        return ScoreResult(bool(json.loads(result.stdout)["passed"]))
    except (json.JSONDecodeError, KeyError, TypeError):
        return ScoreResult(False, invalid=True)


def summarize_scores(
    examples: list[dict[str, Any]],
    scores: list[ScoreResult],
) -> dict[str, Any]:
    """Aggregate accuracy across the frozen regression dimensions."""
    if len(examples) != len(scores):
        raise ValueError("examples and scores must have the same length")

    def accuracy_for(field: str) -> dict[str, float]:
        values: dict[str, list[bool]] = {}
        for example, result in zip(examples, scores, strict=True):
            values.setdefault(example[field], []).append(result.score)
        return {key: sum(results) / len(results) for key, results in sorted(values.items())}

    total = len(scores)
    return {
        "overall_accuracy": sum(result.score for result in scores) / total if total else 0.0,
        "category_accuracy": accuracy_for("category"),
        "evaluation_type_accuracy": accuracy_for("evaluation_type"),
        "difficulty_accuracy": accuracy_for("difficulty"),
        "invalid_output_count": sum(result.invalid for result in scores),
        "total_examples": total,
    }


def git_commit_hash(workdir: str | Path | None = None) -> str | None:
    """Return the current Git commit when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def chat_template_information(tokenizer: Any, model_name: str) -> dict[str, Any]:
    """Describe the exact tokenizer chat template used for a run."""
    template = getattr(tokenizer, "chat_template", None)
    return {
        "model_family": model_family(model_name),
        "tokenizer_class": type(tokenizer).__name__,
        "template": template,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest()
        if isinstance(template, str)
        else None,
        "qwen_thinking_disabled": model_family(model_name) == "qwen",
    }


def build_metadata(
    *,
    model_name: str,
    adapter_path: str | None,
    dataset_path: str | Path,
    dataset_version: str,
    checksum: str,
    generation_parameters: dict[str, Any],
    system_prompt: str,
    chat_template: dict[str, Any],
    output_directory: str | Path,
    selected_examples: int,
    category: str | None,
    limit: int | None,
    timestamp: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build serializable, reproducible run metadata."""
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "dataset_path": str(dataset_path),
        "dataset_version": dataset_version,
        "dataset_checksum": checksum,
        "selected_examples": selected_examples,
        "category_filter": category,
        "limit": limit,
        "generation_parameters": generation_parameters,
        "system_prompt": system_prompt,
        "chat_template": chat_template,
        "output_directory": str(output_directory),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def create_output_directory(
    *,
    model_name: str,
    adapter_path: str | None,
    dataset_version: str,
    checksum: str,
    output_dir: str | Path | None,
) -> Path:
    """Create a new artifact directory without overwriting prior results."""
    if output_dir is not None:
        path = Path(output_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        variant = f"adapter-{_slug(Path(adapter_path).name)}" if adapter_path else "base"
        run_id = f"{timestamp}-{_slug(model_name)}-{variant}-{checksum[:8]}"
        path = Path("outputs/evals/general") / dataset_version / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_general(
    *,
    model_name: str = DEFAULT_MODEL,
    adapter_path: str | None = None,
    dataset_path: str | Path = DEFAULT_DATASET,
    limit: int | None = None,
    category: str | None = None,
    output_dir: str | Path | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    show_progress: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Load a base or adapter model, run the suite, and write artifacts."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    dataset_path = Path(dataset_path)
    examples = load_eval_dataset(dataset_path, category=category, limit=limit)
    checksum = dataset_checksum(dataset_path)
    dataset_version = dataset_path.parent.name

    model, tokenizer = load_model(model_name)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    predictions: list[dict[str, Any]] = []
    score_results: list[ScoreResult] = []
    progress = tqdm(total=len(examples), desc="Evaluating general", unit="examples") if show_progress else None
    started = time.perf_counter()
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [
                format_messages(
                    tokenizer,
                    model_name,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": example["prompt"]},
                    ],
                    add_generation_prompt=True,
                )
                for example in batch
            ]
            outputs = generate_batch(model, tokenizer, prompts, max_new_tokens)
            for example, output in zip(batch, outputs, strict=True):
                result = score_prediction(example, output)
                score_results.append(result)
                predictions.append(
                    {
                        "id": example["id"],
                        "category": example["category"],
                        "prompt": example["prompt"],
                        "output": output,
                        "score": result.score,
                    }
                )
            if progress:
                progress.update(len(batch))
                progress.set_postfix(batches=(start // batch_size) + 1)
    finally:
        if progress:
            progress.close()
    elapsed = time.perf_counter() - started

    metrics = summarize_scores(examples, score_results)
    metrics["evaluation_time_seconds"] = elapsed
    metrics["examples_per_second"] = len(examples) / elapsed if elapsed else None

    artifact_directory = create_output_directory(
        model_name=model_name,
        adapter_path=adapter_path,
        dataset_version=dataset_version,
        checksum=checksum,
        output_dir=output_dir,
    )
    generation_parameters = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
    }
    metadata = build_metadata(
        model_name=model_name,
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        dataset_version=dataset_version,
        checksum=checksum,
        generation_parameters=generation_parameters,
        system_prompt=system_prompt,
        chat_template=chat_template_information(tokenizer, model_name),
        output_directory=artifact_directory,
        selected_examples=len(examples),
        category=category,
        limit=limit,
        git_commit=git_commit_hash(),
    )
    _write_json(artifact_directory / "metadata.json", metadata)
    with (artifact_directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    _write_json(artifact_directory / "metrics.json", metrics)
    return artifact_directory, metrics
