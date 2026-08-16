"""Tiny evaluation primitives for repeatable lab comparisons."""


def exact_match(predictions: list[str], references: list[str]) -> float:
    """Return normalized exact-match accuracy."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not references:
        return 0.0
    matches = sum(
        prediction.strip() == reference.strip()
        for prediction, reference in zip(predictions, references, strict=True)
    )
    return matches / len(references)
