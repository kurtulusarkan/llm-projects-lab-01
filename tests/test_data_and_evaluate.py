import unittest

from datasets import Dataset

from lab_01.data import (
    format_sst2_example,
    make_sst2_experiment_splits,
    sst2_target,
    validate_messages,
)
from lab_01.evaluate import exact_match, parse_sst2_prediction, summarize_sst2_predictions


class DataAndEvaluationTest(unittest.TestCase):
    def test_validates_chat_messages(self):
        example = {"messages": [{"role": "user", "content": "Hello"}]}
        self.assertIs(validate_messages(example), example)

    def test_rejects_invalid_message_role(self):
        with self.assertRaises(ValueError):
            validate_messages({"messages": [{"role": "tool", "content": "Hello"}]})

    def test_exact_match_normalizes_outer_whitespace(self):
        self.assertEqual(exact_match([" answer ", "wrong"], ["answer", "right"]), 0.5)

    def test_sst2_targets_are_strict_json(self):
        self.assertEqual(sst2_target(0), '{"sentiment":"negative"}')
        self.assertEqual(format_sst2_example({"sentence": "Bad", "label": 1})["target"], '{"sentiment":"positive"}')

    def test_sst2_split_selection_is_deterministic_and_keeps_held_out_data(self):
        train = Dataset.from_dict(
            {"sentence": [f"train {index}" for index in range(20)], "label": [index % 2 for index in range(20)]}
        )
        held_out = Dataset.from_dict({"sentence": ["official validation"], "label": [1]})
        first = make_sst2_experiment_splits(train, held_out, train_size=5, validation_size=3, seed=7)
        second = make_sst2_experiment_splits(train, held_out, train_size=5, validation_size=3, seed=7)

        self.assertEqual(first["train"]["sentence"], second["train"]["sentence"])
        self.assertEqual(first["validation"]["sentence"], second["validation"]["sentence"])
        self.assertFalse(set(first["train"]["sentence"]) & set(first["validation"]["sentence"]))
        self.assertEqual(first["test"]["sentence"], ["official validation"])

    def test_sst2_json_metrics_count_invalid_outputs(self):
        predictions = [
            '{"sentiment":"positive"}',
            "not json",
            '{"sentiment":"neutral"}',
            '{"sentiment":"negative"}',
        ]
        metrics = summarize_sst2_predictions(predictions, [1, 0, 1, 0])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["invalid_json_count"], 1)
        self.assertEqual(metrics["invalid_label_count"], 1)
        self.assertEqual(metrics["total_examples"], 4)
        self.assertEqual(parse_sst2_prediction('{"sentiment":"positive"}'), ("positive", None))
