import unittest

from lab_01.data import validate_messages
from lab_01.evaluate import exact_match


class DataAndEvaluationTest(unittest.TestCase):
    def test_validates_chat_messages(self):
        example = {"messages": [{"role": "user", "content": "Hello"}]}
        self.assertIs(validate_messages(example), example)

    def test_rejects_invalid_message_role(self):
        with self.assertRaises(ValueError):
            validate_messages({"messages": [{"role": "tool", "content": "Hello"}]})

    def test_exact_match_normalizes_outer_whitespace(self):
        self.assertEqual(exact_match([" answer ", "wrong"], ["answer", "right"]), 0.5)
