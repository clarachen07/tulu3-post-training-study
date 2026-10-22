import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import score, extract_answer


class ScoringTest(unittest.TestCase):
    def test_cases(self):
        cases = [
            ("We calculate 12 * 3 = 36. The answer is 1,234.", "1234", True),
            ("The answer is -12.", "-12", True),
            ("Final: +12", "12", True),
            ("The answer is -.5", "-0.50", True),
            ("12 then 5, so the answer is 7.0.", "7", True),
            ("", "0", False), ("I cannot solve it.", "1", False),
            ("The answer is 12. Later we obtain 13", "12", False),
            ("6,600", "6,600", True), ("0.00", "0", True)
        ]
        for text, gold, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(score(text, gold)[1], expected)

    def test_only_continuation(self):
        self.assertIsNone(extract_answer("No numerical answer"))

    def test_truncated_output_is_still_scored(self):
        self.assertEqual(score("3 + 4 = 7", "7"), ("7", True))


if __name__ == "__main__":
    unittest.main()
