import itertools
import unittest
from independent_audit import format_valid
from format_policy import answer_only


class FormatPolicyTests(unittest.TestCase):
    def test_allowed(self):
        for value in ['42', ' -1,024.50 ', '+.5', 'Answer: 42', 'FINAL ANSWER: -3.',
                      'The answer is 42.', r'\boxed{42}', r'$\boxed{-3.5}$',
                      r'$$\boxed{12}$$', r'\(\boxed{8}\)', r'\[\boxed{8}\]']:
            with self.subTest(value=value):
                self.assertTrue(answer_only(value))

    def test_rejected(self):
        for value in ['', 'No answer', '2 + 3 = 5', 'Answer: 4 or 5', '42 dollars',
                      'First add 2 and 3. The answer is 5.', '1,23', '4\n5',
                      r'\boxed{2+3}', r'$\boxed{4}$$', 'Answer: 42. Explanation:']:
            with self.subTest(value=value):
                self.assertFalse(answer_only(value))

    def test_independent_parser(self):
        for prefix, wrapper, value, suffix in itertools.product(
                ['', 'Answer:', 'FINAL ANSWER: ', 'The answer is ', 'bad '],
                ['{}', '${}$', '$${}$$', r'\({}\)', r'\[{}\]', r'\boxed{{{}}}', r'$\boxed{{{}}}$'],
                ['42', '-1,234.5', '+.5', '', '2+3', '3 4', '1,23', '42.', ' 42 ', r'\boxed{3}'],
                ['', '.', ' Explanation', '\n']):
            text = prefix + wrapper.format(value) + suffix
            with self.subTest(text=text):
                self.assertEqual(answer_only(text), format_valid(text))
