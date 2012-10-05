"""
Test cases for the NaiveBayesClassifier.

:author: Pascal Hartig <phartig@rdrei.net>
"""

import unittest


class NaiveBayesClassifierTestCase(unittest.TestCase):
    def test_import(self):
        """Smokiest smoke test"""

        from twentiment.naivebayes import NaiveBayesClassifier
