"""
Test cases for the NaiveBayesClassifier.

:author: Pascal Hartig <phartig@rdrei.net>
"""

import unittest


class NaiveBayesClassifierTestCase(unittest.TestCase):
    def test_import(self):
        """Smokiest smoke test"""

        from twentiment.naivebayes import NaiveBayesClassifier

    def test_train(self):
        """Test training phase"""

        from twentiment.naivebayes import NaiveBayesClassifier

        training_features = [
            ({'nice': True, 'pretty': True}, 'pos'),
            ({'ugly': True, 'bald': True}, 'neg')
        ]

        classifier = NaiveBayesClassifier.train(training_features)
        self.assertEqual(list(classifier._labels), ['neg', 'pos'])
