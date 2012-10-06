"""
Test cases for the NaiveBayesClassifier.

:author: Pascal Hartig <phartig@rdrei.net>
"""

import unittest


class NaiveBayesClassifierTestCase(unittest.TestCase):
    def _get_classifier(self):
        from twentiment.naivebayes import NaiveBayesClassifier
        # from nltk.classify.naivebayes import NaiveBayesClassifier

        training_features = [
            ({'nice': True, 'pretty': True}, 'pos'),
            ({'ugly': True, 'bald': True}, 'neg')
        ]

        return NaiveBayesClassifier.train(training_features)

    def test_train(self):
        """Test training phase"""

        classifier = self._get_classifier()
        self.assertEqual(list(classifier._labels), ['neg', 'pos'])

    def test_prob_classify_pos(self):
        """NaiveBayesClassifier.prob_classify() -> 'pos'"""

        classifier = self._get_classifier()
        featureset = {'nice': True, 'pretty': True}

        result = classifier.prob_classify(featureset)
        self.assertEquals(result.max(), 'pos')
        self.assertTrue(result.logprob('pos') > result.logprob('neg'))

    def test_prob_classify_neg(self):
        """NaiveBayesClassifier.prob_classify() -> 'neg'"""

        classifier = self._get_classifier()
        featureset = {'ugly': True, 'bald': True}

        result = classifier.prob_classify(featureset)
        self.assertEquals(result.max(), 'neg')
        self.assertTrue(result.logprob('neg') > result.logprob('pos'))
