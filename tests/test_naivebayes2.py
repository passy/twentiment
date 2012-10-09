"""
More elaborate tests for the naive bayes classifier.

:author: 2012, Pascal Hartig <phartig@weluse.de>
"""

from unittest import TestCase

from twentiment.thirdparty.probability import FreqDist
from twentiment.naivebayes import NaiveBayesClassifier
from twentiment.text import normalize_text


def get_word_features(wordlist):
    return FreqDist(wordlist).keys()


def extract_features(document):
    return {word: True for word in document}


class NaiveBayesTestCase2(TestCase):

    def setUp(self):
        pos_tweets = [('I love this car', 'positive'),
                    ('This view is amazing', 'positive'),
                    ('I feel great this morning', 'positive'),
                    ('I am so excited about the concert', 'positive'),
                    ('He is my best friend', 'positive')]

        neg_tweets = [('I do not like this car', 'negative'),
                    ('This view is horrible', 'negative'),
                    ('I feel tired this morning', 'negative'),
                    ('I am not looking forward to the concert', 'negative'),
                    ('He is my enemy', 'negative')]

        tweets = []
        for (words, sentiment) in pos_tweets + neg_tweets:
            tweets.append((normalize_text(words), sentiment))

        training_set = [(extract_features(doc), label) for (doc, label)
                        in tweets]

        self.classifier = NaiveBayesClassifier.train(training_set)

    def test_garbage(self):
        """Guessing garbage should is neutral"""

        twfeat = extract_features(normalize_text("goregho regeorg egewg"))
        prob_result = self.classifier.prob_classify(twfeat)
        score = prob_result.prob('positive') - prob_result.prob('negative')

        self.assertEqual(score, 0.0)

    def test_polarized(self):
        """Guessing polarized words shouldn't fail"""

        twfeat = extract_features(normalize_text("This car is my best friend "
                                                 "and enemy."))
        prob_result = self.classifier.prob_classify(twfeat)
        score = prob_result.prob('positive') - prob_result.prob('negative')

        self.assertTrue(score > 0, "Score is {}".format(score))

    def test_balance(self):
        """Balanced use of negative and positive words -> neutral"""

        twfeat = extract_features(normalize_text("friend and enemy"))
        prob_result = self.classifier.prob_classify(twfeat)
        score = prob_result.prob('positive') - prob_result.prob('negative')

        self.assertEqual(score, 0)
