"""
Top level classifier based on top of NaiveBayesClassifier that
supports persistance and loading training data from the file system.

:author: 2012, Pascal Hartig <phartig@weluse.de>
:license: Apache 2
"""

import json
from twentiment.naivebayes import NaiveBayesClassifier
from twentiment.extract import extract_features
from twentiment.text import normalize_text


def _extract_documents(tweets, label):
    return [(normalize_text(tweet), label) for tweet in tweets]


def _limited_tweet_split(json, limit=0):
    data = json['trainingData']
    positive, negative = data['positive'], data['negative']

    if limit > 1:
        return positive[:limit // 2], negative[:limit // 2]
    else:
        return positive, negative


class Classifier:

    def __init__(self, naive_bayes_classifier):
        """
        Instantiates a new classifier with a trained naive bayes classifier.
        To train, utilize the :meth:`from_file` or :meth:`from_json` factory
        methods.

        :param naive_bayes_classifier: Trained instance of
            :cls:`~twentiment.naivebayes.NaiveBayesClassifier`.
        """

        self.classifier = naive_bayes_classifier

        def _proxy_classifier_method(method):
            return lambda *a, **kw: getattr(self.classifier, method)(*a, **kw)

        # Proxy some methods to the classifier
        for method in ['prob_classify', 'classify']:
            setattr(self, method, _proxy_classifier_method(method))
            setattr(getattr(self, method), '__doc__',
                    getattr(self.classifier, method).__doc__)

    @classmethod
    def from_file(cls, file, *args, **kwargs):
        """Creates a new instance from the given file handle."""

        return cls.from_json(json.load(file), *args, **kwargs)

    @classmethod
    def from_json(cls, json, max_entries=0):
        """Creates a new instance from the given JSON data as dict data
        structure.

        :param max_entries: Limit training set to a maximum of ``max_entries``
            items. This can be helpful to reduce memory usage. A value of 0 or
            less means no limit.
        """

        pos_tweets, neg_tweets = _limited_tweet_split(json, max_entries)

        tweets = (_extract_documents(pos_tweets, 'positive') +
                  _extract_documents(neg_tweets, 'negative'))

        training_set = [(extract_features(doc), label) for (doc, label)
                        in tweets]

        return cls.from_training_set(training_set)

    @classmethod
    def from_training_set(cls, training_set):
        """Creates a new instance from the given training set."""

        classifier = NaiveBayesClassifier.train(training_set)
        return cls(classifier)
