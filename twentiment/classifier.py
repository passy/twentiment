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
    def from_file(cls, file):
        """Creates a new instance from the given file handle."""

        return cls.from_json(json.load(file))

    @classmethod
    def from_json(cls, json):
        """Creates a new instance from the given JSON data as dict data
        structure.
        """

        data = json['trainingData']
        pos_tweets = data['positive']
        neg_tweets = data['negative']

        tweets = (_extract_documents(data['positive'], 'positive') +
                  _extract_documents(data['negative'], 'negative'))

        training_set = [(extract_features(doc), label) for (doc, label)
                        in tweets]

        return cls.from_training_set(training_set)

    @classmethod
    def from_training_set(cls, training_set):
        """Creates a new instance from the given training set."""

        classifier = NaiveBayesClassifier.train(training_set)
        return cls(classifier)
